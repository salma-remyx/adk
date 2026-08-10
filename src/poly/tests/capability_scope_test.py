"""Tests for the dynamic capability-scope analyzer.

These tests exercise the analyzer logic directly against constructed
``Function`` resources and lightweight reference-surface stubs. The
end-to-end wiring (``AgentStudioProject.validate_project``) is covered by
``CapabilityScopeIntegrationTest`` in ``project_test.py``.

Adapted from arXiv:2607.22445v1 (Mode 2).
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from poly.capability_scope import (
    evaluate_capability_scope,
    find_capability_scope_findings,
)
from poly.resources.flows import FlowStep
from poly.resources.function import Function, FunctionType
from poly.resources.agent_settings import SettingsRules
from poly.resources.topic import Topic


def _global_function(name: str, resource_id: str = None) -> Function:
    """Build a minimal GLOBAL function for analysis."""
    return Function(
        resource_id=resource_id or f"FUNCTION-{name}",
        name=name,
        description=f"{name} description",
        code=f"def {name}(conv: Conversation):\n    return None",
        parameters=[],
        latency_control={},
        function_type=FunctionType.GLOBAL,
    )


class FindCapabilityScopeFindingsTest(unittest.TestCase):
    """Tests for the over-privilege analysis logic."""

    def _resources(self, functions, prompt="", behaviour="", topic_actions="", queries=None):
        return {
            Function: {fn.resource_id: fn for fn in functions},
            FlowStep: {"s1": SimpleNamespace(prompt=prompt)},
            SettingsRules: {"r1": SimpleNamespace(behaviour=behaviour)},
            Topic: {"t1": SimpleNamespace(actions=topic_actions, example_queries=queries or [])},
        }

    def test_unreferenced_global_function_flagged(self):
        used = _global_function("used_fn")
        unused = _global_function("orphan_fn")
        resources = self._resources(
            [used, unused], prompt="Call {{fn:FUNCTION-used_fn}} now"
        )
        findings = find_capability_scope_findings(resources)
        flagged = {f.capability for f in findings}
        self.assertIn("orphan_fn", flagged)
        self.assertNotIn("used_fn", flagged)
        # The over-privilege finding is attributed to the task-context source.
        self.assertTrue(
            any(f.source == "task_context" and f.capability == "orphan_fn" for f in findings)
        )

    def test_all_referenced_functions_yield_no_task_context_finding(self):
        a = _global_function("alpha")
        b = _global_function("beta")
        resources = self._resources(
            [a, b],
            prompt="Use {{fn:alpha}}",
            behaviour="Then use {{fn:beta}}",
        )
        findings = [f for f in find_capability_scope_findings(resources) if f.source == "task_context"]
        self.assertEqual(findings, [])

    def test_reference_by_name_form_is_resolved(self):
        """Tokens may carry the bare name before name->id resolution."""
        fn = _global_function("named_fn")
        resources = self._resources([fn], topic_actions="Trigger {{fn:named_fn}}")
        findings = {
            f.capability for f in find_capability_scope_findings(resources) if f.source == "task_context"
        }
        self.assertNotIn("named_fn", findings)

    def test_topic_example_query_references_count(self):
        fn = _global_function("queried_fn")
        resources = self._resources(
            [fn], queries=["How do I run {{fn:queried_fn}}?"]
        )
        findings = {
            f.capability for f in find_capability_scope_findings(resources) if f.source == "task_context"
        }
        self.assertNotIn("queried_fn", findings)

    def test_non_global_functions_not_analyzed(self):
        start = Function(
            resource_id="start",
            name="start_function",
            description="",
            code="def start_function(conv: Conversation):\n    return None",
            parameters=[],
            latency_control={},
            function_type=FunctionType.START,
        )
        transition = Function(
            resource_id="t-1",
            name="do_transition",
            description="d",
            code="def do_transition(conv: Conversation, flow: Flow):\n    return None",
            parameters=[],
            latency_control={},
            function_type=FunctionType.TRANSITION,
            flow_id="FLOW-1",
            flow_name="flow_one",
        )
        resources = self._resources([start, transition])
        findings = find_capability_scope_findings(resources)
        self.assertEqual(findings, [])

    def test_policy_prohibition_keyword_flags_matching_function(self):
        safe = _global_function("read_record")
        dangerous = _global_function("delete_all_records")
        resources = self._resources(
            [safe, dangerous],
            prompt="Use {{fn:read_record}} and {{fn:delete_all_records}}",
        )
        with patch.dict(os.environ, {"POLY_CAPABILITY_SCOPE_DENY": "delete_all"}):
            findings = find_capability_scope_findings(resources)
        prohibited = {f.capability for f in findings if f.source == "policy_prohibition"}
        self.assertEqual(prohibited, {"delete_all_records"})
        # No task_context finding because both are referenced above.
        self.assertFalse(any(f.source == "task_context" for f in findings))


class EvaluateCapabilityScopeModeTest(unittest.TestCase):
    """Tests for the observe-only / enforcing deployment-mode switch."""

    def _resources_with_orphan(self):
        return {
            Function: {"FUNCTION-orphan": _global_function("orphan")},
            FlowStep: {},
            SettingsRules: {},
            Topic: {},
        }

    def test_observe_only_logs_and_does_not_block(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("POLY_CAPABILITY_SCOPE_ENFORCE", None)
            messages = evaluate_capability_scope(self._resources_with_orphan())
        self.assertEqual(messages, [])

    def test_enforce_mode_returns_blocking_messages(self):
        with patch.dict(os.environ, {"POLY_CAPABILITY_SCOPE_ENFORCE": "1"}):
            messages = evaluate_capability_scope(self._resources_with_orphan())
        self.assertEqual(len(messages), 1)
        self.assertIn("orphan", messages[0])
        self.assertTrue(messages[0].startswith("Capability scope:"))

    def test_no_findings_returns_empty_even_in_enforce_mode(self):
        referenced = _global_function("used")
        resources = {
            Function: {referenced.resource_id: referenced},
            FlowStep: {"s": SimpleNamespace(prompt="{{fn:used}}")},
            SettingsRules: {},
            Topic: {},
        }
        with patch.dict(os.environ, {"POLY_CAPABILITY_SCOPE_ENFORCE": "1"}):
            self.assertEqual(evaluate_capability_scope(resources), [])


if __name__ == "__main__":
    unittest.main()
