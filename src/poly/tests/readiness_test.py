"""Tests for the continuous-assurance readiness check.

Exercises the ``AgentStudioProject.assess_readiness`` wiring (in poly.project)
and the underlying dependency-map / diagnostics in poly.readiness.

Copyright PolyAI Limited
"""

import unittest

from poly.project import AgentStudioProject
from poly.resources import (
    FlowConfig,
    Function,
    FunctionCallAssertion,
    ResourceMapping,
    TestCase,
    TestCaseAssertion,
    TestCaseTags,
)
from poly.readiness import assess_readiness, build_dependency_map


def _mapping(
    resource_id: str,
    resource_type: type,
    name: str,
    prefix: str,
    flow_name: str | None = None,
    file_path: str | None = None,
) -> ResourceMapping:
    """Build a ResourceMapping with sensible defaults for tests."""
    return ResourceMapping(
        resource_id=resource_id,
        resource_type=resource_type,
        resource_name=name,
        file_path=file_path or f"{prefix}/{name}.yaml",
        flow_name=flow_name,
        resource_prefix=prefix,
    )


def _test_case(name: str, calls: list[str]) -> TestCase:
    """Build a TestCase whose assertions call the given function names."""
    return TestCase(
        resource_id=f"TEST-{name}",
        name=name,
        scenario=f"scenario for {name}",
        channel="webchat.polyai",
        language="en",
        assertions=TestCaseAssertion(
            resource_id=f"TEST-{name}",
            name=name,
            prompts=[],
            function_calls=[FunctionCallAssertion(name=call, arguments=[]) for call in calls],
        ),
        tags=TestCaseTags(resource_id=f"TAGS-{name}", name=name),
    )


class AssessReadinessTest(unittest.TestCase):
    """Tests for readiness finding detection over the dependency map."""

    def setUp(self) -> None:
        """Build a project graph with two dangling references and two valid ones."""
        self.resources_dict: dict[type, dict[str, object]] = {
            TestCase: {
                "TEST-case1": _test_case("case1", calls=["ghost_fn", "real_fn"]),
            },
        }
        self.resource_mappings = [
            _mapping("TEST-case1", TestCase, "case1", "test", file_path="test_suite/case1.yaml"),
            _mapping("FUNCTION-real_fn", Function, "real_fn", "fn"),
            _mapping("FUNCTION-in_flow", Function, "in_flow", "fn", flow_name="real_flow"),
            _mapping("FUNCTION-orphan", Function, "orphan_fn", "fn", flow_name="ghost_flow"),
            _mapping("FLOW-real_flow", FlowConfig, "real_flow", "flow"),
        ]

    def test_flags_dangling_test_to_function_reference(self) -> None:
        """A test calling an undeclared function is reported with remediation."""
        findings = assess_readiness(self.resources_dict, self.resource_mappings)
        ghost = [f for f in findings if f.target == "fn:ghost_fn"]
        self.assertEqual(len(ghost), 1)
        self.assertEqual(ghost[0].severity, "error")
        self.assertIn("ghost_fn", ghost[0].remediation)
        self.assertIn("Add the missing fn resource", ghost[0].remediation)

    def test_does_not_flag_resolved_references(self) -> None:
        """References to declared functions/flows produce no finding."""
        findings = assess_readiness(self.resources_dict, self.resource_mappings)
        targets = {f.target for f in findings}
        self.assertNotIn("fn:real_fn", targets)
        self.assertNotIn("flow:real_flow", targets)

    def test_flags_dangling_function_to_flow_reference(self) -> None:
        """A function belonging to an undeclared flow is reported."""
        findings = assess_readiness(self.resources_dict, self.resource_mappings)
        orphan = [f for f in findings if f.target == "flow:ghost_flow"]
        self.assertEqual(len(orphan), 1)
        self.assertIn("belongs_to", orphan[0].relation)

    def test_finding_count_matches_dangling_references(self) -> None:
        """Exactly the two dangling references are reported, no more."""
        findings = assess_readiness(self.resources_dict, self.resource_mappings)
        self.assertEqual(len(findings), 2)

    def test_clean_graph_has_no_findings(self) -> None:
        """A fully consistent graph yields an empty readiness report."""
        clean_mappings = [
            _mapping("TEST-case1", TestCase, "case1", "test", file_path="test_suite/case1.yaml"),
            _mapping("FUNCTION-real_fn", Function, "real_fn", "fn"),
            _mapping("FUNCTION-in_flow", Function, "in_flow", "fn", flow_name="real_flow"),
            _mapping("FLOW-real_flow", FlowConfig, "real_flow", "flow"),
        ]
        clean_resources: dict[type, dict[str, object]] = {
            TestCase: {"TEST-case1": _test_case("case1", calls=["real_fn"])},
        }
        findings = assess_readiness(clean_resources, clean_mappings)
        self.assertEqual(findings, [])


class DependencyMapTest(unittest.TestCase):
    """Tests for the dependency-map edge extraction."""

    def test_map_contains_test_to_function_and_function_to_flow_edges(self) -> None:
        """Edges cover test->function and function->flow relations."""
        resources_dict: dict[type, dict[str, object]] = {
            TestCase: {"TEST-case1": _test_case("case1", calls=["real_fn", "ghost_fn"])},
        }
        resource_mappings = [
            _mapping("TEST-case1", TestCase, "case1", "test"),
            _mapping("FUNCTION-real_fn", Function, "real_fn", "fn"),
            _mapping("FUNCTION-in_flow", Function, "in_flow", "fn", flow_name="real_flow"),
            _mapping("FLOW-real_flow", FlowConfig, "real_flow", "flow"),
        ]
        edges = build_dependency_map(resources_dict, resource_mappings)
        edge_strs = {str(edge) for edge in edges}
        self.assertIn("test:case1 --calls--> fn:real_fn", edge_strs)
        self.assertIn("test:case1 --calls--> fn:ghost_fn", edge_strs)
        self.assertIn("fn:in_flow --belongs_to--> flow:real_flow", edge_strs)


class ProjectWiringTest(unittest.TestCase):
    """The non-new call-site module (poly.project) wires the readiness check."""

    def test_project_assess_readiness_delegates_to_module(self) -> None:
        """AgentStudioProject.assess_readiness returns the module's findings."""
        resources_dict: dict[type, dict[str, object]] = {
            TestCase: {"TEST-case1": _test_case("case1", calls=["ghost_fn"])},
        }
        resource_mappings = [
            _mapping("TEST-case1", TestCase, "case1", "test"),
            _mapping("FUNCTION-real_fn", Function, "real_fn", "fn"),
        ]
        # Exercising the wiring edit in project.py (the call site).
        findings = AgentStudioProject.assess_readiness(resources_dict, resource_mappings)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].target, "fn:ghost_fn")


if __name__ == "__main__":
    unittest.main()
