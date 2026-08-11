"""Tests for the tool-specification safety analyzer (SafeKeep adapted port).

Covers both the analyzer unit behavior and the wiring into
``AgentStudioProject.validate_project`` (the validate-pipeline call site).
"""

import os
import unittest
from unittest.mock import patch

from poly.project import AgentStudioProject
from poly.resources.function import Function, FunctionParameters, FunctionType
from poly.tests.testing_utils import mock_read_from_file
from poly.tool_spec_safety import (
    analyze_tool_spec_safety,
    flatten_tool_spec,
    is_schema_dominated,
)

DIR = os.path.dirname(os.path.abspath(__file__))
TEST_PROJECT_DIR = os.path.join(DIR, "test_projects")
TEST_DIR = os.path.join(TEST_PROJECT_DIR, "test_project")
PROJECT_DATA_LOC = os.path.join(TEST_DIR, "test_project.json")

import json  # noqa: E402

PROJECT_DATA = json.loads(open(PROJECT_DATA_LOC).read())


def _make_function(
    description: str,
    parameters: list[FunctionParameters] | None = None,
    name: str = "send_message",
) -> Function:
    """Build a Function in-memory without touching disk."""
    return Function(
        resource_id=f"FUNCTION-{name}",
        name=name,
        description=description,
        code="def f(): pass",
        parameters=parameters or [],
        latency_control={},
        function_type=FunctionType.GLOBAL,
    )


class FlattenToolSpecTest(unittest.TestCase):
    """Unit tests for the SafeKeep schema -> text flattening."""

    def test_flatten_function_renders_description_and_parameters(self):
        function = _make_function(
            "Send a message to a recipient.",
            parameters=[
                FunctionParameters(name="to", type="string", description="recipient"),
                FunctionParameters(name="body", type="string", description=""),
            ],
        )
        flattened = flatten_tool_spec(function)
        self.assertIn("Send a message to a recipient.", flattened)
        self.assertIn("- to (string): recipient", flattened)
        # A parameter with no description still appears by name + type.
        self.assertIn("- body (string)", flattened)

    def test_flatten_function_with_empty_description(self):
        function = _make_function(
            "",
            parameters=[FunctionParameters(name="q", type="string", description="query")],
        )
        self.assertEqual(flatten_tool_spec(function), "- q (string): query")

    def test_flatten_rejects_non_tool_resource(self):
        with self.assertRaises(TypeError):
            flatten_tool_spec("not a tool spec")  # type: ignore[arg-type]


class IsSchemaDominatedTest(unittest.TestCase):
    """Unit tests for the parameter-free risk-condition proxy."""

    def test_empty_description_with_parameters_is_flagged(self):
        function = _make_function(
            "",
            parameters=[FunctionParameters(name="p", type="string", description="p")],
        )
        self.assertTrue(is_schema_dominated(function))

    def test_thin_description_with_parameters_is_flagged(self):
        function = _make_function(
            "Sends.",  # one word
            parameters=[FunctionParameters(name="p", type="string", description="p")],
        )
        self.assertTrue(is_schema_dominated(function))

    def test_grounded_description_is_not_flagged(self):
        function = _make_function(
            "Sends a message to the configured recipient channel.",
            parameters=[FunctionParameters(name="p", type="string", description="p")],
        )
        self.assertFalse(is_schema_dominated(function))

    def test_parameterless_function_is_not_flagged(self):
        function = _make_function("Do a thing.", parameters=[])
        self.assertFalse(is_schema_dominated(function))


class AnalyzeToolSpecSafetyTest(unittest.TestCase):
    """Unit tests for the analyzer over a resource map."""

    def test_returns_finding_for_schema_dominated_function(self):
        function = _make_function(
            "",
            parameters=[FunctionParameters(name="p", type="string", description="p")],
        )
        findings = analyze_tool_spec_safety({Function: {"FUNCTION-send_message": function}})
        self.assertEqual(len(findings), 1)
        finding = findings[0]
        self.assertEqual(finding.resource_type, "function")
        self.assertEqual(finding.resource_name, "send_message")
        self.assertIn("schema-dominated", finding.message)
        self.assertIn("- p (string)", finding.flattened_spec)

    def test_no_finding_for_grounded_function(self):
        function = _make_function(
            "This helper sends a message to a recipient safely.",
            parameters=[FunctionParameters(name="p", type="string", description="p")],
        )
        self.assertEqual(
            analyze_tool_spec_safety({Function: {"FUNCTION-send_message": function}}),
            [],
        )

    def test_ignores_non_tool_resources(self):
        class NotATool:
            pass

        self.assertEqual(analyze_tool_spec_safety({NotATool: {"x": NotATool()}}), [])


class ValidateProjectToolSpecSafetyTest(unittest.TestCase):
    """Integration test: validate_project wires in the analyzer (call site)."""

    def test_validate_project_invokes_tool_spec_safety_analyzer(self):
        """validate_project calls the analyzer without breaking the error contract."""
        project = AgentStudioProject.from_dict(PROJECT_DATA, TEST_DIR)
        with patch(
            "poly.project.analyze_tool_spec_safety",
            wraps=analyze_tool_spec_safety,
        ) as spy:
            errors = project.validate_project()
        # Advisory pass never adds blocking errors.
        self.assertEqual(errors, [])
        # The wiring edit actually invoked the analyzer.
        self.assertGreaterEqual(spy.call_count, 1)

    def test_validate_project_warns_on_schema_dominated_spec(self):
        """A schema-dominated tool spec surfaces a non-blocking advisory warning."""
        project = AgentStudioProject.from_dict(PROJECT_DATA, TEST_DIR)
        thin_function_source = (
            "from _gen import *  # <AUTO GENERATED>\n\n\n"
            '@func_description("")\n'
            '@func_parameter("param1", "First parameter as string")\n'
            "def test_function_with_parameters(conv: Conversation, param1: str):\n"
            '    """Thin description."""\n'
            '    return ""\n'
        )
        function_path = os.path.join(
            TEST_DIR, "functions", "test_function_with_parameters.py"
        )
        with mock_read_from_file({function_path: thin_function_source}):
            with self.assertLogs("poly.project", level="WARNING") as captured:
                project.validate_project()
        self.assertTrue(
            any("schema-dominated" in message for message in captured.output),
            f"expected a tool-spec safety advisory, got: {captured.output}",
        )


if __name__ == "__main__":
    unittest.main()
