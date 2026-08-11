"""Integration tests for validation-dimension coverage analysis.

Copyright PolyAI Limited
"""

import unittest

from poly.resources.test_suite import (
    FunctionCallArgumentAssertion,
    FunctionCallAssertion,
    TestCase,
    TestCaseAssertion,
    TestCaseTags,
)
from poly.validation_coverage import (
    DIMENSIONS,
    analyze_coverage,
)


def _make_test_case(
    *,
    name: str,
    scenario: str = "A basic behavioural check.",
    tags: list[str] | None = None,
    prompts: list[str] | object = None,
    function_calls: list[FunctionCallAssertion] | object = None,
) -> TestCase:
    """Build a TestCase exercising the existing test_suite data contract.

    ``prompts`` / ``function_calls`` default to a single behavioural assertion;
    pass an empty list explicitly to build an assertion-less case.
    """
    resource_id = f"TEST-{name}"
    if prompts is None and function_calls is None:
        prompts = ["The agent responds appropriately"]
    return TestCase(
        resource_id=resource_id,
        name=name,
        scenario=scenario,
        channel="chat.polyai",
        assertions=TestCaseAssertion(
            resource_id=resource_id,
            name="assertions",
            prompts=prompts if isinstance(prompts, list) else [],
            function_calls=function_calls if isinstance(function_calls, list) else [],
        ),
        tags=TestCaseTags(resource_id=resource_id, name="tags", tags=tags or []),
        language="en-GB",
    )


class AnalyzeCoverageTests(unittest.TestCase):
    """Tests driving analyze_coverage through the existing TestCase contract."""

    def test_behavioral_only_case_leaves_four_gaps(self) -> None:
        report = analyze_coverage([_make_test_case(name="Greeting")])

        self.assertEqual(report.total, 1)
        self.assertEqual(report.covered_dimensions, ["behavioral"])
        self.assertEqual(
            report.gap_dimensions,
            ["safety", "temporal", "regulatory", "multi_agent"],
        )

    def test_safety_keyword_in_scenario_is_detected(self) -> None:
        case = _make_test_case(
            name="Pii refusal",
            scenario="User shares PII; agent must refuse and redact.",
        )
        report = analyze_coverage([case])

        self.assertIn("safety", report.covered_dimensions)

    def test_temporal_keyword_in_tags_is_detected(self) -> None:
        case = _make_test_case(
            name="Resume after pause",
            scenario="Conversation resume after a long pause.",
            tags=["resume", "stateful"],
        )
        report = analyze_coverage([case])

        self.assertIn("temporal", report.covered_dimensions)

    def test_regulatory_keyword_is_detected(self) -> None:
        case = _make_test_case(
            name="Gdpr opt-out",
            scenario="User requests GDPR opt-out; agent confirms.",
        )
        report = analyze_coverage([case])

        self.assertIn("regulatory", report.covered_dimensions)

    def test_multi_agent_function_call_is_detected(self) -> None:
        handoff = FunctionCallAssertion(
            name="handoff_to_human",
            arguments=[
                FunctionCallArgumentAssertion(
                    parameter_name="reason",
                    expected_value="escalation",
                    value_type="string",
                )
            ],
        )
        case = _make_test_case(
            name="Escalate to human",
            scenario="Agent escalates to a live agent.",
            function_calls=[handoff],
        )
        report = analyze_coverage([case])

        self.assertIn("multi_agent", report.covered_dimensions)
        # The function-call assertion also counts as behavioural coverage.
        self.assertIn("behavioral", report.covered_dimensions)

    def test_assertionless_case_covers_nothing(self) -> None:
        case = _make_test_case(
            name="No assertions",
            scenario="Smoke test with no expectations yet.",
            prompts=[],
            function_calls=[],
        )
        report = analyze_coverage([case])

        self.assertEqual(report.covered_dimensions, [])

    def test_from_yaml_dict_round_trip_then_classify(self) -> None:
        """Classification works on TestCases built via the existing YAML parser."""
        yaml_dict = {
            "name": "Timeout recovery",
            "scenario": "Agent recovers after a timeout.",
            "channel": "voice",
            "language": "en-GB",
            "tags": ["timeout", "retry"],
            "prompt_assertions": ["The agent apologises for the delay"],
        }
        case = TestCase.from_yaml_dict(
            yaml_dict, resource_id="TEST-timeout", name="assertions"
        )
        report = analyze_coverage([case])

        self.assertIn("temporal", report.covered_dimensions)
        self.assertIn("behavioral", report.covered_dimensions)

    def test_aggregate_buckets_multiple_cases_per_dimension(self) -> None:
        cases = [
            _make_test_case(name="A", scenario="Agent stays safe."),
            _make_test_case(name="B", tags=["guardrail"]),
        ]
        report = analyze_coverage(cases)

        self.assertEqual(report.total, 2)
        self.assertEqual(set(report.dimensions["safety"].covered), {"A", "B"})
        self.assertEqual(report.dimensions["safety"].count, 2)
        self.assertTrue(report.dimensions["safety"].is_covered)

    def test_empty_suite_reports_all_gaps(self) -> None:
        report = analyze_coverage([])

        self.assertEqual(report.total, 0)
        self.assertEqual(report.gap_dimensions, list(DIMENSIONS))
        self.assertEqual(report.covered_dimensions, [])

    def test_render_and_to_dict_shape(self) -> None:
        report = analyze_coverage([_make_test_case(name="Solo")])
        rendered = report.render()

        self.assertIn("Validation-dimension coverage", rendered)
        self.assertIn("[covered] behavioral", rendered)
        self.assertIn("[GAP] safety", rendered)
        self.assertIn("Coverage gaps:", rendered)

        as_dict = report.to_dict()
        self.assertEqual(as_dict["total"], 1)
        self.assertEqual(set(as_dict["dimensions"]), set(DIMENSIONS))
        self.assertIn("safety", as_dict["gaps"])


if __name__ == "__main__":
    unittest.main()
