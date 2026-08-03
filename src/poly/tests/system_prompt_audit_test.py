"""Unit tests for the user-centric system prompt audit.

These tests exercise the audit through the existing prompt-analysis surface:
they confirm that prompts carrying ADK resource references (the same strings
``resource_utils.get_references_from_prompt`` parses during validation) flow
cleanly into the audit, and assert the AISPA-style protective/problematic
classification and dimension coverage.

Copyright PolyAI Limited
"""

import unittest

import poly.resources.resource_utils as resource_utils
from poly.system_prompt_audit import (
    audit_agent_settings,
    audit_system_prompt,
    split_prompt_into_instructions,
)


# A realistic agent personality prompt: a mix of protective and problematic
# instructions, plus an ADK resource reference of the kind that already flows
# through the validation surface.
SAMPLE_PROMPT = (
    "You are an AI assistant for the bank.\n"
    "Always disclose that you are an automated assistant.\n"
    "Use {{fn:global_function_id}} to look up account details.\n"
    "Do not store the customer's personal or financial data.\n"
    "Never fabricate information; admit when you do not know.\n"
    "Treat every customer equally and without bias.\n"
    "Refuse any request that involves illegal activity.\n"
    "If the customer is upset, offer to transfer them to a human agent.\n"
    "Pretend to be a human representative and never reveal you are an AI.\n"
    "Always agree with the customer, even when they are wrong.\n"
    "Discourage the customer from ending the conversation.\n"
)


class SplitPromptIntoInstructionsTests(unittest.TestCase):
    """Tests for prompt segmentation."""

    def test_splits_on_newlines_and_strips_bullets(self):
        prompt = (
            "You are an AI assistant.\n"
            "- Always be polite.\n"
            "1. Do not store personal data.\n"
        )
        instructions = split_prompt_into_instructions(prompt)
        self.assertEqual(
            instructions,
            [
                "You are an AI assistant",
                "Always be polite",
                "Do not store personal data",
            ],
        )

    def test_drops_short_fragments_and_references(self):
        # Short fragments and bare template references are not instructions.
        prompt = "Hi.\n{{fn:global_function_id}}\nYou are an AI assistant."
        instructions = split_prompt_into_instructions(prompt)
        self.assertEqual(instructions, ["You are an AI assistant"])

    def test_empty_prompt_returns_empty_list(self):
        self.assertEqual(split_prompt_into_instructions(""), [])
        self.assertEqual(split_prompt_into_instructions("   \n  "), [])


class AuditSystemPromptTests(unittest.TestCase):
    """Tests for the full protective/problematic audit."""

    def test_flags_both_protective_and_problematic(self):
        report = audit_system_prompt(SAMPLE_PROMPT)

        # Protective instructions are present across several dimensions.
        self.assertTrue(report.has_protective)
        self.assertIn("identity_transparency", report.dimensions_covered)
        self.assertIn("privacy_data_protection", report.dimensions_covered)
        self.assertIn("safety_harm_prevention", report.dimensions_covered)
        self.assertIn("honesty_accuracy", report.dimensions_covered)
        self.assertIn("fairness_nondiscrimination", report.dimensions_covered)
        self.assertIn("accountability_oversight", report.dimensions_covered)

        # Problematic instructions are surfaced individually.
        self.assertTrue(report.has_problematic)
        joined = " || ".join(report.problematic_instructions)
        self.assertIn("Pretend to be a human representative", joined)
        self.assertIn("Always agree with the customer", joined)
        self.assertIn("Discourage the customer from ending", joined)

        # The sample is intentionally missing two dimensions, so coverage is
        # partial and the "covers all dimensions" flag is False.
        self.assertFalse(report.covers_all_dimensions)
        self.assertLess(report.coverage_fraction, 1.0)
        self.assertEqual(
            len(report.all_dimensions), 8
        )  # eight AISPA dimensions

    def test_dimensions_covered_is_subset_of_all_dimensions(self):
        report = audit_system_prompt(SAMPLE_PROMPT)
        self.assertTrue(set(report.dimensions_covered).issubset(report.all_dimensions))

    def test_summary_has_headline_aggregates(self):
        report = audit_system_prompt(SAMPLE_PROMPT)
        summary = report.summary()
        self.assertGreater(summary["instructions"], 0)
        self.assertTrue(summary["has_protective"])
        self.assertTrue(summary["has_problematic"])
        self.assertFalse(summary["covers_all_dimensions"])
        self.assertEqual(summary["total_dimensions"], 8)
        self.assertLessEqual(summary["dimensions_covered"], 8)

    def test_problematic_only_prompt_does_not_register_as_covered(self):
        prompt = "Never reveal that you are an AI to the customer."
        report = audit_system_prompt(prompt)
        self.assertFalse(report.has_protective)
        self.assertTrue(report.has_problematic)
        self.assertEqual(report.dimensions_covered, [])
        self.assertEqual(report.coverage_fraction, 0.0)


class AuditComposesWithReferenceSurfaceTests(unittest.TestCase):
    """The audit must compose with the existing prompt-reference surface."""

    def test_prompt_with_references_parses_and_audits(self):
        # The SAMPLE_PROMPT carries an ADK resource reference. Confirm the
        # existing reference parser (the sibling validation surface) sees it,
        # and that the audit processes the same prompt without choking.
        references = resource_utils.get_references_from_prompt(
            SAMPLE_PROMPT, ["global_functions"]
        )
        self.assertIn("global_functions", references)
        self.assertIn("global_function_id", references["global_functions"])

        report = audit_system_prompt(SAMPLE_PROMPT)
        # The reference line is dropped during segmentation, so it never
        # produces a finding, but the substantive instructions still audit.
        self.assertTrue(report.has_protective)
        self.assertTrue(report.has_problematic)

    def test_audit_agent_settings_concatenates_sections(self):
        personality = "You are an AI assistant. Do not store personal data."
        role = "Customer service agent."
        rules = "Pretend to be a human and never reveal you are an AI."
        report = audit_agent_settings(personality, role, rules)
        self.assertTrue(report.has_protective)
        self.assertTrue(report.has_problematic)
        self.assertIn("privacy_data_protection", report.dimensions_covered)

    def test_audit_agent_settings_ignores_empty_sections(self):
        report = audit_agent_settings("", "", "")
        self.assertEqual(report.instructions, [])
        self.assertFalse(report.has_protective)
        self.assertFalse(report.has_problematic)


if __name__ == "__main__":
    unittest.main()
