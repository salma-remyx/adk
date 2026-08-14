"""Tests for multidimensional execution metrics on chat conversations.

Covers the scorer directly and the wiring into AgentStudioCLI.chat via the
--metrics flag (which enables metadata capture and enriches the conversation
with a 'metrics' key).

Copyright PolyAI Limited
"""

import unittest
from unittest.mock import MagicMock, patch

from poly.cli import AgentStudioCLI
from poly.execution_metrics import score_conversation

TEST_DIR = "/tmp/test_project"


def _turn(
    input_text="hi",
    response="ok",
    function_events=None,
    flow=None,
    state_changes=None,
    error=None,
):
    turn = {"input": input_text}
    if response is not None:
        turn["response"] = response
    if function_events is not None:
        turn["function_events"] = function_events
    if flow is not None:
        turn["flow"] = flow
    if state_changes is not None:
        turn["state_changes"] = state_changes
    if error is not None:
        turn["error"] = error
    return turn


class ScoreConversationTest(unittest.TestCase):
    """Tests for poly.execution_metrics.score_conversation."""

    def test_empty_conversation_has_zero_metrics(self):
        """A conversation without turns scores all-zero without dividing by zero."""
        metrics = score_conversation({"turns": []}).to_dict()

        self.assertEqual(metrics["turns_total"], 0)
        self.assertEqual(metrics["error_rate"], 0)
        self.assertEqual(metrics["tool_use"]["function_calls_per_turn"], 0)

    def test_function_events_counted_per_turn_and_distinct(self):
        """Tool calls are counted in total and de-duplicated by function name."""
        conversation = {
            "turns": [
                _turn(function_events=[{"name": "get_weather"}, {"name": "get_weather"}]),
                _turn(function_events=[{"name": "book_table"}]),
            ]
        }

        metrics = score_conversation(conversation).to_dict()

        self.assertEqual(metrics["tool_use"]["function_calls_total"], 3)
        self.assertEqual(metrics["tool_use"]["function_calls_per_turn"], 1.5)
        self.assertEqual(metrics["tool_use"]["distinct_functions"], ["book_table", "get_weather"])

    def test_flow_step_transitions_counted_as_planning_steps(self):
        """Each distinct flow/step transition is a planning step; repeats are not."""
        conversation = {
            "turns": [
                _turn(flow={"in_flow": "greeting", "in_step": "ask_name"}),
                _turn(flow={"in_flow": "greeting", "in_step": "ask_name"}),
                _turn(flow={"in_flow": "booking", "in_step": "confirm"}),
            ]
        }

        metrics = score_conversation(conversation).to_dict()

        self.assertEqual(metrics["planning"]["planning_steps"], 2)
        self.assertEqual(metrics["planning"]["flows_entered"], ["greeting", "booking"])

    def test_transport_error_without_response_counts_unrecovered(self):
        """A turn with an error key and no response is an unrecovered error turn."""
        conversation = {"turns": [_turn(response=None, error="500")]}

        metrics = score_conversation(conversation).to_dict()

        self.assertEqual(metrics["turns_with_errors"], 1)
        self.assertEqual(metrics["errors_total"], 1)
        self.assertEqual(metrics["recovered_turns"], 0)
        self.assertEqual(metrics["error_recovery_rate"], 0)

    def test_function_event_error_counted_and_recovered_with_response(self):
        """A function-event error on a turn that still responds counts as recovered."""
        conversation = {
            "turns": [
                _turn(response="still answered", function_events=[{"name": "f", "error": "boom"}]),
            ]
        }

        metrics = score_conversation(conversation).to_dict()

        self.assertEqual(metrics["errors_total"], 1)
        self.assertEqual(metrics["recovered_turns"], 1)
        self.assertEqual(metrics["error_recovery_rate"], 1)

    def test_state_changes_counted_across_added_updated_removed(self):
        """Added, updated, and removed state variables all count toward the total."""
        conversation = {
            "turns": [
                _turn(
                    state_changes=[
                        {"added": {"a": 1, "b": 2}, "updated": {"c": 3}, "removed": ["d"]}
                    ]
                ),
            ]
        }

        self.assertEqual(score_conversation(conversation).state_changes_total, 4)


class ChatMetricsIntegrationTest(unittest.TestCase):
    """Tests for the --metrics wiring in AgentStudioCLI.chat."""

    def setUp(self):
        self.mock_load_patcher = patch("poly.cli.AgentStudioCLI._load_project")
        self.mock_load = self.mock_load_patcher.start()
        self.proj = MagicMock()
        self.proj.branch_id = "main"
        self.proj.account_id = "test_account"
        self.proj.project_id = "test_project"
        self.proj.create_chat_session.return_value = {
            "conversation_id": "conv-123",
            "response": "Hello!",
            "conversation_ended": False,
            "metadata": {
                "function_events": [{"name": "greet"}],
                "in_flow": "welcome",
                "in_step": "intro",
            },
        }
        self.proj.send_message.return_value = {
            "response": "Reply",
            "conversation_ended": False,
            "metadata": {
                "function_events": [{"name": "lookup"}],
                "in_flow": "welcome",
                "in_step": "intro",
            },
        }
        self.proj.end_chat.return_value = None
        self.proj.get_conversation_url.return_value = "https://example.com/conv-123"
        self.mock_load.return_value = self.proj

    def tearDown(self):
        patch.stopall()

    @patch("poly.cli.json_print")
    def test_metrics_enriches_json_conversation(self, mock_json):
        """--metrics --json attaches a 'metrics' key to each conversation."""
        AgentStudioCLI.chat(
            TEST_DIR,
            environment="sandbox",
            input_messages=["Hi"],
            metrics=True,
            output_json=True,
        )

        payload = mock_json.call_args[0][0]
        conv = payload["conversations"][0]
        self.assertIn("metrics", conv)
        self.assertEqual(conv["metrics"]["tool_use"]["function_calls_total"], 2)
        self.assertEqual(conv["metrics"]["turns_total"], 2)

    @patch("poly.cli.print_execution_metrics")
    def test_metrics_prints_panel_without_json(self, mock_print):
        """--metrics without --json prints an execution metrics panel."""
        AgentStudioCLI.chat(
            TEST_DIR,
            environment="sandbox",
            input_messages=["Hi"],
            metrics=True,
        )

        mock_print.assert_called_once()

    @patch("poly.cli.json_print")
    def test_no_metrics_leaves_conversation_without_metrics_key(self, mock_json):
        """Without --metrics, conversations carry no 'metrics' key."""
        AgentStudioCLI.chat(
            TEST_DIR,
            environment="sandbox",
            input_messages=["Hi"],
            output_json=True,
        )

        payload = mock_json.call_args[0][0]
        conv = payload["conversations"][0]
        self.assertNotIn("metrics", conv)
