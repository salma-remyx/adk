"""Multidimensional execution metrics for chat conversations.

Scores an agent conversation trace along several dimensions — execution
efficiency, tool use, planning, and error recovery — going beyond the
pass/fail correctness assertions of the test suite.

The trace format scored here is the conversation dict emitted by
``poly chat --json`` (see ``AgentStudioCLI._run_chat_loop``): a list of
turns, each optionally carrying ``function_events``, ``flow`` (in_flow /
in_step), ``state_changes``, ``response``, and ``error``.

Adapted from the multidimensional evaluation metrics of A²E (Agent
Auditing Engine, arXiv:2608.07346). The paper's instrumented Monitor is
replaced by the trace already captured by the chat command; the metric
dimensions are kept.

Copyright PolyAI Limited
"""

from dataclasses import dataclass, field

from rich.panel import Panel
from rich.table import Table

from poly.output.console import console


@dataclass
class ExecutionMetrics:
    """Multidimensional execution metrics for a single conversation."""

    turns_total: int = 0
    turns_with_response: int = 0
    turns_with_errors: int = 0
    errors_total: int = 0
    error_rate: float = 0.0
    recovered_turns: int = 0
    error_recovery_rate: float = 0.0
    function_calls_total: int = 0
    distinct_functions: list[str] = field(default_factory=list)
    function_calls_per_turn: float = 0.0
    flows_entered: list[str] = field(default_factory=list)
    planning_steps: int = 0
    state_changes_total: int = 0

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict of all metrics."""
        return {
            "turns_total": self.turns_total,
            "turns_with_response": self.turns_with_response,
            "turns_with_errors": self.turns_with_errors,
            "errors_total": self.errors_total,
            "error_rate": round(self.error_rate, 4),
            "recovered_turns": self.recovered_turns,
            "error_recovery_rate": round(self.error_recovery_rate, 4),
            "tool_use": {
                "function_calls_total": self.function_calls_total,
                "distinct_functions": self.distinct_functions,
                "function_calls_per_turn": round(self.function_calls_per_turn, 4),
            },
            "planning": {
                "flows_entered": self.flows_entered,
                "planning_steps": self.planning_steps,
            },
            "state_changes_total": self.state_changes_total,
        }


def _turn_has_response(turn: dict) -> bool:
    """A turn counts as answered when it carries a non-empty agent response."""
    return bool(turn.get("response"))


def _turn_errors(turn: dict) -> int:
    """Count errors in one turn: transport-level errors plus function-event errors."""
    count = 1 if "error" in turn else 0
    for event in turn.get("function_events") or []:
        if event.get("error"):
            count += 1
    return count


def _flow_step(turn: dict) -> tuple[str, str] | None:
    """Return the (flow, step) the agent was in during this turn, if any."""
    flow = turn.get("flow") or {}
    in_flow = flow.get("in_flow")
    in_step = flow.get("in_step")
    if not in_flow and not in_step:
        return None
    return (in_flow or "", in_step or "")


def _state_change_count(turn: dict) -> int:
    """Count state variables added, updated, or removed during this turn."""
    count = 0
    for change in turn.get("state_changes") or []:
        count += len(change.get("added") or {})
        count += len(change.get("updated") or {})
        count += len(change.get("removed") or [])
    return count


def score_conversation(conversation: dict) -> ExecutionMetrics:
    """Score one conversation trace on multidimensional execution metrics.

    Args:
        conversation: A conversation dict as returned by
            ``AgentStudioCLI._run_chat_loop`` — must contain a ``turns`` list.

    Returns:
        ExecutionMetrics covering execution efficiency, tool use, planning,
        and error recovery.
    """
    metrics = ExecutionMetrics()
    turns = conversation.get("turns") or []

    distinct_functions: set[str] = set()
    flows_entered: list[str] = []
    previous_step: tuple[str, str] | None = None

    for turn in turns:
        metrics.turns_total += 1

        if _turn_has_response(turn):
            metrics.turns_with_response += 1

        turn_errors = _turn_errors(turn)
        if turn_errors:
            metrics.turns_with_errors += 1
            metrics.errors_total += turn_errors
            if _turn_has_response(turn):
                metrics.recovered_turns += 1

        for event in turn.get("function_events") or []:
            metrics.function_calls_total += 1
            if name := event.get("name"):
                distinct_functions.add(name)

        step = _flow_step(turn)
        if step is not None:
            flow_name = step[0]
            if flow_name and flow_name not in flows_entered:
                flows_entered.append(flow_name)
            if step != previous_step:
                metrics.planning_steps += 1
            previous_step = step

        metrics.state_changes_total += _state_change_count(turn)

    if metrics.turns_total:
        metrics.error_rate = metrics.turns_with_errors / metrics.turns_total
        metrics.function_calls_per_turn = metrics.function_calls_total / metrics.turns_total
    if metrics.turns_with_errors:
        metrics.error_recovery_rate = metrics.recovered_turns / metrics.turns_with_errors

    metrics.distinct_functions = sorted(distinct_functions)
    metrics.flows_entered = flows_entered
    return metrics


def print_execution_metrics(metrics: ExecutionMetrics) -> None:
    """Print a console summary panel of execution metrics.

    Groups the metrics by dimension — efficiency, tool use, planning, and
    error recovery — mirroring the multidimensional breakdown.
    """
    d = metrics.to_dict()
    table = Table(show_header=False, box=None, padding=(0, 1), expand=False)
    table.add_column("metric", style="bold", no_wrap=True)
    table.add_column("value", overflow="fold")

    table.add_row("Turns", str(d["turns_total"]))
    table.add_row("Turns with response", str(d["turns_with_response"]))
    table.add_row(
        "Function calls",
        f"{d['tool_use']['function_calls_total']} "
        f"({d['tool_use']['function_calls_per_turn']}/turn, "
        f"{len(d['tool_use']['distinct_functions'])} distinct)",
    )
    if d["tool_use"]["distinct_functions"]:
        table.add_row("Functions used", ", ".join(d["tool_use"]["distinct_functions"]))
    table.add_row(
        "Planning steps",
        f"{d['planning']['planning_steps']} across {len(d['planning']['flows_entered'])} flow(s)",
    )
    if d["planning"]["flows_entered"]:
        table.add_row("Flows entered", ", ".join(d["planning"]["flows_entered"]))
    table.add_row("State changes", str(d["state_changes_total"]))
    table.add_row(
        "Errors",
        f"{d['errors_total']} on {d['turns_with_errors']} turns "
        f"(rate {d['error_rate']:.0%})",
    )
    table.add_row(
        "Recovered turns",
        f"{d['recovered_turns']} (recovery rate {d['error_recovery_rate']:.0%})",
    )

    console.print(
        Panel(
            table,
            title="[bold]Execution Metrics[/bold]",
            border_style="cyan",
            padding=(0, 1),
        )
    )
