"""Dynamic capability scoping for Agent Studio projects.

Adapted (Mode 2) from "Dynamic Capability Scoping for Enterprise AI Agents:
A Synthetic Dataset and Three-Source Permission Architecture"
(arXiv:2607.22445v1).

The paper argues that enterprise agents are *persistently over-privileged*:
they hold every credential they might ever need for every task, which
expands the attack surface, and that capability scoping should be a
prevention mechanism rather than a detection one — "a credential that does
not exist in an agent's context cannot be misused regardless of the agent's
reasoning or evasion sophistication."

This module instantiates that idea as a static analysis over a project's
real artifact graph. A global ``Function`` is an agent's always-available
tool (the static-credential analog). A ``FlowStep`` prompt, a rule, or a
topic action that references it via ``{{fn:<id>}}`` is a task context that
needs it. A global Function that is configured but referenced by no task
surface is *persistent over-privilege*: a credential the agent holds even
though no task requires it.

Mapping to the paper's three-source architecture (Mode 2 substitutions):

* **Task-context classifier (Source 2)** -> CORE, full fidelity. A global
  function unreferenced by any task surface is flagged as over-privileged.
* **Policy prohibitions (Source 3)** -> parameter-free keyword denylist
  (``POLY_CAPABILITY_SCOPE_DENY``). Global functions whose name matches a
  prohibited capability keyword are flagged. Defaults to empty (opt-in),
  substituting the paper's derived combination-prohibition engine with a
  simple keyword rule.
* **Role-based ceiling (Source 1)** -> intentionally scoped out. The repo's
  ``SettingsRole`` is a free-text persona, not a structured RBAC role set,
  so a precise per-role capability ceiling is not computable statically.

Like the paper, the analyzer supports two deployment modes: *observe-only*
(the default — logs findings as a behavioral signal) and *enforcing*
(promotes findings to push-blocking validation errors when
``POLY_CAPABILITY_SCOPE_ENFORCE`` is set).
"""

import logging
import os
import re
from dataclasses import dataclass

from poly.resources.agent_settings import SettingsRules
from poly.resources.flows import FlowStep
from poly.resources.function import Function, FunctionType
from poly.resources.topic import Topic

logger = logging.getLogger(__name__)

# Matches {{fn:<token>}} references to global functions in resource text.
# The token may be a resolved resource id (e.g. FUNCTION-format_date) or, for
# text that has not yet been name->id resolved, the bare function name.
_FN_REFERENCE_RE = re.compile(r"{{fn:([\w-]+)}}")

_ENFORCE_VALUES = {"1", "true", "yes", "on"}


def _is_enforce_mode() -> bool:
    """Return True if capability-scope findings should block pushes.

    Reads the ``POLY_CAPABILITY_SCOPE_ENFORCE`` environment variable so the
    enforcing/observe-only switch matches the paper's deployment-mode choice
    without requiring a schema change to project config.
    """
    return os.environ.get("POLY_CAPABILITY_SCOPE_ENFORCE", "").strip().lower() in _ENFORCE_VALUES


def _denied_keywords() -> set[str]:
    """Return the configured prohibited-capability keywords (lowercased).

    Sourced from the comma-separated ``POLY_CAPABILITY_SCOPE_DENY`` env var.
    Empty by default, so Source 3 is a no-op unless opted in.
    """
    raw = os.environ.get("POLY_CAPABILITY_SCOPE_DENY", "")
    return {kw.strip().lower() for kw in raw.split(",") if kw.strip()}


@dataclass(frozen=True)
class CapabilityFinding:
    """A single capability-scoping finding for one global function."""

    capability: str
    """The global function's name."""

    resource_id: str
    """The global function's resource id."""

    file_path: str
    """The global function's source file path."""

    source: str
    """Origin of the finding: ``task_context`` or ``policy_prohibition``."""

    def render(self) -> str:
        """Render the finding as a validation-style message."""
        if self.source == "policy_prohibition":
            reason = "matches a prohibited capability keyword (policy prohibition)"
        else:
            reason = "not referenced by any flow step, rule, or topic (persistent over-privilege)"
        return f"Capability scope: '{self.capability}' ({self.file_path}) is {reason}."


def _collect_fn_tokens(resources_by_type: dict) -> set[str]:
    """Collect every ``{{fn:<token>}}`` token across all reference surfaces.

    Global functions may be consumed by flow step prompts, the rules
    behaviour text, or topic actions/example queries. All three are scanned
    so the over-privilege signal does not false-positive on a function that
    is reached through a less common surface.
    """
    tokens: set[str] = set()
    for flow_step in resources_by_type.get(FlowStep, {}).values():
        tokens.update(_FN_REFERENCE_RE.findall(flow_step.prompt or ""))
    for rules in resources_by_type.get(SettingsRules, {}).values():
        tokens.update(_FN_REFERENCE_RE.findall(rules.behaviour or ""))
    for topic in resources_by_type.get(Topic, {}).values():
        tokens.update(_FN_REFERENCE_RE.findall(topic.actions or ""))
        for query in topic.example_queries or []:
            tokens.update(_FN_REFERENCE_RE.findall(query or ""))
    return tokens


def find_capability_scope_findings(resources_by_type: dict) -> list[CapabilityFinding]:
    """Find over-privilege findings over the project's artifact graph.

    Args:
        resources_by_type: A mapping of resource type -> {resource_id:
            resource}, as produced by ``AgentStudioProject.validate_project``
            / ``validate_resources``.

    Returns:
        One ``CapabilityFinding`` per over-privileged global function.
        Sources emitted: ``task_context`` (a global function referenced by
        no task surface) and ``policy_prohibition`` (a global function whose
        name matches a configured denylist keyword).
    """
    global_functions = [
        fn
        for fn in resources_by_type.get(Function, {}).values()
        if fn.function_type == FunctionType.GLOBAL
    ]
    if not global_functions:
        return []

    # Resolve tokens against both id and name so the analysis is robust to
    # whether the surrounding text has been name->id resolved or not.
    fn_by_id = {fn.resource_id: fn for fn in global_functions}
    fn_by_name = {fn.name: fn for fn in global_functions}
    referenced_ids: set[str] = set()
    for token in _collect_fn_tokens(resources_by_type):
        if token in fn_by_id:
            referenced_ids.add(token)
        elif token in fn_by_name:
            referenced_ids.add(fn_by_name[token].resource_id)

    denied = _denied_keywords()

    findings: list[CapabilityFinding] = []
    for fn in global_functions:
        if fn.resource_id not in referenced_ids:
            findings.append(
                CapabilityFinding(
                    capability=fn.name,
                    resource_id=fn.resource_id,
                    file_path=fn.file_path,
                    source="task_context",
                )
            )
        if denied and any(keyword in fn.name.lower() for keyword in denied):
            findings.append(
                CapabilityFinding(
                    capability=fn.name,
                    resource_id=fn.resource_id,
                    file_path=fn.file_path,
                    source="policy_prohibition",
                )
            )
    return findings


def evaluate_capability_scope(resources_by_type: dict) -> list[str]:
    """Run capability scoping and return messages that should block a push.

    In observe-only mode (the default), findings are logged as warnings — a
    behavioral signal for review — and an empty list is returned so pushes
    are not blocked. In enforcing mode (``POLY_CAPABILITY_SCOPE_ENFORCE``),
    the rendered findings are returned so they surface as push-blocking
    validation errors via the project validate loop.
    """
    findings = find_capability_scope_findings(resources_by_type)
    if not findings:
        return []

    messages = [finding.render() for finding in findings]
    if _is_enforce_mode():
        return messages
    for message in messages:
        logger.warning("%s", message)
    return []
