"""Tool-specification safety analyzer for the local validate pipeline.

Adapted from SafeKeep (arXiv:2607.29254v1, "Tool Specifications Matter:
Uncovering and Mitigating Safety Risks in AI Agents"), which shows that
schema-formatted tool specifications weaken an LLM's internal refusal signals
and that assessing requests against *flattened textual* tool specs — while
keeping the original schema for execution — restores safer behavior.

ADK's first-class artifacts ARE tool specifications (Function
parameters/description and API integration operations), so the paper's core
mechanism ports directly onto the existing ``validate`` pipeline rather than
requiring a runtime agent:

* :func:`flatten_tool_spec` reproduces SafeKeep's schema -> text flattening,
  the inference-time transformation that lets a safety check reason over prose
  instead of structured schema.
* :func:`analyze_tool_spec_safety` flags tool specs whose schema-dominated
  shape is the condition SafeKeep links to safety degradation, and returns the
  flattened textual spec as the recommended mitigation surface.

Mode 2 (adapted port): the paper evaluates refusal rates with an LLM judge
across two benchmarks and four LLMs. ADK is a configuration/sync CLI with no
agent runtime, so the learned refusal signal is replaced by a parameter-free
heuristic that surfaces the same risk condition deterministically at validate
time. The benchmark / LLM evaluation is intentionally out of scope. Findings
are advisory and never block validation.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from poly.resources.api_integration import ApiIntegration
from poly.resources.function import Function
from poly.resources.resource import Resource

logger = logging.getLogger(__name__)

# Minimum natural-language words a tool description needs to be considered
# grounded enough for safety reasoning. Below this, with structured schema
# present, the spec is schema-dominated: the model sees parameters/operations
# with little prose to ground a refusal — the condition SafeKeep links to
# safety degradation.
_MIN_GROUNDING_WORDS = 3


@dataclass
class ToolSpecSafetyFinding:
    """Advisory finding for a tool spec that may degrade agent safety.

    Attributes:
        resource_type: Human-readable artifact kind (e.g. ``"function"``).
        resource_name: Name of the flagged tool.
        file_path: On-disk location of the flagged spec.
        message: Why the spec was flagged.
        flattened_spec: SafeKeep-style textual rendering; the suggested
            surface for safety checks or for enriching the description.
    """

    resource_type: str
    resource_name: str
    file_path: str
    message: str
    flattened_spec: str


def flatten_tool_spec(resource: Function | ApiIntegration) -> str:
    """Flatten a schema-formatted tool spec into natural-language text.

    Reproduces SafeKeep's inference-time transformation: structured schema
    (function parameters / API operations) is rendered as a prose block so a
    safety judgment can be made over text rather than schema, which the paper
    shows preserves the model's refusal signal. The original schema is left
    untouched for execution.

    Args:
        resource: A Function or ApiIntegration tool specification.

    Returns:
        A flattened textual rendering of the tool spec.

    Raises:
        TypeError: If ``resource`` is not a Function or ApiIntegration.
    """
    if isinstance(resource, Function):
        return _flatten_function(resource)
    if isinstance(resource, ApiIntegration):
        return _flatten_integration(resource)
    raise TypeError(
        f"flatten_tool_spec expects a Function or ApiIntegration, got {type(resource).__name__}"
    )


def _flatten_function(function: Function) -> str:
    """Render a Function's description + parameters as a prose block."""
    lines: list[str] = []
    description = (function.description or "").strip()
    if description:
        lines.append(description)
    for param in function.parameters:
        type_label = (getattr(param, "type", "") or "").strip()
        param_desc = (getattr(param, "description", "") or "").strip()
        head = f"- {param.name}"
        if type_label:
            head += f" ({type_label})"
        lines.append(f"{head}: {param_desc}" if param_desc else head)
    return "\n".join(lines).strip()


def _flatten_integration(integration: ApiIntegration) -> str:
    """Render an ApiIntegration's description + operations as a prose block."""
    lines: list[str] = []
    description = (integration.description or "").strip()
    if description:
        lines.append(description)
    for operation in integration.operations:
        method = (operation.method or "").strip().upper()
        resource = (operation.resource or "").strip()
        name = (operation.name or "").strip()
        label = f"{method} {resource}".strip()
        lines.append(f"- {label} ({name})" if name else f"- {label}")
    return "\n".join(lines).strip()


def is_schema_dominated(resource: Function | ApiIntegration) -> bool:
    """Return whether a tool spec is schema-dominated (low textual grounding).

    Parameter-free proxy for the risk condition SafeKeep identifies: the spec
    carries structured schema (parameters / operations) but too little
    natural-language description to ground a refusal.

    Args:
        resource: A Function or ApiIntegration tool specification.

    Returns:
        ``True`` if the spec has schema with insufficient textual grounding.
    """
    if isinstance(resource, Function):
        schema_units = len(resource.parameters)
    elif isinstance(resource, ApiIntegration):
        schema_units = len(resource.operations)
    else:
        return False
    if schema_units == 0:
        return False
    description = (getattr(resource, "description", "") or "").strip()
    return len(description.split()) < _MIN_GROUNDING_WORDS


def analyze_tool_spec_safety(
    resources: Mapping[type, Mapping[str, Resource]],
) -> list[ToolSpecSafetyFinding]:
    """Flag schema-dominated tool specs across a project's resources.

    Walks a ``ResourceMap`` (as built by ``AgentStudioProject.validate_project``)
    and returns an advisory finding for every Function / ApiIntegration whose
    schema-dominated shape may degrade downstream agent safety reasoning.

    Args:
        resources: A mapping of resource type to resource-id -> Resource.

    Returns:
        Advisory findings. Never raises; safe to call from the validate path.
    """
    findings: list[ToolSpecSafetyFinding] = []
    for resource_dict in resources.values():
        for resource in resource_dict.values():
            if not isinstance(resource, (Function, ApiIntegration)):
                continue
            if not is_schema_dominated(resource):
                continue
            kind = "function" if isinstance(resource, Function) else "api_integration"
            name = getattr(resource, "name", "") or ""
            try:
                file_path = resource.file_path
            except Exception:  # pragma: no cover - defensive, file_path is stable
                logger.exception("Could not resolve file_path for %s", resource)
                file_path = ""
            findings.append(
                ToolSpecSafetyFinding(
                    resource_type=kind,
                    resource_name=name,
                    file_path=file_path,
                    message=(
                        f"Tool spec '{name}' is schema-dominated: its natural-language "
                        "description is too thin relative to its structured schema. "
                        "Schema-formatted tool specs weaken agent refusal signals; "
                        "ground safety checks on the flattened textual spec or enrich "
                        "the description before deploying this agent."
                    ),
                    flattened_spec=flatten_tool_spec(resource),
                )
            )
    return findings


def _format_finding(finding: ToolSpecSafetyFinding) -> str:
    """Render a finding as a single advisory line for log output."""
    location = f" ({finding.file_path})" if finding.file_path else ""
    return f"[tool-spec safety]{location} {finding.message}"


def log_findings(
    findings: list[ToolSpecSafetyFinding],
    logger: logging.Logger | None = None,
) -> None:
    """Emit advisory findings as non-blocking warnings.

    Args:
        findings: Findings produced by :func:`analyze_tool_spec_safety`.
        logger: Logger to emit through; defaults to this module's logger so
            the call site (e.g. ``validate_project``) can attribute advisories
            to its own logger for clearer output.
    """
    log = logger if logger is not None else logging.getLogger(__name__)
    for finding in findings:
        log.warning(_format_finding(finding))


__all__: list[Any] = [
    "ToolSpecSafetyFinding",
    "analyze_tool_spec_safety",
    "flatten_tool_spec",
    "is_schema_dominated",
    "log_findings",
]
