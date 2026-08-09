"""Continuous-assurance readiness checks for Agent Studio projects.

Builds a cross-resource dependency map and reports readiness findings
(dangling cross-resource references) with actionable diagnostics. The check
reuses the existing ``ResourceMap`` / ``ResourceMapping`` contract so it slots
into the project's validate / push flow without introducing a new data shape.

Adapted from "Toward Continuous Assurance for the Democratization of AI Agent
Creation in Industry" (arXiv:2607.21495v1). The paper's bespoke prototype
auditor and scenario-based assessment framework are replaced by this repo's
native resource graph; the core mechanism -- dependency mapping, readiness
contracts, scheduled checks, and diagnostics -- is preserved (Mode 2 adapted
port). The paper's external-service / scheduled-cron checks are intentionally
out of scope: this module reasons only about dependencies the ADK already
tracks locally (functions <-> variables, functions -> flows, tests -> functions).

Copyright PolyAI Limited
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Optional

from poly.resources import (
    Function,
    FunctionCallAssertion,
    Resource,
    ResourceMapping,
    TestCase,
)

logger = logging.getLogger(__name__)

# Resource prefixes used in cross-resource references.
# See each resource type's ``get_resource_prefix``.
FUNCTION_PREFIX = "fn"
FLOW_PREFIX = "flow"
VARIABLE_PREFIX = "vrbl"


@dataclass
class DependencyEdge:
    """A directed dependency from one resource to another."""

    source_prefix: str
    source_name: str
    source_file: Optional[str]
    target_prefix: str
    target_name: str
    relation: str

    def __str__(self) -> str:
        """Render the edge as ``src --relation--> tgt``."""
        return (
            f"{self.source_prefix}:{self.source_name} --{self.relation}--> "
            f"{self.target_prefix}:{self.target_name}"
        )


@dataclass
class ReadinessFinding:
    """A single readiness diagnostic for an unresolved dependency."""

    severity: str
    source: str
    target: str
    relation: str
    message: str
    remediation: str


@dataclass
class _DeclaredIndex:
    """Index of declared resources, keyed by prefix for fast lookup."""

    names_by_prefix: dict[str, set[str]]

    def has(self, prefix: str, name: str) -> bool:
        """Return True if a resource with ``(prefix, name)`` is declared."""
        return name in self.names_by_prefix.get(prefix, set())


def _build_declared_index(resource_mappings: list[ResourceMapping]) -> _DeclaredIndex:
    """Build an index of declared resource names grouped by prefix."""
    names_by_prefix: dict[str, set[str]] = {}
    for mapping in resource_mappings:
        prefix = mapping.resource_prefix or ""
        name = mapping.resource_name
        if prefix and name:
            names_by_prefix.setdefault(prefix, set()).add(name)
    return _DeclaredIndex(names_by_prefix)


def _mapping_by_id(
    resource_mappings: list[ResourceMapping],
) -> dict[tuple[type[Resource], str], ResourceMapping]:
    """Index resource mappings by ``(resource_type, resource_id)``."""
    return {(m.resource_type, m.resource_id): m for m in resource_mappings}


def _iter_edges(
    resources_dict: dict[type[Resource], dict[str, Resource]],
    resource_mappings: list[ResourceMapping],
) -> Iterator[DependencyEdge]:
    """Yield directed dependency edges drawn from the project's resources.

    Edge sources are grounded in attributes the repo already maintains:
      * function -> flow, from a function mapping's ``flow_name``;
      * test -> function, from a test case's asserted function calls;
      * function -> variable, from a function's resolved ``variable_references``.
    """
    mappings = _mapping_by_id(resource_mappings)

    # function -> flow (declared via the function's mapping flow_name).
    for mapping in resource_mappings:
        if mapping.resource_prefix != FUNCTION_PREFIX:
            continue
        flow_name = mapping.flow_name
        if not flow_name:
            continue
        yield DependencyEdge(
            source_prefix=FUNCTION_PREFIX,
            source_name=mapping.resource_name,
            source_file=mapping.file_path,
            target_prefix=FLOW_PREFIX,
            target_name=flow_name,
            relation="belongs_to",
        )

    # test -> function (each asserted function call).
    for test_case in resources_dict.get(TestCase, {}).values():
        mapping = mappings.get((TestCase, test_case.resource_id))
        source_prefix = (mapping.resource_prefix if mapping else None) or "test"
        assertions = getattr(test_case, "assertions", None)
        function_calls = getattr(assertions, "function_calls", None) or []
        for call in function_calls:
            if not isinstance(call, FunctionCallAssertion):
                continue
            yield DependencyEdge(
                source_prefix=source_prefix,
                source_name=test_case.name,
                source_file=(mapping.file_path if mapping else None),
                target_prefix=FUNCTION_PREFIX,
                target_name=call.name,
                relation="calls",
            )

    # function -> variable (resolved references; informational only).
    for function in resources_dict.get(Function, {}).values():
        mapping = mappings.get((Function, function.resource_id))
        variable_references = getattr(function, "variable_references", None) or {}
        for variable_id in variable_references:
            variable_name = next(
                (
                    candidate.resource_name
                    for candidate in resource_mappings
                    if candidate.resource_prefix == VARIABLE_PREFIX
                    and candidate.resource_id == variable_id
                ),
                None,
            )
            if not variable_name:
                continue
            yield DependencyEdge(
                source_prefix=FUNCTION_PREFIX,
                source_name=function.name,
                source_file=(mapping.file_path if mapping else None),
                target_prefix=VARIABLE_PREFIX,
                target_name=variable_name,
                relation="references",
            )


def build_dependency_map(
    resources_dict: dict[type[Resource], dict[str, Resource]],
    resource_mappings: list[ResourceMapping],
) -> list[DependencyEdge]:
    """Build the cross-resource dependency map (edges) for the project.

    Args:
        resources_dict: Loaded resources keyed by type then id.
        resource_mappings: Declared resource mappings (source of truth for
            what exists).

    Returns:
        list[DependencyEdge]: Directed dependency edges across resources.
    """
    return list(_iter_edges(resources_dict, resource_mappings))


def assess_readiness(
    resources_dict: dict[type[Resource], dict[str, Resource]],
    resource_mappings: list[ResourceMapping],
) -> list[ReadinessFinding]:
    """Assess operational readiness against the dependency map.

    A resource is "not ready" when it depends on a resource that is not
    declared in ``resource_mappings`` (a dangling reference) -- the local
    shape of the silent-degradation risk the paper describes. Each finding
    carries an actionable remediation hint. The function never raises; it
    returns findings for the caller to surface.

    Args:
        resources_dict: Loaded resources keyed by type then id.
        resource_mappings: Declared resource mappings.

    Returns:
        list[ReadinessFinding]: Readiness findings (empty if everything
        resolves).
    """
    declared = _build_declared_index(resource_mappings)
    findings: list[ReadinessFinding] = []
    for edge in _iter_edges(resources_dict, resource_mappings):
        if declared.has(edge.target_prefix, edge.target_name):
            continue
        location = f" in {edge.source_file}" if edge.source_file else ""
        findings.append(
            ReadinessFinding(
                severity="error",
                source=f"{edge.source_prefix}:{edge.source_name}",
                target=f"{edge.target_prefix}:{edge.target_name}",
                relation=edge.relation,
                message=(
                    f"{edge.source_prefix}:{edge.source_name} {edge.relation} "
                    f"undeclared {edge.target_prefix}:{edge.target_name}."
                ),
                remediation=(
                    f"Add the missing {edge.target_prefix} resource "
                    f"'{edge.target_name}' or remove the reference{location}."
                ),
            )
        )
    return findings


def format_readiness_report(findings: list[ReadinessFinding]) -> str:
    """Format readiness findings as a human-readable, actionable report.

    Args:
        findings: Readiness findings returned by :func:`assess_readiness`.

    Returns:
        str: A multi-line report (or an all-clear message when there are no
        findings).
    """
    if not findings:
        return "All cross-resource dependencies resolve."
    lines = [f"Readiness check: {len(findings)} unresolved dependency reference(s)."]
    for finding in findings:
        lines.append(f"[{finding.severity.upper()}] {finding.message} -> {finding.remediation}")
    return "\n".join(lines)
