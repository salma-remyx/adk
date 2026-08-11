"""Validation-dimension coverage analysis for Agent Studio test suites.

Adapted from the five-dimension validation taxonomy in "Beyond Component
Testing: Validating Agentic AI Systems" (arXiv:2607.29405). That survey argues
that trustworthy deployment of agentic AI depends on validating *trajectories
in context* across five concerns -- behavioral, safety, temporal, regulatory,
and multi-agent -- and that behavioral evaluation is comparatively mature while
the other four remain under-developed.

ADK already stores Agent Studio ``TestCase`` resources (scenario text, tags,
prompt assertions, and function-call assertions) but never analyses *which*
validation dimensions a project's test suite actually exercises. This module
classifies a project's ``TestCase`` objects along the paper's five dimensions
and emits a coverage / gap report, so a team can see -- before they push --
that, for example, no test probes temporal validity or multi-agent handoff.

Mode-2 adapted port: the paper's taxonomy is reproduced at full fidelity, but
its learned / LLM-based dimension classifier is replaced by a deterministic,
parameter-free keyword proxy (vocab-overlap over the fields ADK already
stores). The paper's three case studies, runtime-evidence structures, and
lifecycle research agenda are intentionally out of scope -- they belong in a
downstream integration. The natural call site is the ``poly validate`` /
``poly push`` validate loop; wiring it there is left for a follow-up that can
edit the existing validate path.
"""

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from poly.resources.test_suite import TestCase

logger = logging.getLogger(__name__)

# Canonical validation dimensions, in the order the paper presents them.
DIMENSIONS: tuple[str, ...] = (
    "behavioral",
    "safety",
    "temporal",
    "regulatory",
    "multi_agent",
)

# Parameter-free keyword proxies for the four non-behavioral dimensions. Each
# token is matched as a substring against a normalized (lower-cased, separator-
# collapsed) blob of the test case's scenario, tags, asserted prompts, and
# asserted function-call names. The proxy is deliberately coarse -- it
# approximates the paper's learned classifier's *signal* without its weights.
DIMENSION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "safety": (
        "safe",
        "unsafe",
        "guardrail",
        "guard rail",
        "refusal",
        "refuse",
        "pii",
        "sensitive",
        "out of domain",
        "off topic",
        "toxic",
        "fallback",
        "abuse",
        "harmful",
        "profanity",
        "redact",
        "jailbreak",
        "prompt injection",
        "safety filter",
    ),
    "temporal": (
        "resume",
        "pause",
        "timeout",
        "time out",
        "delay",
        "persist",
        "session",
        "re engage",
        "reengage",
        "continuity",
        "elapsed",
        "retry",
        "re enter",
        "reenter",
        "long pause",
        "remember",
        "recall",
        "previous turn",
        "earlier",
        "stateful",
        "conversation history",
        "chat history",
    ),
    "regulatory": (
        "compliance",
        "gdpr",
        "pci",
        "ccpa",
        "hipaa",
        "consent",
        "opt out",
        "optout",
        "disclosure",
        "regulat",
        "legal",
        "audit",
        "retention",
        "data protection",
        "privacy policy",
        "terms of service",
        "right to be forgotten",
        "do not sell",
    ),
    "multi_agent": (
        "handoff",
        "hand off",
        "handover",
        "transfer",
        "escalat",
        "route",
        "routing",
        "delegate",
        "supervisor",
        "human agent",
        "live agent",
        "multi agent",
        "receptionist",
        "agent to agent",
        "sub agent",
    ),
}

_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lower-case ``text`` and collapse separator characters to single spaces.

    Hyphens and underscores become spaces so that ``"out-of-domain"``,
    ``"out_of_domain"`` and ``"out of domain"`` all match the same keyword.
    """
    collapsed = text.replace("-", " ").replace("_", " ")
    return _WS_RE.sub(" ", collapsed).strip().lower()


def _signal_text(test_case: TestCase) -> str:
    """Collect every ADK-stored field that can hint at a validation dimension.

    Draws on tags, the scenario description, asserted prompts, and asserted
    function-call names -- the exact surface the paper's classifier would read.
    """
    parts: list[str] = []
    if test_case.tags and test_case.tags.tags:
        parts.extend(test_case.tags.tags)
    if test_case.scenario:
        parts.append(test_case.scenario)
    assertions = test_case.assertions
    if assertions:
        parts.extend(assertions.prompts or [])
        for function_call in assertions.function_calls or []:
            if function_call.name:
                parts.append(function_call.name)
    return _normalize(" ".join(parts))


def _covered_dimensions(test_case: TestCase) -> set[str]:
    """Return the set of validation dimensions a single test case exercises.

    ``behavioral`` -- the paper's mature baseline -- is covered whenever the
    case asserts an expected prompt or function call. The remaining dimensions
    are covered when their keyword proxy matches the case's signal text.
    """
    covered: set[str] = set()
    assertions = test_case.assertions
    if assertions and (assertions.prompts or assertions.function_calls):
        covered.add("behavioral")
    signal = _signal_text(test_case)
    for dimension, keywords in DIMENSION_KEYWORDS.items():
        if any(keyword in signal for keyword in keywords):
            covered.add(dimension)
    return covered


@dataclass
class DimensionCoverage:
    """How many -- and which -- test cases exercise a single dimension."""

    dimension: str
    covered: tuple[str, ...] = field(default_factory=tuple)

    @property
    def count(self) -> int:
        """Number of test cases covering this dimension."""
        return len(self.covered)

    @property
    def is_covered(self) -> bool:
        """Whether at least one test case exercises this dimension."""
        return bool(self.covered)


@dataclass
class CoverageReport:
    """Cross-dimensional coverage / gap report for a project's test suite."""

    total: int
    dimensions: dict[str, DimensionCoverage]

    @property
    def covered_dimensions(self) -> list[str]:
        """Dimensions with at least one exercising test case."""
        return [d for d in DIMENSIONS if self.dimensions[d].is_covered]

    @property
    def gap_dimensions(self) -> list[str]:
        """Dimensions no test case exercises -- the paper's under-developed set."""
        return [d for d in DIMENSIONS if not self.dimensions[d].is_covered]

    def to_dict(self) -> dict:
        """Serialize the report to a plain dict for tooling / JSON output."""
        return {
            "total": self.total,
            "dimensions": {
                d: {
                    "count": self.dimensions[d].count,
                    "covered": list(self.dimensions[d].covered),
                }
                for d in DIMENSIONS
            },
            "gaps": self.gap_dimensions,
        }

    def render(self) -> str:
        """Render a human-readable coverage / gap summary."""
        lines = [f"Validation-dimension coverage across {self.total} test case(s):"]
        for dimension in DIMENSIONS:
            coverage = self.dimensions[dimension]
            label = dimension.replace("_", "-")
            mark = "covered" if coverage.is_covered else "GAP"
            names = ", ".join(coverage.covered) if coverage.covered else "none"
            lines.append(f"  [{mark}] {label}: {coverage.count} ({names})")
        if self.gap_dimensions:
            gap_label = ", ".join(d.replace("_", "-") for d in self.gap_dimensions)
            lines.append(f"Coverage gaps: {gap_label}")
        else:
            lines.append("Coverage gaps: none")
        return "\n".join(lines)


def analyze_coverage(test_cases: Iterable[TestCase]) -> CoverageReport:
    """Classify each ``TestCase`` across the five validation dimensions.

    Args:
        test_cases: the project's ``TestCase`` resources, e.g. gathered from an
            ``AgentStudioProject`` during the validate / push loop.

    Returns:
        A :class:`CoverageReport` bucketing every test case by the dimensions it
        exercises and surfacing the dimensions no test case exercises.
    """
    buckets: dict[str, list[str]] = {dimension: [] for dimension in DIMENSIONS}
    total = 0
    for test_case in test_cases:
        total += 1
        for dimension in _covered_dimensions(test_case):
            buckets[dimension].append(test_case.name)
    dimensions = {
        dimension: DimensionCoverage(dimension, tuple(names))
        for dimension, names in buckets.items()
    }
    report = CoverageReport(total=total, dimensions=dimensions)
    if report.gap_dimensions:
        logger.info(
            "Validation coverage gaps: %s",
            ", ".join(d.replace("_", "-") for d in report.gap_dimensions),
        )
    return report
