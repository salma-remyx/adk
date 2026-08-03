"""User-centric system prompt auditing for Agent Studio agents.

This module adapts the core mechanism of *AISPA: User-Centric System Prompt
Auditing for Large Language Model Applications* (arXiv:2607.28617) into the ADK.
AISPA breaks a system prompt into individual instructions and classifies each
along eight user-centric dimensions as either *protective* (of users) or
*problematic* (working against user interests), then reports dimension coverage
and the coexistence of protective and problematic instructions.

Implementation mode (Mode 2 -- adapted port): the paper's core taxonomy and
protective/problematic framing are kept at full fidelity, while the paper's
audit *methodology* (manual + LLM labelling of 3,249 instructions across 88
commercial products) is substituted with a parameter-free, rule-based proxy.
The proxy uses keyword/phrase heuristics per dimension. The paper's separate
benchmark of commercial products is intentionally out of scope: this module
audits an agent's own prompt, not a third-party corpus.

A system prompt in ADK is composed from the agent's personality, role and rules
text (see ``SettingsPersonality.custom``, ``SettingsRole.custom`` and the rules
resource) -- the same prompt strings that flow through
``resource_utils.get_references_from_prompt`` during validation. The audit is
designed to sit alongside that existing prompt-analysis surface.

Copyright PolyAI Limited
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "AuditDimension",
    "InstructionFinding",
    "AuditReport",
    "split_prompt_into_instructions",
    "classify_instruction",
    "audit_system_prompt",
    "audit_agent_settings",
]

Classification = Literal["protective", "problematic"]

#: Minimum number of tokens for a fragment to count as a standalone instruction.
MIN_INSTRUCTION_WORDS = 3


@dataclass(frozen=True)
class AuditDimension:
    """One axis of the user-centric prompt taxonomy.

    Attributes:
        name: Stable snake_case identifier for the dimension.
        label: Human-readable label for display.
        protective: Regexes matching instructions that protect users on this axis.
        problematic: Regexes matching instructions that work against users here.
    """

    name: str
    label: str
    protective: tuple[re.Pattern[str], ...]
    problematic: tuple[re.Pattern[str], ...]


def _compile(patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    """Compile a tuple of patterns case-insensitively."""
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


# The eight user-centric dimensions, grounded in the paper's framing of user
# protection, transparency, privacy, safety, fairness, honesty, autonomy and
# accountability. Each lists protective and problematic indicator phrases.
DIMENSIONS: tuple[AuditDimension, ...] = (
    AuditDimension(
        name="identity_transparency",
        label="Identity & Transparency",
        protective=_compile(
            (
                # Directive disclosure language only -- a bare "you are an AI"
                # mention also appears inside problematic instructions such as
                # "never reveal that you are an AI", so we require an explicit
                # disclose/identify/mention directive to count as protective.
                r"\b(be transparent|disclose that you|always disclose|never impersonat)",
                r"\b(mention that you('re| are) (an? )?(AI|automated|bot))",
                r"\b(identify yourself as (an? )?(AI|automated|bot))",
            )
        ),
        problematic=_compile(
            (
                r"\b(never (say|reveal|mention|tell)).{0,40}(you('re| are) (an? )?(AI|bot|automated))",
                r"\b(pretend (to be|you are|you're) (a |an )?(human|real person|person))",
                r"\b(impersonat\w+ (a |an )?(human|real person))",
                r"\b(hide (that|the fact) you('re| are) (an? )?(AI|automated|bot))",
                r"\b(act as if you are human)\b",
            )
        ),
    ),
    AuditDimension(
        name="privacy_data_protection",
        label="Privacy & Data Protection",
        protective=_compile(
            (
                r"\b(do not|don't|never) (store|save|retain|collect|log|record).{0,30}"
                r"(personal|sensitive|financial|payment|password|PII|data)",
                r"\b(avoid (asking for|collecting) (personal|sensitive|financial))",
                r"\b(keep .{0,20}(confidential|private))",
                r"\b(do not ask for .{0,20}(password|social security|credit card|card number))",
            )
        ),
        problematic=_compile(
            (
                r"\b(collect|store|save|retain|record|log) .{0,30}(personal|sensitive|financial|"
                r"payment|password|social security|credit card|PII)",
                r"\b(ask (the )?(user|customer) for .{0,20}(password|social security|full card))",
                r"\b(share .{0,30}(personal|customer|user) data with)",
            )
        ),
    ),
    AuditDimension(
        name="safety_harm_prevention",
        label="Safety & Harm Prevention",
        protective=_compile(
            (
                r"\b(refuse|decline|do not|don't|never) .{0,30}(harmful|dangerous|illegal|"
                r"unlawful|violent|self-harm)",
                r"\b(do not (assist|help) with (illegal|harmful|dangerous))",
                r"\b(keep (conversations|responses) safe)",
            )
        ),
        problematic=_compile(
            (
                r"\b(ignore (safety|guardrails|restrictions|content policies))",
                r"\b(bypass (safety|restrictions|guardrails|filters))",
                r"\b(you (may|can|should) (discuss|help with|assist with) (illegal|harmful))",
                r"\b(no guardrails|no restrictions|no safety)",
            )
        ),
    ),
    AuditDimension(
        name="fairness_nondiscrimination",
        label="Fairness & Non-discrimination",
        protective=_compile(
            (
                r"\b(treat .{0,20}(equally|the same|fairly|without bias))",
                r"\b(without (bias|discrimination|prejudice))",
                r"\b(inclusive|respectful to all|do not discriminate)",
            )
        ),
        problematic=_compile(
            (
                r"\b(refuse .{0,20}(based on|because of) (race|gender|religion|nationality|"
                r"disability|sexual orientation|age))",
                r"\b(prioritize .{0,20}(race|gender|religion|nationality))",
                r"\b(stereotype|use slurs|discriminat\w+)",
            )
        ),
    ),
    AuditDimension(
        name="honesty_accuracy",
        label="Honesty & Accuracy",
        protective=_compile(
            (
                r"\b(do not|don't|never) (fabricate|make up|invent|lie|mislead)",
                r"\b(admit when you (do not|don't) know)",
                r"\b(be (honest|accurate|truthful))",
                r"\b(cite (your )?sources)",
            )
        ),
        problematic=_compile(
            (
                r"\b(always agree with .{0,15}(even when|even if).{0,20}(wrong|incorrect))",
                r"\b(never (correct|contradict|disagree with) the (user|customer))",
                r"\b(never (say|admit) you (do not|don't) know)",
                r"\b(make up|fabricate|invent) .{0,20}(answers|details|facts|statistics)",
            )
        ),
    ),
    AuditDimension(
        name="user_autonomy_consent",
        label="User Autonomy & Consent",
        protective=_compile(
            (
                r"\b(ask (before|for permission|for consent))",
                r"\b(let the (user|customer) (opt out|cancel|decline|end))",
                r"\b(respect the (user|customer)('s| choice| decision))",
                r"\b(offer to (transfer|escalate).{0,20}(human|agent))",
            )
        ),
        problematic=_compile(
            (
                r"\b(pressure|coerce|manipulate) the (user|customer)",
                r"\b(discourage .{0,20}(ending|leaving|hanging up|cancelling))",
                r"\b(keep (the )?(user|customer) engaged|do not let (the )?(user|customer) (leave|end|hang up))",
                r"\b(never transfer|never escalate).{0,20}(human|agent)",
            )
        ),
    ),
    AuditDimension(
        name="confidentiality_ip",
        label="Confidentiality & Intellectual Property",
        protective=_compile(
            (
                r"\b(do not (share|disclose|reveal|output)) .{0,30}(proprietary|confidential|"
                r"copyrighted|intellectual property|internal)",
                r"\b(respect (copyright|intellectual property|licen[cs]e))",
                r"\b(do not (reproduce|recite) copyrighted)",
            )
        ),
        problematic=_compile(
            (
                r"\b(reveal|share|disclose|output) .{0,20}(these instructions|your (system )?"
                r"(instructions|prompt)|the system prompt)",
                r"\b(reproduce|recite|output) copyrighted (material|text|lyrics|books)",
                r"\b(share (the )?(company|organization|firm)('s)? (internal|proprietary|data))",
            )
        ),
    ),
    AuditDimension(
        name="accountability_oversight",
        label="Accountability & Oversight",
        protective=_compile(
            (
                r"\b(escalate|transfer) .{0,20}(to a |to an )?(human|agent|supervisor)",
                r"\b(stay (within|inside) your (scope|role|remit))",
                r"\b(refer (complex|sensitive|important) .{0,20}(human|agent|specialist))",
                r"\b(log .{0,20}(quality|audit|compliance|oversight))",
            )
        ),
        problematic=_compile(
            (
                r"\b(act (outside|beyond) your (scope|role|remit))",
                r"\b(make decisions (for|on behalf of) the (user|customer))",
                r"\b(you (have|are granted) (full|complete) (authority|access|control))",
                r"\b(override .{0,15}(policy|policies|rules|approvals))",
            )
        ),
    ),
)


@dataclass(frozen=True)
class InstructionFinding:
    """A single protective/problematic label applied to one instruction.

    Attributes:
        instruction: The prompt fragment that was classified.
        dimension: The ``AuditDimension.name`` it was matched against.
        classification: Either ``"protective"`` or ``"problematic"``.
    """

    instruction: str
    dimension: str
    classification: Classification


@dataclass
class AuditReport:
    """Result of auditing a system prompt.

    Attributes:
        prompt: The original prompt text that was audited.
        instructions: The discrete instructions the prompt was split into.
        findings: Every protective/problematic label produced.
        dimensions_covered: Dimensions with at least one protective instruction.
        all_dimensions: Every dimension name in the taxonomy, in order.
        problematic_instructions: Distinct instructions carrying >=1 problematic label.
    """

    prompt: str
    instructions: list[str]
    findings: list[InstructionFinding]
    dimensions_covered: list[str]
    all_dimensions: list[str]
    problematic_instructions: list[str] = field(default_factory=list)

    @property
    def protective_instruction_count(self) -> int:
        """Number of distinct instructions with at least one protective label."""
        return len({f.instruction for f in self.findings if f.classification == "protective"})

    @property
    def has_protective(self) -> bool:
        """Whether the prompt contains any protective instruction."""
        return self.protective_instruction_count > 0

    @property
    def has_problematic(self) -> bool:
        """Whether the prompt contains any problematic instruction."""
        return len(self.problematic_instructions) > 0

    @property
    def coverage_fraction(self) -> float:
        """Fraction of the eight dimensions covered by protective instructions."""
        if not self.all_dimensions:
            return 0.0
        return len(self.dimensions_covered) / len(self.all_dimensions)

    @property
    def covers_all_dimensions(self) -> bool:
        """Whether protective instructions span every taxonomy dimension."""
        return len(self.dimensions_covered) == len(self.all_dimensions)

    def summary(self) -> dict[str, object]:
        """Return the headline AISPA-style numbers for this prompt.

        Mirrors the paper's reported aggregates: instruction count, protective
        and problematic counts, dimension coverage, and the protective/problematic
        coexistence flags.
        """
        return {
            "instructions": len(self.instructions),
            "protective": self.protective_instruction_count,
            "problematic": len(self.problematic_instructions),
            "dimensions_covered": len(self.dimensions_covered),
            "total_dimensions": len(self.all_dimensions),
            "coverage_fraction": round(self.coverage_fraction, 4),
            "has_protective": self.has_protective,
            "has_problematic": self.has_problematic,
            "covers_all_dimensions": self.covers_all_dimensions,
        }


_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_prompt_into_instructions(prompt: str) -> list[str]:
    """Break a system prompt into discrete, meaningful instructions.

    Splits on newlines and sentence boundaries, strips bullet/number markers,
    and drops fragments shorter than ``MIN_INSTRUCTION_WORDS`` words so that
    template references (e.g. ``{{fn:global_function_id}}``) and noise do not
    pollute the audit.

    Args:
        prompt: The raw system prompt text.

    Returns:
        The ordered list of instruction fragments.
    """
    if not prompt or not prompt.strip():
        return []
    instructions: list[str] = []
    for raw_line in prompt.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = _BULLET_RE.sub("", line, count=1)
        for part in _SENTENCE_SPLIT_RE.split(line):
            part = part.strip(" .-")
            if part and len(part.split()) >= MIN_INSTRUCTION_WORDS:
                instructions.append(part)
    return instructions


def classify_instruction(instruction: str) -> list[InstructionFinding]:
    """Label a single instruction across the taxonomy dimensions.

    An instruction may collect both protective and problematic findings
    (e.g. a line that discloses AI identity while also hiding another fact),
    reflecting the paper's observation that protective and problematic
    instructions coexist.

    Args:
        instruction: A single instruction fragment.

    Returns:
        Findings for every dimension the instruction matches.
    """
    findings: list[InstructionFinding] = []
    for dimension in DIMENSIONS:
        if any(pattern.search(instruction) for pattern in dimension.protective):
            findings.append(InstructionFinding(instruction, dimension.name, "protective"))
        if any(pattern.search(instruction) for pattern in dimension.problematic):
            findings.append(InstructionFinding(instruction, dimension.name, "problematic"))
    return findings


def audit_system_prompt(prompt: str) -> AuditReport:
    """Audit a full system prompt with the AISPA-style taxonomy.

    Args:
        prompt: The raw system prompt text (personality + role + rules, etc.).

    Returns:
        An ``AuditReport`` with per-instruction findings, dimension coverage,
        and the problematic-instruction list.
    """
    instructions = split_prompt_into_instructions(prompt)
    findings: list[InstructionFinding] = []
    for instruction in instructions:
        findings.extend(classify_instruction(instruction))

    covered: set[str] = set()
    problematic_instructions: list[str] = []
    seen_problematic: set[str] = set()
    for finding in findings:
        if finding.classification == "protective":
            covered.add(finding.dimension)
        elif finding.instruction not in seen_problematic:
            seen_problematic.add(finding.instruction)
            problematic_instructions.append(finding.instruction)

    all_names = [dimension.name for dimension in DIMENSIONS]
    return AuditReport(
        prompt=prompt,
        instructions=instructions,
        findings=findings,
        dimensions_covered=sorted(covered),
        all_dimensions=all_names,
        problematic_instructions=problematic_instructions,
    )


def audit_agent_settings(
    personality: str,
    role: str = "",
    rules: str = "",
) -> AuditReport:
    """Audit the combined system prompt of an ADK agent.

    Concatenates the personality, role and rules text -- the three sources that
    form an agent's system prompt in Agent Studio -- and audits the result.

    Args:
        personality: The ``SettingsPersonality.custom`` text.
        role: The ``SettingsRole`` text (value + additional info + custom).
        rules: The agent's rules text.

    Returns:
        An ``AuditReport`` over the combined prompt.
    """
    sections = [personality.strip(), role.strip(), rules.strip()]
    combined = "\n".join(section for section in sections if section)
    return audit_system_prompt(combined)
