"""Power-maximising proxy metrics for accelerated A/B tests.

Adapted from "Learning Metrics that Maximise Power for Accelerated
A/B-Tests" (arXiv:2402.03915). A North Star metric (long-term revenue,
retention, ...) is delayed and insensitive, so A/B tests must run long and
still suffer type-II errors. The paper learns a metric transformation that
minimises the variance of the treatment-effect estimator while preserving
its direction, which maximises statistical power and lets the experiment
reach a decision with fewer samples.

This module ports that core objective onto ADK's ``poly deployments
ab-test`` domain. Given per-arm outcome observations (and optional
pre-experiment covariates measured on the same units), it learns the
closed-form variance-minimising adjustment coefficients and reports the
accelerated proxy metric alongside the raw one.

Adaptations (Mode 2):
  * The paper's general learned estimator and bespoke optimizer are
    replaced by the closed-form least-squares adjustment, which is the
    exact power-maximising solution for a linear metric combination.
  * The paper's separate evaluation harness is cut (a downstream concern).
  * Inputs are ADK-native A/B-test records plus user-supplied per-arm
    observations rather than the paper's experimental datasets.

Copyright PolyAI Limited
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

CONTROL_ARM_KEY = "control"
VARIANT_ARM_KEY = "variant"

# z -> two-sided p-value uses the standard normal CDF; 95% CI half-width
# uses the 0.975 quantile of the standard normal.
_Z_975 = 1.959963984540054


@dataclass
class EffectEstimate:
    """A two-sample treatment-effect estimate for a single metric.

    effect is the variant mean minus the control mean; se is the
    Welch (unequal-variance) standard error of that difference.
    """

    effect: float
    se: float
    n_control: int
    n_variant: int

    @property
    def z(self) -> float:
        """Standardised effect (effect / se)."""
        if self.se == 0.0:
            if self.effect > 0.0:
                return math.inf
            if self.effect < 0.0:
                return -math.inf
            return 0.0
        return self.effect / self.se

    @property
    def p_value(self) -> float:
        """Two-sided p-value from the normal approximation."""
        z_score = self.z
        if not math.isfinite(z_score):
            return 0.0 if self.effect != 0.0 else 1.0
        return 1.0 - math.erf(abs(z_score) / math.sqrt(2.0))

    @property
    def ci_half_width(self) -> float:
        """Half-width of an approximate 95% confidence interval for the effect."""
        return _Z_975 * self.se


@dataclass
class PowerMaximisedMetric:
    """Raw versus power-maximised proxy metric for an A/B test.

    variance_reduction is the fraction of the effect estimator's variance
    removed by the learned adjustment (1 - (adj_se / raw_se) ** 2), i.e. the
    power gained on the A/B-test decision itself; sample_acceleration is the
    reciprocal, the factor by which the required sample size shrinks.
    """

    raw: EffectEstimate
    adjusted: EffectEstimate
    coefficients: list[float]
    variance_reduction: float
    sample_acceleration: float
    arm_labels: dict[str, str]


def _mean(values: Sequence[float]) -> float:
    """Arithmetic mean of a non-empty sequence."""
    return sum(values) / len(values)


def _sample_variance(values: Sequence[float]) -> float:
    """Unbiased sample variance (ddof=1); 0.0 for fewer than two values."""
    n = len(values)
    if n < 2:
        return 0.0
    mu = _mean(values)
    return sum((v - mu) ** 2 for v in values) / (n - 1)


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve ``A x = b`` by Gaussian elimination with partial pivoting.

    Raises:
        ValueError: if ``A`` is singular (collinear covariates).
    """
    augmented = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]
    n = len(augmented)
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            raise ValueError(
                "covariates are collinear; cannot fit a unique power-maximising metric"
            )
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        pivot_value = augmented[col][col]
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col] / pivot_value
            for cell in range(col, n + 1):
                augmented[row][cell] -= factor * augmented[col][cell]
    return [augmented[i][n] / augmented[i][i] for i in range(n)]


def _least_squares_coefficients(
    centered_columns: list[list[float]], outcomes: Sequence[float]
) -> list[float]:
    """Closed-form least-squares coefficients for mean-centered covariates.

    Solves the normal equations ``(X^T X) b = X^T y`` over the centered
    covariate columns, which is the variance-minimising linear combination.
    """
    k = len(centered_columns)
    if k == 0:
        return []
    n = len(outcomes)
    matrix = [
        [sum(centered_columns[i][m] * centered_columns[j][m] for m in range(n)) for j in range(k)]
        for i in range(k)
    ]
    rhs = [sum(centered_columns[i][m] * outcomes[m] for m in range(n)) for i in range(k)]
    return _solve_linear_system(matrix, rhs)


def fit_power_coefficients(
    outcomes: Sequence[float], covariate_columns: Sequence[Sequence[float]]
) -> list[float]:
    """Learn the variance-minimising (power-maximising) adjustment coefficients.

    Coefficients are learned with the CUPED construction: each covariate is
    mean-centered, then the closed-form least-squares solution minimises the
    residual variance of the outcome. For a single covariate this reduces to
    ``Cov(Y, X) / Var(X)``. Callers should pass outcomes pooled across both
    arms so the covariate means used for centering are the experiment-wide
    means.

    Args:
        outcomes: Per-unit outcome values (e.g. the North Star metric).
        covariate_columns: One sequence per candidate pre-experiment signal,
            each parallel to ``outcomes``.

    Returns:
        One coefficient per covariate column; an empty list when no
        covariates are supplied.

    Raises:
        ValueError: if a covariate is not parallel to the outcomes or has
            zero variance.
    """
    columns = [list(col) for col in covariate_columns]
    for col in columns:
        if len(col) != len(outcomes):
            raise ValueError("each covariate must have one value per observation")
        if _sample_variance(col) <= 0.0:
            raise ValueError("covariate has zero variance; it carries no power signal")
    centered = [[v - _mean(col) for v in col] for col in columns]
    return _least_squares_coefficients(centered, outcomes)


def effect_estimate(
    control_outcomes: Sequence[float], variant_outcomes: Sequence[float]
) -> EffectEstimate:
    """Estimate the variant-minus-control effect (Welch two-sample).

    Args:
        control_outcomes: Per-unit outcomes for the control arm.
        variant_outcomes: Per-unit outcomes for the variant arm.

    Returns:
        An :class:`EffectEstimate` with the mean difference and its SE.

    Raises:
        ValueError: if either arm has no observations.
    """
    n_control = len(control_outcomes)
    n_variant = len(variant_outcomes)
    if n_control == 0 or n_variant == 0:
        raise ValueError("each arm needs at least one observation")
    mean_control = _mean(control_outcomes)
    mean_variant = _mean(variant_outcomes)
    se = math.sqrt(
        _sample_variance(control_outcomes) / n_control
        + _sample_variance(variant_outcomes) / n_variant
    )
    return EffectEstimate(
        effect=mean_variant - mean_control,
        se=se,
        n_control=n_control,
        n_variant=n_variant,
    )


def _pooled_covariate_columns(
    covariates_by_arm: dict[str, Sequence[Sequence[float]]] | None,
) -> list[list[float]]:
    """Concatenate control and variant covariate columns into pooled columns."""
    if not covariates_by_arm:
        return []
    control_cols = [list(col) for col in covariates_by_arm.get(CONTROL_ARM_KEY, [])]
    variant_cols = [list(col) for col in covariates_by_arm.get(VARIANT_ARM_KEY, [])]
    if len(control_cols) != len(variant_cols):
        raise ValueError("control and variant must provide the same number of covariates")
    return [control + variant for control, variant in zip(control_cols, variant_cols, strict=True)]


def _adjust_arm(
    outcomes: list[float],
    arm_covariates: list[list[float]],
    coefficients: list[float],
    pooled_means: list[float],
) -> list[float]:
    """Apply the learned adjustment to one arm using experiment-wide covariate means."""
    adjusted = list(outcomes)
    for col, coef, mean in zip(arm_covariates, coefficients, pooled_means, strict=True):
        adjusted = [y - coef * (x - mean) for y, x in zip(adjusted, col, strict=True)]
    return adjusted


def power_maximised_metric(
    ab_test: dict | None,
    outcomes_by_arm: dict[str, Sequence[float]],
    covariates_by_arm: dict[str, Sequence[Sequence[float]]] | None = None,
) -> PowerMaximisedMetric:
    """Compute the raw and power-maximised proxy metric for an A/B test.

    The adjustment coefficient is learned on the pooled control+variant
    sample and applied with experiment-wide covariate means, so a balanced
    pre-experiment covariate preserves the treatment-effect direction while
    shrinking its variance. This is the analytic power optimum from
    arXiv:2402.03915 expressed on ADK-native A/B-test data.

    Args:
        ab_test: An A/B-test record as returned by
            :meth:`poly.project.AgentStudioProject.list_ab_tests` (used to
            label arms by deployment id); ``None`` for generic labels.
        outcomes_by_arm: Per-unit outcomes keyed by ``control`` / ``variant``.
        covariates_by_arm: Optional pre-experiment covariates keyed by arm,
            each arm mapping to one sequence per covariate, parallel to that
            arm's outcomes.

    Returns:
        A :class:`PowerMaximisedMetric` comparing the raw and accelerated
        effect estimates.

    Raises:
        ValueError: if an arm is missing outcomes or a covariate is not
            parallel to the outcomes, has zero variance, or is collinear.
    """
    for arm in (CONTROL_ARM_KEY, VARIANT_ARM_KEY):
        if arm not in outcomes_by_arm:
            raise ValueError(f"outcomes_by_arm is missing the '{arm}' arm")

    control_outcomes = list(outcomes_by_arm[CONTROL_ARM_KEY])
    variant_outcomes = list(outcomes_by_arm[VARIANT_ARM_KEY])
    pooled_outcomes = control_outcomes + variant_outcomes

    pooled_columns = _pooled_covariate_columns(covariates_by_arm)
    centered_columns: list[list[float]] = []
    pooled_means: list[float] = []
    for col in pooled_columns:
        if len(col) != len(pooled_outcomes):
            raise ValueError("each covariate must have one value per observation")
        if _sample_variance(col) <= 0.0:
            raise ValueError("covariate has zero variance; it carries no power signal")
        mean = _mean(col)
        pooled_means.append(mean)
        centered_columns.append([v - mean for v in col])
    coefficients = _least_squares_coefficients(centered_columns, pooled_outcomes)

    control_covariates = (
        [list(col) for col in covariates_by_arm.get(CONTROL_ARM_KEY, [])]
        if covariates_by_arm
        else []
    )
    variant_covariates = (
        [list(col) for col in covariates_by_arm.get(VARIANT_ARM_KEY, [])]
        if covariates_by_arm
        else []
    )
    adjusted_control = _adjust_arm(control_outcomes, control_covariates, coefficients, pooled_means)
    adjusted_variant = _adjust_arm(variant_outcomes, variant_covariates, coefficients, pooled_means)

    raw = effect_estimate(control_outcomes, variant_outcomes)
    adjusted = effect_estimate(adjusted_control, adjusted_variant)
    variance_reduction, sample_acceleration = _power_gain(raw.se, adjusted.se)

    record = ab_test or {}
    arm_labels = {
        CONTROL_ARM_KEY: str(record.get("control_deployment_id", CONTROL_ARM_KEY)),
        VARIANT_ARM_KEY: str(record.get("variant_deployment_id", VARIANT_ARM_KEY)),
    }

    return PowerMaximisedMetric(
        raw=raw,
        adjusted=adjusted,
        coefficients=coefficients,
        variance_reduction=variance_reduction,
        sample_acceleration=sample_acceleration,
        arm_labels=arm_labels,
    )


def _power_gain(raw_se: float, adjusted_se: float) -> tuple[float, float]:
    """Variance-reduction and sample-acceleration from the effect SEs.

    Returns (variance_reduction, sample_acceleration) where the reduction is
    measured on the effect estimator's variance (the power of the A/B-test
    decision), not the raw outcome variance. Both are clamped so that no
    adjustment yields (0.0, 1.0).
    """
    if raw_se == 0.0:
        return 0.0, 1.0
    variance_reduction = max(0.0, min(1.0, 1.0 - (adjusted_se / raw_se) ** 2))
    if variance_reduction >= 1.0:
        return 1.0, math.inf
    sample_acceleration = 1.0 / (1.0 - variance_reduction)
    return variance_reduction, sample_acceleration
