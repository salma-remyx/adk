"""Multi-armed-bandit recommendations for deployment A/B tests.

Adapted from Yang et al., "A Multi-Armed Bandit Approach to Online Selection
and Evaluation of Generative Models" (arXiv:2406.07451), which allocates
traffic adaptively across candidate generative models so the best one is
identified with the fewest sampled interactions, then stops with a winner.

This module applies that core idea to ADK deployment A/B tests. Each
deployment version is a bandit arm and each conversation-interaction outcome
is a reward in ``[0, 1]``. Thompson sampling over a Beta posterior recommends
the next traffic split for ``poly deployments ab-test update`` and the
posterior-best deployment for ``poly deployments ab-test end``. The existing
platform-sync methods (``create_ab_test`` / ``update_ab_test`` /
``end_ab_test``) are unchanged — these helpers only compute the recommendation
that feeds their ``traffic_percentage`` / ``chosen_deployment_id`` arguments.

Adaptation notes (Mode 2 — auxiliary components substituted):
  * The paper's learned generative-model reward estimator is replaced by a
    parameter-free Beta sufficient-statistic estimator over observed
    interaction rewards (``ArmStats``).
  * The paper's bespoke sequential stopping framework is replaced by ADK's
    native ``update_ab_test`` / ``end_ab_test`` flow; ``probability_best``
    exposes the confidence signal that would drive the "stop with a winner"
    decision.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArmStats:
    """Sufficient statistics for one bandit arm (a deployment version).

    Rewards are graded in ``[0, 1]``: ``0`` is a failed interaction, ``1`` a
    fully successful one, and intermediate values (e.g. a partial goal
    completion) are allowed. ``successes`` accumulates reward mass and
    ``trials`` counts interactions, so a Beta(``successes`` + 1,
    ``failures`` + 1) posterior summarises everything the bandit needs.

    Attributes:
        successes: Total reward mass observed for the arm.
        trials: Number of interactions observed for the arm.
    """

    successes: float
    trials: int

    @property
    def failures(self) -> float:
        """Return the reward mass in ``[0, 1]`` that was not a success."""
        return max(self.trials - self.successes, 0.0)

    @property
    def mean(self) -> float:
        """Return the empirical mean reward; ``0.0`` for an arm with no data."""
        return self.successes / self.trials if self.trials else 0.0


def stats_from_rewards(rewards_by_arm: Mapping[str, Sequence[float]]) -> dict[str, ArmStats]:
    """Aggregate per-arm reward observations into sufficient statistics.

    Args:
        rewards_by_arm: Maps each arm name (typically a deployment id) to the
            sequence of interaction rewards it has observed. Every reward must
            lie in ``[0, 1]``.

    Returns:
        A mapping from arm name to its :class:`ArmStats`, preserving the
        input insertion order.

    Raises:
        ValueError: If any reward falls outside ``[0, 1]``.
    """
    stats: dict[str, ArmStats] = {}
    for arm, rewards in rewards_by_arm.items():
        values = list(rewards)
        for value in values:
            if value < 0.0 or value > 1.0:
                raise ValueError(f"Reward {value!r} for arm {arm!r} is outside [0, 1].")
        stats[arm] = ArmStats(successes=float(sum(values)), trials=len(values))
    return stats


def _beta_sample(stats: ArmStats, rng: random.Random) -> float:
    """Draw one sample from the arm's Beta(successes+1, failures+1) posterior."""
    return rng.betavariate(stats.successes + 1.0, stats.failures + 1.0)


def recommend_traffic_split(
    stats_by_arm: Mapping[str, ArmStats],
    variant_arm: str,
    control_arm: str | None = None,
    *,
    rng: random.Random | None = None,
) -> int:
    """Recommend a traffic percentage for ``variant_arm`` via Thompson sampling.

    One Thompson sample is drawn from each arm's posterior and the variant's
    share of the total sampled reward mass becomes its recommended traffic
    percentage. An untried arm has a uniform Beta(1, 1) prior, so two unseen
    arms split traffic roughly evenly and a new arm keeps drawing exploratory
    traffic until evidence accumulates — this is the exploration that lets the
    bandit favour the best arm while still querying sub-optimal ones sparingly.

    Args:
        stats_by_arm: Per-arm sufficient statistics.
        variant_arm: The arm whose traffic share is recommended (the returned
            percentage routes traffic to this deployment).
        control_arm: The baseline arm. When ``None``, the first arm that is not
            the variant (in insertion order) is used.
        rng: Seeded RNG for reproducible draws. When ``None`` the process-level
            RNG is used.

    Returns:
        Integer percentage in ``[0, 100]`` of traffic to route to
        ``variant_arm``.

    Raises:
        ValueError: If ``variant_arm`` is unknown or fewer than two arms are
            provided.
    """
    if variant_arm not in stats_by_arm:
        raise ValueError(f"Unknown variant arm: {variant_arm!r}.")
    if len(stats_by_arm) < 2:
        raise ValueError("A traffic split requires at least two arms.")

    if control_arm is None:
        control_arm = next(name for name in stats_by_arm if name != variant_arm)
    elif control_arm not in stats_by_arm:
        raise ValueError(f"Unknown control arm: {control_arm!r}.")

    sampler = rng if rng is not None else random
    variant_sample = _beta_sample(stats_by_arm[variant_arm], sampler)
    control_sample = _beta_sample(stats_by_arm[control_arm], sampler)
    total = variant_sample + control_sample
    share = variant_sample / total if total > 0 else 0.5
    return int(round(max(0.0, min(1.0, share)) * 100))


def recommend_winner(stats_by_arm: Mapping[str, ArmStats]) -> str:
    """Return the arm with the highest posterior-mean reward.

    This is the "stop with a winner" selection rule: the empirically best arm.
    Ties are broken by insertion order; with no observations the first arm is
    returned (uniform prior).

    Args:
        stats_by_arm: Per-arm sufficient statistics.

    Returns:
        The name of the best arm.

    Raises:
        ValueError: If ``stats_by_arm`` is empty.
    """
    if not stats_by_arm:
        raise ValueError("recommend_winner requires at least one arm.")
    return max(stats_by_arm, key=lambda name: stats_by_arm[name].mean)


def probability_best(
    stats_by_arm: Mapping[str, ArmStats],
    arm: str,
    *,
    rng: random.Random | None = None,
    samples: int = 2000,
) -> float:
    """Estimate the posterior probability that ``arm`` is the best arm.

    Uses Monte-Carlo Thompson sampling: ``samples`` joint posterior draws are
    taken and the fraction where ``arm`` has the largest sample is returned.
    This is the confidence signal for the "stop with a winner" decision — end
    the A/B test once it crosses a chosen threshold (e.g. ``0.95``).

    Args:
        stats_by_arm: Per-arm sufficient statistics.
        arm: The arm whose win probability is estimated.
        rng: Seeded RNG for reproducible estimates.
        samples: Number of posterior draws. Higher is more accurate.

    Returns:
        Posterior probability in ``[0, 1]`` that ``arm`` is the best arm.

    Raises:
        ValueError: If ``arm`` is unknown or ``samples`` is not positive.
    """
    if arm not in stats_by_arm:
        raise ValueError(f"Unknown arm: {arm!r}.")
    if samples <= 0:
        raise ValueError("samples must be positive.")
    if len(stats_by_arm) == 1:
        return 1.0

    sampler = rng if rng is not None else random
    names = list(stats_by_arm)
    wins = 0
    for _ in range(samples):
        drawn = max(names, key=lambda name: _beta_sample(stats_by_arm[name], sampler))
        if drawn == arm:
            wins += 1
    return wins / samples
