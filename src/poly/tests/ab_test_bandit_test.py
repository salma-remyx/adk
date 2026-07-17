"""Tests for the A/B-test multi-armed-bandit recommender.

Copyright PolyAI Limited
"""

import random
import unittest
from unittest.mock import patch

from poly.ab_test_bandit import (
    ArmStats,
    probability_best,
    recommend_traffic_split,
    recommend_winner,
    stats_from_rewards,
)
from poly.cli import AgentStudioCLI
from poly.tests.ab_test_test import SAMPLE_AB_TEST
from poly.tests.project_test import TEST_DIR


class ArmStatsTest(unittest.TestCase):
    """Tests for the ArmStats sufficient-statistic container."""

    def test_mean_and_failures_from_binary_rewards(self):
        """successes/trials yield the empirical mean and its complement."""
        stats = ArmStats(successes=3, trials=10)
        self.assertAlmostEqual(stats.mean, 0.3)
        self.assertAlmostEqual(stats.failures, 7.0)

    def test_mean_is_zero_for_untried_arm(self):
        """An arm with no observations has zero mean and no failures."""
        empty = ArmStats(successes=0, trials=0)
        self.assertEqual(empty.mean, 0.0)
        self.assertEqual(empty.failures, 0.0)


class StatsFromRewardsTest(unittest.TestCase):
    """Tests for stats_from_rewards aggregation and validation."""

    def test_aggregates_graded_rewards(self):
        """Reward mass is summed and trials counted, preserving order."""
        stats = stats_from_rewards({"a": [0.0, 0.5, 1.0], "b": [1.0, 1.0]})
        self.assertAlmostEqual(stats["a"].successes, 1.5)
        self.assertEqual(stats["a"].trials, 3)
        self.assertEqual(stats["b"].trials, 2)
        self.assertEqual(list(stats), ["a", "b"])

    def test_rejects_reward_outside_unit_interval(self):
        """Rewards outside [0, 1] raise ValueError."""
        with self.assertRaises(ValueError):
            stats_from_rewards({"a": [1.5]})


class RecommendWinnerTest(unittest.TestCase):
    """Tests for the best-arm (winner) recommendation."""

    def test_picks_higher_mean_arm(self):
        """The arm with the higher empirical mean is selected."""
        stats = stats_from_rewards({"control": [0, 0, 1], "variant": [1, 1, 1, 0]})
        self.assertEqual(recommend_winner(stats), "variant")

    def test_ties_break_to_insertion_order(self):
        """Ties resolve to the first arm in insertion order."""
        stats = stats_from_rewards({"first": [1, 1], "second": [1, 1]})
        self.assertEqual(recommend_winner(stats), "first")

    def test_empty_raises(self):
        """No arms raises ValueError."""
        with self.assertRaises(ValueError):
            recommend_winner({})


class RecommendTrafficSplitTest(unittest.TestCase):
    """Tests for the Thompson-sampled traffic-split recommendation."""

    def test_returns_percentage_in_range(self):
        """A recommendation is always an integer percentage in [0, 100]."""
        stats = stats_from_rewards({"control": [1, 0], "variant": [0, 1]})
        pct = recommend_traffic_split(stats, "variant", rng=random.Random(0))
        self.assertIsInstance(pct, int)
        self.assertTrue(0 <= pct <= 100)

    def test_dominant_variant_gets_most_traffic(self):
        """A clearly better variant receives the majority of traffic on average."""
        stats = stats_from_rewards({"control": [1] + [0] * 9, "variant": [1] * 9 + [0]})
        shares = [
            recommend_traffic_split(stats, "variant", rng=random.Random(seed)) for seed in range(50)
        ]
        self.assertGreater(sum(shares) / len(shares), 50)

    def test_two_untried_arms_split_evenly(self):
        """Two untried arms (symmetric uniform priors) split traffic ~50/50 on average."""
        stats = {
            "control": ArmStats(successes=0, trials=0),
            "variant": ArmStats(successes=0, trials=0),
        }
        shares = [
            recommend_traffic_split(stats, "variant", rng=random.Random(seed))
            for seed in range(200)
        ]
        average = sum(shares) / len(shares)
        self.assertTrue(40 < average < 60, f"untried split {average} not near 50")

    def test_unknown_variant_raises(self):
        """An unknown variant arm raises ValueError."""
        stats = stats_from_rewards({"a": [1], "b": [0]})
        with self.assertRaises(ValueError):
            recommend_traffic_split(stats, "nope")

    def test_single_arm_raises(self):
        """A split cannot be recommended for a single arm."""
        stats = stats_from_rewards({"a": [1]})
        with self.assertRaises(ValueError):
            recommend_traffic_split(stats, "a")


class ProbabilityBestTest(unittest.TestCase):
    """Tests for the posterior win-probability estimate."""

    def test_dominant_arm_has_high_probability(self):
        """A clearly dominant arm is the best arm with high probability."""
        stats = stats_from_rewards({"control": [0] * 9 + [1], "variant": [1] * 9 + [0]})
        self.assertGreater(probability_best(stats, "variant", rng=random.Random(0)), 0.95)

    def test_single_arm_is_certainly_best(self):
        """The only arm is the best arm with probability 1.0."""
        stats = stats_from_rewards({"only": [1, 0]})
        self.assertEqual(probability_best(stats, "only"), 1.0)


class BanditIntegrationWithCLITest(unittest.TestCase):
    """Drive the existing A/B-test CLI commands with bandit recommendations."""

    def setUp(self):
        """Patch project loading so the real CLI commands run against a mock."""
        patcher = patch("poly.cli.AgentStudioCLI._load_project")
        self.mock_load = patcher.start()
        self.proj = self.mock_load.return_value
        self.proj.get_active_ab_test.return_value = dict(SAMPLE_AB_TEST)
        self.addCleanup(patch.stopall)

    def _conversation_rewards(self) -> dict[str, list[int]]:
        """Return simulated per-deployment interaction signal (variant dominates)."""
        return {
            SAMPLE_AB_TEST["control_deployment_id"]: [1] + [0] * 9,
            SAMPLE_AB_TEST["variant_deployment_id"]: [1] * 9 + [0],
        }

    @patch("poly.cli.json_print")
    def test_bandit_recommendation_feeds_ab_test_update(self, mock_json):
        """The bandit's traffic split is accepted by the existing update flow."""
        # Active test currently routes 0% of traffic to the variant.
        self.proj.get_active_ab_test.return_value = dict(SAMPLE_AB_TEST, traffic_percentage=0)
        stats = stats_from_rewards(self._conversation_rewards())
        recommended = recommend_traffic_split(
            stats, SAMPLE_AB_TEST["variant_deployment_id"], rng=random.Random(42)
        )
        self.assertNotEqual(recommended, 0)
        self.assertTrue(0 <= recommended <= 100)

        AgentStudioCLI.ab_test_update(TEST_DIR, traffic_percentage=recommended, output_json=True)

        self.proj.update_ab_test.assert_called_once_with("ab-001", recommended)
        self.assertTrue(mock_json.call_args[0][0]["success"])

    @patch("poly.cli.json_print")
    def test_bandit_winner_feeds_ab_test_end(self, mock_json):
        """The bandit's winner flows through the existing end-and-promote flow."""
        deployments = [
            {
                "id": SAMPLE_AB_TEST["control_deployment_id"],
                "version_hash": "live00000",
                "deployment_metadata": {"deployment_message": "v1 stable"},
            },
            {
                "id": SAMPLE_AB_TEST["variant_deployment_id"],
                "version_hash": "variant111",
                "deployment_metadata": {"deployment_message": "v2 candidate"},
            },
        ]
        self.proj.get_deployments.return_value = (deployments, {})
        self.proj.end_ab_test.return_value = dict(SAMPLE_AB_TEST, status="ended")

        stats = stats_from_rewards(self._conversation_rewards())
        winner_id = recommend_winner(stats)
        # Map the winning deployment back to its version hash, as the team would
        # before calling `poly deployments ab-test end --chosen-version`.
        dep_by_id = {dep["id"]: dep for dep in deployments}
        chosen_version = dep_by_id[winner_id]["version_hash"]

        AgentStudioCLI.ab_test_end(TEST_DIR, chosen_version=chosen_version, output_json=True)

        self.assertEqual(winner_id, SAMPLE_AB_TEST["variant_deployment_id"])
        self.proj.end_ab_test.assert_called_once_with("ab-001", winner_id)
        # A variant winner is promoted to live by the existing flow.
        self.proj.promote_deployment.assert_called_once()
        self.assertTrue(mock_json.call_args[0][0]["success"])


if __name__ == "__main__":
    unittest.main()
