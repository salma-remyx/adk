"""Integration tests for the power-maximising A/B-test metric.

These tests exercise the NON-NEW ``poly.project`` module (fetching a real
A/B-test record through ``AgentStudioProject.list_ab_tests``) and feed that
record, plus per-arm observations, into ``poly.ab_test_metrics``. The
metric module is the analysis layer that the ab-test domain's data flows
into.

Copyright PolyAI Limited
"""

import unittest
from unittest.mock import MagicMock, patch

from poly.ab_test_metrics import (
    CONTROL_ARM_KEY,
    VARIANT_ARM_KEY,
    PowerMaximisedMetric,
    effect_estimate,
    fit_power_coefficients,
    power_maximised_metric,
)
from poly.project import AgentStudioProject
from poly.tests.project_test import PROJECT_DATA, TEST_DIR

SAMPLE_AB_TEST = {
    "id": "ab-001",
    "name": "v2 test",
    "control_deployment_id": "dep-live",
    "variant_deployment_id": "dep-variant",
    "traffic_percentage": 50,
    "status": "active",
}

# A pre-experiment covariate that tracks each unit's outcome within an arm
# (reducing variance) while staying balanced across arms (preserving the
# treatment-effect direction).
_CONTROL_OUTCOMES = [1.0, 1.2, 0.9, 1.1, 1.0, 0.8, 1.1, 1.2, 0.9, 1.0]
_VARIANT_OUTCOMES = [1.3, 1.5, 1.2, 1.4, 1.3, 1.1, 1.4, 1.5, 1.2, 1.3]
_PRE_COVARIATE = [0.85, 1.15, 0.95, 1.05, 1.00, 0.70, 1.00, 1.25, 0.80, 1.05]


class PowerMaximisedMetricIntegrationTest(unittest.TestCase):
    """Wiring between poly.project (data source) and poly.ab_test_metrics."""

    def setUp(self) -> None:
        """Patch save_config so the api_handler property stays offline."""
        patch.object(AgentStudioProject, "save_config").start()

    def tearDown(self) -> None:
        patch.stopall()

    def _project_with_ab_tests(self, records: list[dict]) -> AgentStudioProject:
        """Build a real project whose list_ab_tests returns ``records``."""
        project = AgentStudioProject.from_dict(PROJECT_DATA, TEST_DIR)
        project._api_handler = MagicMock()
        project._api_handler.list_ab_tests.return_value = {"ab_tests": records}
        return project

    def test_metric_consumes_record_from_project_list_ab_tests(self):
        """A record pulled via the existing project path feeds the metric."""
        project = self._project_with_ab_tests([SAMPLE_AB_TEST])
        ab_tests = project.list_ab_tests(limit=10)
        self.assertEqual(ab_tests, [SAMPLE_AB_TEST])

        result = power_maximised_metric(
            ab_tests[0],
            outcomes_by_arm={
                CONTROL_ARM_KEY: _CONTROL_OUTCOMES,
                VARIANT_ARM_KEY: _VARIANT_OUTCOMES,
            },
            covariates_by_arm={
                CONTROL_ARM_KEY: [_PRE_COVARIATE],
                VARIANT_ARM_KEY: [_PRE_COVARIATE],
            },
        )

        self.assertIsInstance(result, PowerMaximisedMetric)
        self.assertEqual(result.arm_labels[CONTROL_ARM_KEY], "dep-live")
        self.assertEqual(result.arm_labels[VARIANT_ARM_KEY], "dep-variant")
        # A correlated, balanced covariate removes variance...
        self.assertGreater(result.variance_reduction, 0.5)
        self.assertLess(result.variance_reduction, 1.0)
        self.assertGreaterEqual(result.sample_acceleration, 1.0)
        self.assertLess(result.adjusted.se, result.raw.se)
        # ...while preserving the treatment-effect direction exactly.
        self.assertAlmostEqual(result.adjusted.effect, result.raw.effect, places=6)
        self.assertEqual(len(result.coefficients), 1)

    def test_no_covariates_leaves_metric_unchanged(self):
        """Without a covariate the proxy equals the raw metric."""
        project = self._project_with_ab_tests([SAMPLE_AB_TEST])
        record = project.list_ab_tests()[0]

        result = power_maximised_metric(
            record,
            outcomes_by_arm={
                CONTROL_ARM_KEY: _CONTROL_OUTCOMES,
                VARIANT_ARM_KEY: _VARIANT_OUTCOMES,
            },
        )

        self.assertEqual(result.coefficients, [])
        self.assertAlmostEqual(result.variance_reduction, 0.0)
        self.assertAlmostEqual(result.sample_acceleration, 1.0)
        self.assertAlmostEqual(result.adjusted.effect, result.raw.effect, places=6)
        self.assertAlmostEqual(result.adjusted.se, result.raw.se, places=9)


class FitPowerCoefficientsTest(unittest.TestCase):
    """Unit checks for the closed-form learned coefficient."""

    def test_single_covariate_recovers_cov_over_var(self):
        """One covariate yields theta = Cov(Y, X) / Var(X)."""
        outcomes = [1.0, 2.0, 3.0, 4.0, 5.0]
        covariate = [2.0, 4.0, 6.0, 8.0, 10.0]  # outcomes = 0.5 * covariate
        coefficients = fit_power_coefficients(outcomes, [covariate])
        self.assertAlmostEqual(coefficients[0], 0.5, places=6)

    def test_collinear_covariates_raise(self):
        """Collinear covariates cannot define a unique metric."""
        outcomes = [1.0, 2.0, 3.0]
        covariate = [1.0, 2.0, 3.0]
        with self.assertRaises(ValueError):
            fit_power_coefficients(outcomes, [covariate, covariate])

    def test_zero_variance_covariate_raises(self):
        """A constant covariate carries no power signal."""
        with self.assertRaises(ValueError):
            fit_power_coefficients([1.0, 2.0, 3.0], [[5.0, 5.0, 5.0]])


class EffectEstimateTest(unittest.TestCase):
    """Sanity checks for the raw effect estimate."""

    def test_constant_lift_is_significant(self):
        """A constant variant lift gives effect == lift and a small p-value."""
        estimate = effect_estimate([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(estimate.effect, 1.0)
        self.assertLess(estimate.p_value, 1e-6)

    def test_empty_arm_raises(self):
        """An arm with no observations is a validation error."""
        with self.assertRaises(ValueError):
            effect_estimate([], [1.0, 2.0])
