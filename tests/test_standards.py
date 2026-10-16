import unittest

import numpy as np

from backscatter.standards import (
    _measure_double_rod_response,
    estimate_fwhm_kde,
    fit_erf_edge_response,
    mtf_from_edge_spread,
)


class StandardTargetTests(unittest.TestCase):
    def test_gaussian_fwhm_estimate(self):
        rng = np.random.default_rng(24061873)
        values = rng.normal(loc=-30.0, scale=2.0, size=20000)
        estimate = estimate_fwhm_kde(values)
        expected = 2.0 * np.sqrt(8.0 * np.log(2.0))
        self.assertEqual(estimate["status"], "ok")
        self.assertAlmostEqual(estimate["mode"], -30.0, delta=0.5)
        self.assertAlmostEqual(estimate["fwhm"], expected, delta=0.25)

    def test_fwhm_refuses_tiny_samples(self):
        estimate = estimate_fwhm_kde(np.arange(10.0))
        self.assertEqual(estimate["status"], "insufficient_events")
        self.assertIsNone(estimate["fwhm"])

    def test_gaussian_edge_mtf50(self):
        coordinate = np.linspace(-40.0, 40.0, 1601)
        sigma = 2.0
        from scipy.special import erf

        edge = 0.5 * (
            1.0 + erf(coordinate / (np.sqrt(2.0) * sigma))
        )
        report = mtf_from_edge_spread(coordinate, edge)
        expected = np.sqrt(np.log(2.0)) / (
            np.sqrt(2.0) * np.pi * sigma
        )
        self.assertAlmostEqual(
            report["mtf50_cycles_per_mm"],
            expected,
            delta=0.01,
        )

    def test_count_aware_edge_fit_recovers_gaussian_width(self):
        from scipy.special import erf

        rng = np.random.default_rng(24061873)
        coordinate = np.arange(-30.0, 31.0, 1.0)
        sigma = 3.0
        relative_response = 0.5 * (
            1.0
            + erf(coordinate / (np.sqrt(2.0) * sigma))
        )
        reference_counts = rng.poisson(300.0, len(coordinate))
        edge_counts = rng.poisson(
            reference_counts * relative_response
        )
        report = fit_erf_edge_response(
            coordinate,
            edge_counts,
            reference_counts,
            edge_primaries=2000000,
            reference_primaries=2000000,
            n_bootstrap=40,
            random_seed=24061874,
        )
        expected_mtf50 = np.sqrt(np.log(2.0)) / (
            np.sqrt(2.0) * np.pi * sigma
        )
        self.assertEqual(report["status"], "ok")
        self.assertAlmostEqual(
            report["gaussian_sigma_mm"],
            sigma,
            delta=0.35,
        )
        self.assertAlmostEqual(
            report["mtf50_cycles_per_mm"],
            expected_mtf50,
            delta=0.01,
        )
        self.assertEqual(report["bootstrap"]["successful"], 40)

    def test_double_rod_measurement_resolves_well_separated_peaks(self):
        rng = np.random.default_rng(24061873)
        centers = np.asarray([-10.0, 10.0])
        values = np.concatenate(
            [
                rng.normal(centers[0], 2.0, 4000),
                rng.normal(centers[1], 2.0, 4000),
            ]
        )
        report = _measure_double_rod_response(
            x_values=values,
            centers=centers,
            bandwidth_scale=1.0,
        )
        self.assertEqual(report["status"], "ok")
        self.assertTrue(
            report["metrics"]["operationally_resolved_at_0p735"]
        )

    def test_double_rod_measurement_rejects_merged_peaks(self):
        rng = np.random.default_rng(24061874)
        centers = np.asarray([-2.0, 2.0])
        values = np.concatenate(
            [
                rng.normal(centers[0], 4.0, 4000),
                rng.normal(centers[1], 4.0, 4000),
            ]
        )
        report = _measure_double_rod_response(
            x_values=values,
            centers=centers,
            bandwidth_scale=1.0,
        )
        self.assertFalse(
            report["metrics"]["operationally_resolved_at_0p735"]
        )


if __name__ == "__main__":
    unittest.main()
