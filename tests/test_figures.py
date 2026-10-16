import unittest

import numpy as np

from backscatter.figures import _histogram_fwhm


class FigureMetricTests(unittest.TestCase):
    def test_histogram_fwhm_recovers_gaussian_width(self):
        x = np.arange(400.0, 622.0, 2.0)
        expected_fwhm = 50.0
        sigma = expected_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        counts = 1000.0 * np.exp(-0.5 * ((x - 510.0) / sigma) ** 2)
        result = _histogram_fwhm(x, counts, 420.0, 600.0)
        self.assertAlmostEqual(result["mode_keV"], 510.0)
        self.assertAlmostEqual(result["fwhm_keV"], expected_fwhm, delta=0.2)
        self.assertAlmostEqual(
            result["resolution_percent"],
            100.0 * expected_fwhm / 510.0,
            delta=0.05,
        )

    def test_histogram_fwhm_rejects_unbracketed_peak(self):
        with self.assertRaisesRegex(ValueError, "not bracketed"):
            _histogram_fwhm(
                np.asarray([500.0, 502.0, 504.0]),
                np.asarray([2.0, 3.0, 2.0]),
                500.0,
                504.0,
            )


if __name__ == "__main__":
    unittest.main()
