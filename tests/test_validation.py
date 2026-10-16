import unittest

from backscatter.validation import _ci_standard_error, _difference_z


class ValidationTests(unittest.TestCase):
    def test_ci_width_is_converted_to_standard_error(self):
        self.assertAlmostEqual(_ci_standard_error([8.04, 11.96]), 1.0)

    def test_difference_z_combines_independent_uncertainties(self):
        value = _difference_z(10.0, 1.0, 12.0, 1.0)
        self.assertAlmostEqual(value, 2.0**0.5)

    def test_invalid_ci_returns_none(self):
        self.assertIsNone(_ci_standard_error([2.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
