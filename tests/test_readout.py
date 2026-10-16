import unittest

import numpy as np

from backscatter.readout import quantize_to_channel_centers


class ReadoutTests(unittest.TestCase):
    def test_even_array_quantizes_to_half_pitch_centers(self):
        values = np.asarray([-1.49, -0.51, -0.49, 0.0, 0.49, 0.51, 1.49])
        result = quantize_to_channel_centers(values, pitch_mm=1.0)
        expected = np.asarray([-1.5, -0.5, -0.5, 0.5, 0.5, 0.5, 1.5])
        np.testing.assert_array_equal(result, expected)

    def test_nonpositive_pitch_is_rejected(self):
        with self.assertRaises(ValueError):
            quantize_to_channel_centers(np.asarray([0.0]), pitch_mm=0.0)


if __name__ == "__main__":
    unittest.main()
