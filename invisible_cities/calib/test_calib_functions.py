import unittest
import numpy as np
from calib_functions import (bin_waveforms, spaced_integrals, integral_limits, 
                             valid_integral_limits)

class TestCalibrationFunctions(unittest.TestCase):

    def test_bin_waveforms(self):
        waveforms = np.array([[1, 2, 3], [4, 5, 6]])
        bins = [0, 2, 4, 6]
        expected_result = np.array([[1, 0], [1, 0]])
        result = bin_waveforms(waveforms, bins)
        np.testing.assert_array_equal(result, expected_result)

    def test_spaced_integrals(self):
        wfs = np.array([[1, 2, 3], [4, 5, 6]])
        limits = np.array([0, 2])
        expected_result = np.array([[3], [11]])  # Integrals calculated as per limits
        result = spaced_integrals(wfs, limits)
        np.testing.assert_array_equal(result, expected_result)

    def test_integral_limits(self):
        sample_width = 1.0
        n_integrals = 3
        integral_start = 1.0
        integral_width = 2.0
        period = 1.0
        corr, anti = integral_limits(sample_width, n_integrals, integral_start, 
                                     integral_width, period)
        expected_corr = np.array([1, 3, 5, 4, 6, 8])
        expected_anti = expected_corr - 2
        np.testing.assert_array_equal(corr, expected_corr)
        np.testing.assert_array_equal(anti, expected_anti)

    def test_valid_integral_limits(self):
        sample_width = 1.0
        n_integrals = 3
        integral_start = 1.0
        integral_width = 2.0
        period = 1.0
        buffer_length = 10
        corr, anti = valid_integral_limits(sample_width, n_integrals, 
                                            integral_start, integral_width, 
                                            period, buffer_length)
        self.assertTrue(np.all(corr >= 0) and np.all(corr < buffer_length))
        self.assertTrue(np.all(anti >= 0) and np.all(anti < buffer_length))

if __name__ == '__main__':
    unittest.main()
