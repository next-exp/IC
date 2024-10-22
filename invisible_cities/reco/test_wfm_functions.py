import unittest
import numpy as np
from your_module_path.wfm_functions import to_adc, to_pes, suppress_wf, noise_suppression, cwf_from_rwf, compare_cwf_blr

class TestWFMFunctions(unittest.TestCase):

    def setUp(self):
        # Sample data for tests
        self.waveforms = np.array([[1.0, 2.0, 3.0], [0.0, 0.5, 1.5], [3.0, 2.0, 1.0]])
        self.adc_to_pes = np.array([10, 20, 30])
        self.threshold = 1.0
        self.padding = 1
        self.pmtrwf = np.random.rand(10, 5, 100)  # 10 events, 5 channels, 100 samples
        self.event_list = [0, 1, 2]  # Example event indices
        self.calib_vectors = type('CalibVectors', (object,), {
            'coeff_c': np.random.rand(5),  # Mock coefficients
            'coeff_blr': np.random.rand(5)
        })()
        self.deconv_params = type('DeconvParams', (object,), {
            'n_baseline': 10,
            'thr_trigger': 0.1
        })()

    def test_to_adc(self):
        adc_wfs = to_adc(self.waveforms, self.adc_to_pes)
        expected = self.waveforms * self.adc_to_pes.reshape(-1, 1)
        np.testing.assert_array_equal(adc_wfs, expected)

    def test_to_pes(self):
        pes_wfs = to_pes(self.waveforms, self.adc_to_pes)
        expected = self.waveforms / self.adc_to_pes.reshape(-1, 1)
        np.testing.assert_array_equal(pes_wfs, expected)

    def test_suppress_wf(self):
        suppressed_wf = suppress_wf(self.waveforms[0], self.threshold, self.padding)
        expected = np.copy(self.waveforms[0])
        expected[expected <= self.threshold] = 0
        np.testing.assert_array_equal(suppressed_wf, expected)

    def test_noise_suppression(self):
        thresholds = [0.5, 0.5, 1.5]
        suppressed_wfs = noise_suppression(self.waveforms, thresholds, self.padding)
        for i in range(self.waveforms.shape[0]):
            expected = suppress_wf(self.waveforms[i], thresholds[i], self.padding)
            np.testing.assert_array_equal(suppressed_wfs[i], expected)

    def test_cwf_from_rwf(self):
        # Mocking the csf.means function to return a fixed value
        original_means = csf.means
        csf.means = lambda x: np.mean(x, axis=1)
        
        cwf = cwf_from_rwf(self.pmtrwf, self.event_list, self.calib_vectors, self.deconv_params)
        
        # Ensure the returned CWF has the same length as event_list
        self.assertEqual(len(cwf), len(self.event_list))

        # Restore original means function
        csf.means = original_means

    def test_compare_cwf_blr(self):
        # Generate mock CWF and BLR arrays
        cwf = np.random.rand(3, 5, 500)  # 3 events, 5 channels, 500 samples
        pmtblr = np.random.rand(3, 5, 500)  # 3 events, 5 channels, 500 samples
        diffs = compare_cwf_blr(cwf, pmtblr, self.event_list)
        
        # Ensure the output is as expected
        self.assertEqual(diffs.shape[0], len(self.event_list))

if __name__ == '__main__':
    unittest.main()
