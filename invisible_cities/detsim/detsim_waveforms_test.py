import numpy as np

from . detsim_waveforms import histogram_s1_times

def test_histogram_s1_times():
    s1_times = np.array([[1, 2, 3, 4, 5]])
    buffer_length = 5
    bin_width     = 1
    start_time    = 1

    expected = np.array([[1, 1, 1, 1, 1]])
    h = histogram_s1_times(s1_times, buffer_length, bin_width, start_time)

    np.testing.assert_allclose(h, expected)
