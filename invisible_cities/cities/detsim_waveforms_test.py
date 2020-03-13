import numpy as np

from invisible_cities.cities.detsim_waveforms import *



def test_create_waveform():
    #data
    wf_bin_time = 1.
    bins = np.arange(0., 10. + wf_bin_time, wf_bin_time)
    nsamples = 2
    times = np.array([1, 2, 5, 6, 9, 10])
    pes   = np.array([2, 2, 2, 4, 1,  3])

    expected = np.array([0, 1, 2, 1, 0, 1, 3, 2, 0, 2, 2], dtype=float)

    # test
    waveform = create_waveform(times, pes, bins, wf_bin_time, nsamples)

    assert np.all(waveform == expected)
