import numpy as np

from numpy.testing import assert_allclose

from .             import low_frequency_noise as lfn


def test_frequency_contribution():

    time_bins = 25e-9 * np.arange(80)

    freq_contribution = lfn.frequency_contribution(1, 1, 0, time_bins)

    assert_allclose(freq_contribution, 2 * np.cos(time_bins))


def test_buffer_and_limits():

    buffer_len        = 40 ## dummy 1 mus buffer
    buffer_sample_wid = 25e-9
    frequency_bin      = 500
    frequencies       = np.arange(15000, 20000, frequency_bin)

    freq_cont, lowf, highf = lfn.buffer_and_limits(buffer_len       ,
                                                   buffer_sample_wid,
                                                   frequencies      )

    assert np.all(freq_cont.keywords['time_bins'] == buffer_sample_wid * np.arange(buffer_len))
    assert np.all(np.diff(lowf) == frequency_bin)
    assert np.all(np.diff(highf) == frequency_bin)
    assert lowf[0] == frequencies[0] - frequency_bin / 2
    assert highf[-1] == frequencies[-1] + frequency_bin / 2


def test_low_frequency_noise(dbnew):

    ## Variable for basic definitions
    run_no     = 6000
    buffer_len = 32000 ## 800 mus buffer

    noise_func = lfn.low_frequency_noise(dbnew, run_no, buffer_len)

    ## Get an array with all pmts
    pmt_noise = np.array(list(map(noise_func, np.arange(12))))

    assert pmt_noise.shape[1] == buffer_len

    ## Which pmts should have the same noise and which not
    ## depends on the mapping which is a databse issue.
    ## They shouldn't all be the same though as they should be
    ## grouped by febox.
    ## First check some are different
    assert np.any(np.diff(pmt_noise, axis = 0))
    ## then that not all are
    assert not np.all(np.diff(pmt_noise, axis = 0))
