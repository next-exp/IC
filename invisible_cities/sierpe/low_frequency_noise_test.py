import pytest

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
    frequencies       = np.arange(15000, 20000, 500) ## subset of frequences
    
    freq_cont, lowf, highf = lfn.buffer_and_limits(buffer_len       ,
                                                   buffer_sample_wid,
                                                   frequencies      )

    assert np.all(freq_cont.keywords['time_bins'] == buffer_sample_wid * np.arange(buffer_len))
    assert np.all(np.diff(lowf) == 500)
    assert np.all(np.diff(highf) == 500)
    assert lowf[0] == frequencies[0] - 250
    assert highf[-1] == frequencies[-1] + 250


def test_low_frequency_noise():

    ## Variable for basic definitions
    run_no     = 6000
    buffer_len = 32000 ## 800 mus buffer

    noise_func = lfn.low_frequency_noise(run_no, buffer_len)

    ## Get an array with all pmts
    pmt_noise = np.array(list(map(noise_func, np.arange(12))))

    assert pmt_noise.shape[1] == buffer_len

    ## Which should be the same and which not depends on
    ## the mapping whhich is a databse issue but they shouldn't
    ## all be the same only some.
    ## Some are different
    assert np.any(np.diff(pmt_noise, axis = 0))
    ## but not all
    assert not np.all(np.diff(pmt_noise, axis = 0))
    
