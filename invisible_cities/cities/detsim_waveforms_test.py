import pytest
from hypothesis import given
from hypothesis.strategies import composite
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis.strategies import lists

import numpy as np

from invisible_cities.cities.detsim_waveforms import create_waveform


def test_create_waveform():
    #data
    bin_width = 1.
    bins = np.arange(0., 10. + bin_width, bin_width)
    nsamples = 2
    times = np.array([1, 2, 5, 6, 9, 10])
    pes   = np.array([2, 2, 2, 4, 1,  3])

    expected = np.array([0, 1, 2, 1, 0, 1, 3, 2, 0, 2, 2], dtype=float)

    # test
    waveform = create_waveform(times, pes, bins, nsamples)

    assert np.allclose(waveform, expected)


@composite
def times_and_pes_from_bins(draw, bins):
    min, max = bins[0], bins[-1]

    times = draw(lists(integers(min_value=min, max_value=max), min_size=1))
    pes   = draw(lists(integers(min_value=0), min_size=len(times), max_size=len(times)))
    return times, pes, bins

@given(times_and_pes_from_bins(bins=np.arange(0, 10)),
       integers())
def test_create_waveform_Exception_Warning_Sum(times_and_pes, nsamples):
    times, pes, bins = times_and_pes

    if (nsamples<1) or (nsamples>len(bins)):
        pytest.raises(ValueError,  create_waveform, times, pes, bins, nsamples)
    else:
        waveform = create_waveform(times, pes, bins, nsamples)
        assert np.allclose(np.sum(pes), np.sum(waveform))
