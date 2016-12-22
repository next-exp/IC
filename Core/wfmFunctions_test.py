
from hypothesis import given, assume
from hypothesis.strategies import lists, integers, floats
from hypothesis.extra.numpy import arrays

import wfmFunctions as wfm
import ICython.Core.peakFunctions as cpf
import numpy as np


def ndarrays_of_shape(shape, lo=-1000.0, hi=1000.0):
    return arrays('float64', shape=shape, elements=floats(min_value=lo,
                   max_value=hi))


def test_rebin_wf():
    # First, consider a simple test function

    t = np.arange(1.,100., 0.1)
    e = np.exp(-t/t**2)

    T, E   = wfm.rebin_waveform(t, e, stride=10)
    T2, E2 = wfm.rebin_wf(t, e, stride=10)
    T3, E3 = cpf.rebin_waveform(t, e, stride=10)
    np.testing.assert_allclose(T, T2, rtol=1e-5, atol=0)
    np.testing.assert_allclose(T, T3, rtol=1e-5, atol=0)
    np.testing.assert_allclose(E, E2, rtol=1e-5, atol=0)
    np.testing.assert_allclose(E, E3, rtol=1e-5, atol=0)
    np.testing.assert_allclose(np.sum(e), np.sum(E), rtol=1e-5, atol=0)


@given(t = ndarrays_of_shape(shape=(1), lo=-1000.0, hi=1000.0))
def test_rebin_wf2(t):
    # First, consider a simple test function

    e = np.exp(-t/t**2)

    T, E   = wfm.rebin_waveform(t, e, stride=10)
    # np.testing.assert_allclose(T, T2, rtol=1e-5, atol=1e-5)
    # np.testing.assert_allclose(T, T3, rtol=1e-5, atol=1e-5)
    # np.testing.assert_allclose(E, E2, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.sum(e), np.sum(E), rtol=1e-5, atol=1e-5)
