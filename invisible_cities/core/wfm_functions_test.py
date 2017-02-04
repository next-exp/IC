from __future__ import absolute_import
import os
from os import path
from hypothesis import given, assume
from hypothesis.strategies import lists, integers, floats
from hypothesis.extra.numpy import arrays

from . import wfm_functions as wfm
import invisible_cities.core.peak_functions_c as cpf
import invisible_cities.core.tbl_functions as tbl
import numpy as np
import tables as tb


def ndarrays_of_shape(shape, lo=-1000.0, hi=1000.0):
    return arrays('float64', shape=shape,
                  elements=floats(min_value=lo, max_value=hi))


def test_rebin_wf():
    # First, consider a simple test function

    t = np.arange(1, 100, 0.1)
    e = np.exp(-t / t ** 2)

    T, E   = wfm.rebin_waveform(t, e, stride=10)
    T2, E2 = wfm.rebin_wf(t, e, stride=10)
    T3, E3 = cpf.rebin_waveform(t, e, stride=10)
    np.testing.assert_allclose(T, T2, rtol=1e-5, atol=0)
    np.testing.assert_allclose(T, T3, rtol=1e-5, atol=0)
    np.testing.assert_allclose(E, E2, rtol=1e-5, atol=0)
    np.testing.assert_allclose(E, E3, rtol=1e-5, atol=0)
    np.testing.assert_allclose(np.sum(e), np.sum(E), rtol=1e-5, atol=0)


@given(t = ndarrays_of_shape(shape=(1), lo=0.1, hi=1000.0))
def test_rebin_wf2(t):
    """ test using automatic array generation"""
    # First, consider a simple test function

    e = np.exp(-t / t ** 2)

    T, E   = wfm.rebin_waveform(t, e, stride=10)
    # np.testing.assert_allclose(T, T2, rtol=1e-5, atol=1e-5)
    # np.testing.assert_allclose(T, T3, rtol=1e-5, atol=1e-5)
    # np.testing.assert_allclose(E, E2, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.sum(e), np.sum(E), rtol=1e-5, atol=1e-5)


def test_compare_cwf_blr():
    """Test functions cwf_from_rwf() and compare_cwf_blr().
    The test:
    1) Computes CWF from RWF (function cwf_from_rwf())
    2) Computes the difference between CWF and BLR (compare_cwf_blr())
    3) Asserts that the differences are small.
    For 10 events and 12 PMTs per event, all differences are less than 0.1 %
    Input file (needed in repository): electrons_40keV_z250_RWF.h5
    """

    RWF_file = path.join(os.environ['ICDIR'],
                         'database/test_data/electrons_40keV_z250_RWF.h5')
    h5rwf = tb.open_file(RWF_file,'r')
    pmtrwf, pmtblr, sipmrwf = tbl.get_vectors(h5rwf)
    NEVT, NPMT, PMTWL = pmtrwf.shape

    CWF = wfm.cwf_from_rwf(pmtrwf, event_list=range(NEVT))
    diff = wfm.compare_cwf_blr(CWF, pmtblr,
                               event_list=range(NEVT), window_size=200)

    assert max(diff) < 0.1
