from __future__ import absolute_import

from pytest import mark
import sys, os

import pandas as pd
import numpy as np
import numpy.testing as npt
import tables as tb

from   invisible_cities.database import load_db
import invisible_cities.sierpe.blr as blr
import invisible_cities.core.tbl_functions as tbl
import invisible_cities.core.peak_functions_c as cpf
import invisible_cities.core.peak_functions as pf
import invisible_cities.core.sensor_functions as sf
import invisible_cities.core.core_functions as cf
from   invisible_cities.core.system_of_units_c import units

@mark.slow
def test_csum_zs_blr_cwf():
    """Test that:
     1) the calibrated sum (csum) of the BLR and the CWF is the same
    within tolerance.
     2) csum and zeros-supressed sum (zs) are the same
    within tolerance
    """

    RWF_file = (os.environ['ICDIR']
               + '/database/test_data/electrons_40keV_z250_RWF.h5')

    with tb.open_file(RWF_file, 'r') as h5rwf:
        pmtrwf, pmtblr, sipmrwf = tbl.get_vectors(h5rwf)
        DataPMT = load_db.DataPMT()
        coeff_c    = abs(DataPMT.coeff_c.values)
        coeff_blr  = abs(DataPMT.coeff_blr.values)
        adc_to_pes = abs(DataPMT.adc_to_pes.values)

        for event in range(10):
            CWF = blr.deconv_pmt(pmtrwf[event], coeff_c, coeff_blr)
            csum_cwf, _ = cpf.calibrated_pmt_sum(
                             CWF,
                             adc_to_pes,
                             n_MAU=100, thr_MAU=3)

            csum_blr, _ = cpf.calibrated_pmt_sum(
                             pmtblr[event].astype(np.float64),
                             adc_to_pes,
                             n_MAU=100, thr_MAU=3)

            wfzs_ene, wfzs_indx = cpf.wfzs(csum_cwf, threshold=0.5)

            assert np.isclose(np.sum(csum_cwf), np.sum(csum_blr), rtol=0.01)
            assert np.isclose(np.sum(csum_cwf), np.sum(wfzs_ene), rtol=0.1)


@mark.slow
def test_csum_python_cython():
    """Test that python and cython functions yield the same result for csum
    """

    RWF_file = (os.environ['ICDIR']
               + '/database/test_data/electrons_40keV_z250_RWF.h5')

    DataPMT = load_db.DataPMT()
    adc_to_pes = abs(DataPMT.adc_to_pes.values)

    with tb.open_file(RWF_file, 'r') as h5rwf:
        pmtrwf, pmtblr, sipmrwf = tbl.get_vectors(h5rwf)
        for event in range(10):
            csum_blr, _  =      cpf.calibrated_pmt_sum(
                                   pmtblr[event].astype(np.float64),
                                   adc_to_pes,
                                   n_MAU=100, thr_MAU=3)
            csum_blr_py, _, _ = pf.calibrated_pmt_sum(
                                   pmtblr[event].astype(np.float64),
                                   adc_to_pes,
                                   n_MAU=100, thr_MAU=3)

            assert abs(np.sum(csum_blr) - np.sum(csum_blr_py)) < 1e-4

            wfzs_ene,    wfzs_indx    = cpf.wfzs(csum_blr,    threshold=0.5)
            wfzs_ene_py, wfzs_indx_py =  pf.wfzs(csum_blr_py, threshold=0.5)

            assert abs(np.sum(wfzs_ene) - np.sum(wfzs_ene_py)) < 1e-4
            assert abs(np.sum(wfzs_ene) - np.sum(wfzs_ene_py)) < 1e-4
            npt.assert_array_equal(wfzs_indx, wfzs_indx_py)


def toy_pmt_signal():
    """ Mimick a PMT waveform."""
    v0 = cf.np_constant(200, 1)
    v1 = cf.np_range(1.1, 2.1, 0.1)
    v2 = cf.np_constant(10, 2)
    v3 = cf.np_reverse_range(1.1, 2.1, 0.1)

    v   = np.concatenate((v0, v1, v2, v3, v0))
    pmt = np.concatenate((v, v, v))
    return pmt


def toy_cwf_and_adc(v, npmt=10):
    """Return CWF and adc_to_pes for toy example"""
    CWF = [v] * npmt
    adc_to_pes = np.ones(v.shape[0])
    return np.array(CWF), adc_to_pes


def vsum_zsum(vsum, threshold=10):
    """Compute ZS over vsum"""
    return vsum[vsum > threshold]


def test_csum_zs_s12():
    """Several sequencial tests:
    1) Test that csum (the object of the test) and vsum (sum of toy pmt
    waveforms) yield the same result.
    2) Same for ZS sum
    3) Test that time_from_index is the same in python and cython functions.
    4) test that rebin is the same in python and cython functions.
    5) test that find_S12 is the same in python and cython functions.
    """
    v = toy_pmt_signal()
    npmt = 10
    vsum = v * npmt
    CWF, adc_to_pes = toy_cwf_and_adc(v, npmt=npmt)
    csum, _ = cpf.calibrated_pmt_sum(CWF, adc_to_pes, n_MAU=1, thr_MAU=0)
    npt.assert_allclose(vsum, csum)

    vsum_zs = vsum_zsum(vsum, threshold=10)
    wfzs_ene, wfzs_indx = cpf.wfzs(csum, threshold=10)
    npt.assert_allclose(vsum_zs, wfzs_ene)

    t1 = pf.time_from_index(wfzs_indx)
    t2 = cpf.time_from_index(wfzs_indx)
    npt.assert_allclose(t1, t2)

    t = pf.time_from_index(wfzs_indx)
    e = wfzs_ene
    t1, e1  = pf.rebin_waveform(t, e, stride=10)
    t2, e2 = cpf.rebin_waveform(t, e, stride=10)
    npt.assert_allclose(t1, t2)
    npt.assert_allclose(e1, e2)

    S12L1 = pf.find_S12(wfzs_ene, wfzs_indx,
             tmin = 0, tmax = 1e+6,
             lmin = 0, lmax = 1000000,
             stride=4, rebin=False, rebin_stride=40)

    S12L2 = cpf.find_S12(wfzs_ene, wfzs_indx,
             tmin = 0, tmax = 1e+6,
             lmin = 0, lmax = 1000000,
             stride=4, rebin=False, rebin_stride=40)

    for i in S12L1:
        t1 = S12L1[i][0]
        e1 = S12L1[i][1]
        t2 = S12L2[i][0]
        e2 = S12L2[i][1]
        npt.assert_allclose(t1, t2)
        npt.assert_allclose(e1, e2)

    # toy yields 3 idential vectors of energy
    E = np.array([ 11,  12,  13,  14,  15,  16,  17,  18,  19,  20,
                   20,  20,  20,  20,  20,  20,  20,  20,  20,  20,
                   20,  19,  18,  17,  16,  15,  14,  13,  12,  11])
    for i in S12L2.keys():
        e = S12L2[i][1]
        npt.assert_allclose(e,E)

    # rebin
    S12L2 = cpf.find_S12(wfzs_ene, wfzs_indx,
             tmin = 0, tmax = 1e+6,
             lmin = 0, lmax = 1000000,
             stride=10, rebin=True, rebin_stride=10)

    E = np.array([155,  200,  155])

    for i in S12L2.keys():
        e = S12L2[i][1]
        npt.assert_allclose(e, E)
