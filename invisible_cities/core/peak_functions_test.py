from __future__ import absolute_import

from pytest import mark, fixture
import sys, os
from os import path

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
from invisible_cities.core.params import S12Params, ThresholdParams
from   invisible_cities.core.system_of_units_c import units

electron_files = ['electrons_40keV_z250_RWF.h5',
                  'electrons_511keV_z250_RWF.h5',
                  'electrons_1250keV_z250_RWF.h5',
                  'electrons_2500keV_z250_RWF.h5']

@fixture(scope='module', params=electron_files)
def csum_zs_blr_cwf(request):
    """Test that:
     1) the calibrated sum (csum) of the BLR and the CWF is the same
    within tolerance.
     2) csum and zeros-supressed sum (zs) are the same
    within tolerance
    """

    RWF_file = path.join(os.environ['ICDIR'],
                         'database/test_data',
                         request.param)

    with tb.open_file(RWF_file, 'r') as h5rwf:
        pmtrwf, pmtblr, sipmrwf = tbl.get_vectors(h5rwf)
        DataPMT = load_db.DataPMT()
        coeff_c    = abs(DataPMT.coeff_c.values)
        coeff_blr  = abs(DataPMT.coeff_blr.values)
        adc_to_pes = abs(DataPMT.adc_to_pes.values)

        #for event in range(10):

        event = 0
        CWF = blr.deconv_pmt(pmtrwf[event], coeff_c, coeff_blr)
        csum_cwf, _ = cpf.calibrated_pmt_sum(CWF,
                                             adc_to_pes,
                                               n_MAU = 100,
                                             thr_MAU =   3)

        csum_blr, _ = cpf.calibrated_pmt_sum(pmtblr[event].astype(np.float64),
                               adc_to_pes,
                                 n_MAU = 100,
                               thr_MAU =   3)

        csum_blr_py, _, _ = pf.calibrated_pmt_sum(
                               pmtblr[event].astype(np.float64),
                               adc_to_pes,
                               n_MAU=100, thr_MAU=3)

        wfzs_ene,    wfzs_indx    = cpf.wfzs(csum_blr,    threshold=0.5)
        wfzs_ene_py, wfzs_indx_py =  pf.wfzs(csum_blr_py, threshold=0.5)

        # TODO: check that BLR/CWF OK
        # wfzs_ene,    wfzs_indx    = cpf.wfzs(csum_cwf,    threshold=0.5)
        # wfzs_ene_py, wfzs_indx_py =  pf.wfzs(csum_blr_py, threshold=0.5)

        from collections import namedtuple

        return (namedtuple('Csum',
                        """csum_cwf csum_blr csum_blr_py
                           wfzs_ene wfzs_ene_py
                           wfzs_indx wfzs_indx_py""")
        (csum_cwf     = csum_cwf,
         csum_blr     = csum_blr,
         wfzs_ene     = wfzs_ene,
         csum_blr_py  = csum_blr_py,
         wfzs_ene_py  = wfzs_ene_py,
         wfzs_indx    = wfzs_indx,
         wfzs_indx_py = wfzs_indx_py))


def test_csum_cwf_close_to_csum_blr(csum_zs_blr_cwf):
    p = csum_zs_blr_cwf
    assert np.isclose(np.sum(p.csum_cwf), np.sum(p.csum_blr), rtol=0.01)

def test_csum_cwf_close_to_wfzs_ene(csum_zs_blr_cwf):
    p = csum_zs_blr_cwf
    assert np.isclose(np.sum(p.csum_cwf), np.sum(p.wfzs_ene), rtol=0.1)

def test_csum_blr_close_to_csum_blr_py(csum_zs_blr_cwf):
    p = csum_zs_blr_cwf
    assert np.isclose(np.sum(p.csum_blr), np.sum(p.csum_blr_py), rtol=1e-4)

def test_wfzs_ene_close_to_wfzs_ene_py(csum_zs_blr_cwf):
    p = csum_zs_blr_cwf
    assert np.isclose(np.sum(p.wfzs_ene), np.sum(p.wfzs_ene_py), atol=1e-4)

def test_wfzs_indx_close_to_wfzs_indx_py(csum_zs_blr_cwf):
    p = csum_zs_blr_cwf
    npt.assert_array_equal(p.wfzs_indx, p.wfzs_indx_py)

@fixture(scope='module', params=electron_files)
def pmaps_electrons(request):
    """Compute PMAPS for ele of 40 keV. Check that results are consistent."""

    event = 0
    RWF_file = path.join(os.environ['ICDIR'],
                         'database/test_data',
                         request.param)

    s1par = S12Params(tmin   =  99 * units.mus,
                      tmax   = 101 * units.mus,
                      lmin   =   4,
                      lmax   =  20,
                      stride =   4,
                      rebin  = False)

    s2par = S12Params(tmin   =    101 * units.mus,
                      tmax   =   1199 * units.mus,
                      lmin   =     80,
                      lmax   = 200000,
                      stride =     40,
                      rebin  = True)

    thr = ThresholdParams(thr_s1   =  0.2 * units.pes,
                          thr_s2   =  1   * units.pes,
                          thr_MAU  =  3   * units.adc,
                          thr_sipm =  5   * units.pes,
                          thr_SIPM = 30   * units.pes )

    with tb.open_file(RWF_file,'r') as h5rwf:
        pmtrwf, pmtblr, sipmrwf = tbl.get_vectors(h5rwf)

        csum, pmp = pf.compute_csum_and_pmaps(pmtrwf,
                                              sipmrwf,
                                              s1par,
                                              s2par,
                                              thr,
                                              event)

        _, pmp2 = pf.compute_csum_and_pmaps(pmtrwf,
                                            sipmrwf,
                                            s1par,
                                            s2par._replace(rebin=False),
                                            thr,
                                            event)

    return pmp, pmp2, csum

def test_rebinning_does_not_affect_the_sum_of_S2(pmaps_electrons):
    pmp, pmp2, _ = pmaps_electrons
    np.isclose(np.sum(pmp.S2[0][1]), np.sum(pmp2.S2[0][1]), rtol=1e-05)

def test_sum_of_S2_and_sum_of_calibrated_sum_vector_must_be_close(pmaps_electrons):
    pmp, _, csum = pmaps_electrons
    np.isclose(np.sum(pmp.S2[0][1]), np.sum(csum.csum), rtol=1e-02)

def test_length_of_S1_time_array_must_match_energy_array(pmaps_electrons):
    pmp, _, _ = pmaps_electrons
    if pmp.S1:
        assert len(pmp.S1[0][0]) == len(pmp.S1[0][1])

def test_length_of_S2_time_array_must_match_energy_array(pmaps_electrons):
    pmp, _, _ = pmaps_electrons
    if pmp.S2:
        assert len(pmp.S2[0][0]) == len(pmp.S2[0][1])

def test_length_of_S2_time_array_and_length_of_S2Si_energy_array_must_be_the_same(pmaps_electrons):
    pmp, _, _ = pmaps_electrons

    if pmp.S2 and pmp.S2Si:
        for _, Es in pmp.S2Si[0]:
            assert len(pmp.S2[0][0]) == len(Es)

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

    S12L1 = pf.find_S12_py(wfzs_ene, wfzs_indx,
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
