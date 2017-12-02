"""
code: peak_functions_test.py
description: tests for peak functions.

credits: see ic_authors_and_legal.rst in /doc

last revised: JJGC, 10-July-2017
"""

from collections import namedtuple

import tables        as tb
import numpy         as np
import numpy.testing as npt

from pytest import fixture

from .. core                   import core_functions   as cf
from .. core.system_of_units_c import units

from .. database               import load_db

from .. sierpe                 import blr

from .                         import peak_functions   as pf
from .                         import calib_sensors_functions_c as cpf
from .                         import peak_functions_c as pfc
from .                         import tbl_functions    as tbl
from .. evm.ic_containers      import S12Params
from .. evm.ic_containers      import ThresholdParams
from .. evm.ic_containers      import DeconvParams
from .. evm.ic_containers      import CalibVectors
from .. types.ic_types         import minmax


# TODO: rethink this test (list(6) could stop working anytime if DataPMT is changed)
@fixture(scope='module')
def csum_zs_blr_cwf(electron_RWF_file):
    """Test that:
     1) the calibrated sum (csum) of the BLR and the CWF is the same
    within tolerance.
     2) csum and zeros-supressed sum (zs) are the same
    within tolerance
    """

    run_number = 0

    with tb.open_file(electron_RWF_file, 'r') as h5rwf:
        pmtrwf, pmtblr, sipmrwf = tbl.get_vectors(h5rwf)
        DataPMT = load_db.DataPMT(run_number)
        pmt_active = np.nonzero(DataPMT.Active.values)[0].tolist()
        coeff_c    = abs(DataPMT.coeff_c.values)
        coeff_blr  = abs(DataPMT.coeff_blr.values)
        adc_to_pes = abs(DataPMT.adc_to_pes.values)

        event = 0
        CWF  = blr.deconv_pmt(pmtrwf[event], coeff_c, coeff_blr, pmt_active)
        CWF6 = blr.deconv_pmt(pmtrwf[event], coeff_c, coeff_blr, list(range(6)))
        csum_cwf, _ =      cpf.calibrated_pmt_sum(
                               CWF,
                               adc_to_pes,
                               pmt_active = pmt_active,
                               n_MAU = 100,
                               thr_MAU =   3)

        csum_blr, _ =      cpf.calibrated_pmt_sum(
                               pmtblr[event].astype(np.float64),
                               adc_to_pes,
                               pmt_active = pmt_active,
                               n_MAU = 100,
                               thr_MAU =   3)

        csum_blr_py, _, _ = pf._calibrated_pmt_sum(
                               pmtblr[event].astype(np.float64),
                               adc_to_pes,
                               pmt_active = pmt_active,
                               n_MAU=100, thr_MAU=3)

        csum_cwf_pmt6, _ = cpf.calibrated_pmt_sum(
                               CWF,
                               adc_to_pes,
                               pmt_active = list(range(6)),
                               n_MAU = 100,
                               thr_MAU =   3)

        csum_blr_pmt6, _ = cpf.calibrated_pmt_sum(
                               pmtblr[event].astype(np.float64),
                               adc_to_pes,
                               pmt_active = list(range(6)),
                               n_MAU = 100,
                               thr_MAU =   3)

        csum_blr_py_pmt6, _, _ = pf._calibrated_pmt_sum(
                                    pmtblr[event].astype(np.float64),
                                    adc_to_pes,
                                    pmt_active = list(range(6)),
                                    n_MAU=100, thr_MAU=3)

        CAL_PMT, CAL_PMT_MAU  =  cpf.calibrated_pmt_mau(
                                     CWF,
                                     adc_to_pes,
                                     pmt_active = pmt_active,
                                     n_MAU = 100,
                                     thr_MAU =   3)


        wfzs_ene,    wfzs_indx    = pfc.wfzs(csum_blr,    threshold=0.5)
        wfzs_ene_py, wfzs_indx_py =  pf._wfzs(csum_blr_py, threshold=0.5)

        return (namedtuple('Csum',
                        """cwf cwf6
                           csum_cwf csum_blr csum_blr_py
                           csum_cwf_pmt6 csum_blr_pmt6 csum_blr_py_pmt6
                           CAL_PMT, CAL_PMT_MAU,
                           wfzs_ene wfzs_ene_py
                           wfzs_indx wfzs_indx_py""")
        (cwf               = CWF,
         cwf6              = CWF6,
         csum_cwf          = csum_cwf,
         csum_blr          = csum_blr,
         csum_blr_py       = csum_blr_py,
         csum_cwf_pmt6     = csum_cwf_pmt6,
         csum_blr_pmt6     = csum_blr_pmt6,
         CAL_PMT           = CAL_PMT,
         CAL_PMT_MAU       = CAL_PMT_MAU,
         csum_blr_py_pmt6  = csum_blr_py_pmt6,
         wfzs_ene          = wfzs_ene,
         wfzs_ene_py       = wfzs_ene_py,
         wfzs_indx         = wfzs_indx,
         wfzs_indx_py      = wfzs_indx_py))


def test_csum_cwf_close_to_csum_of_calibrated_pmts(csum_zs_blr_cwf):
    p = csum_zs_blr_cwf

    csum = 0
    for pmt in p.CAL_PMT:
        csum += np.sum(pmt)

    assert np.isclose(np.sum(p.csum_cwf), np.sum(csum), rtol=0.0001)


def test_csum_cwf_close_to_csum_blr(csum_zs_blr_cwf):
    p = csum_zs_blr_cwf
    assert np.isclose(np.sum(p.csum_cwf), np.sum(p.csum_blr), rtol=0.01)


def test_csum_cwf_pmt_close_to_csum_blr_pmt(csum_zs_blr_cwf):
    p = csum_zs_blr_cwf
    assert np.isclose(np.sum(p.csum_cwf_pmt6), np.sum(p.csum_blr_pmt6),
                      rtol=0.01)


def test_csum_cwf_close_to_wfzs_ene(csum_zs_blr_cwf):
    p = csum_zs_blr_cwf
    assert np.isclose(np.sum(p.csum_cwf), np.sum(p.wfzs_ene), rtol=0.1)


def test_csum_blr_close_to_csum_blr_py(csum_zs_blr_cwf):
    p = csum_zs_blr_cwf
    assert np.isclose(np.sum(p.csum_blr), np.sum(p.csum_blr_py), rtol=1e-4)


def test_csum_blr_pmt_close_to_csum_blr_py_pmt(csum_zs_blr_cwf):
    p = csum_zs_blr_cwf
    assert np.isclose(np.sum(p.csum_blr_pmt6), np.sum(p.csum_blr_py_pmt6),
                      rtol=1e-3)


def test_wfzs_ene_close_to_wfzs_ene_py(csum_zs_blr_cwf):
    p = csum_zs_blr_cwf
    assert np.isclose(np.sum(p.wfzs_ene), np.sum(p.wfzs_ene_py), atol=1e-4)


def test_wfzs_indx_close_to_wfzs_indx_py(csum_zs_blr_cwf):
    p = csum_zs_blr_cwf
    npt.assert_array_equal(p.wfzs_indx, p.wfzs_indx_py)


def test_cwf_are_empty_for_masked_pmts(csum_zs_blr_cwf):
    assert np.all(csum_zs_blr_cwf.cwf6[6:] == 0.)


@fixture(scope="session")
def toy_sipm_signal():
    NSIPM = 1792
    WL    = 100

    common_threshold      = np.random.uniform(0.3, 0.7)
    individual_thresholds = np.random.uniform(0.3, 0.7, size=NSIPM)

    adc_to_pes  = np.full(NSIPM, 100, dtype=np.double)
    signal_adc  = np.random.randint(0, 100, size=(NSIPM, WL), dtype=np.int16)

    # subtract baseline and convert to pes
    signal_pes  = signal_adc - np.mean(signal_adc, axis=1)[:, np.newaxis]
    signal_pes /= adc_to_pes[:, np.newaxis]

    signal_zs_common_threshold = np.array(signal_pes)
    signal_zs_common_threshold[signal_pes < common_threshold] = 0

    # thresholds must be reshaped to allow broadcasting
    individual_thresholds_reshaped = individual_thresholds[:, np.newaxis]

    signal_zs_individual_thresholds = np.array(signal_pes)
    signal_zs_individual_thresholds[signal_pes < individual_thresholds_reshaped] = 0

    return (signal_adc, adc_to_pes,
            signal_zs_common_threshold,
            signal_zs_individual_thresholds,
            common_threshold,
            individual_thresholds)


def test_signal_sipm_common_threshold(toy_sipm_signal):
    (signal_adc, adc_to_pes,
     signal_zs_common_threshold, _,
     common_threshold, _) = toy_sipm_signal

    zs_wf = cpf.signal_sipm(signal_adc, adc_to_pes, common_threshold)
    assert np.allclose(zs_wf, signal_zs_common_threshold)


def test_signal_sipm_individual_thresholds(toy_sipm_signal):
    (signal_adc, adc_to_pes,
     _, signal_zs_individual_thresholds,
     _, individual_thresholds) = toy_sipm_signal

    zs_wf = cpf.signal_sipm(signal_adc, adc_to_pes, individual_thresholds)
    assert np.allclose(zs_wf, signal_zs_individual_thresholds)
