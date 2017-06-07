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
from .                         import peak_functions_c as cpf
from .                         import tbl_functions    as tbl
from . params                  import S12Params
from . params                  import ThresholdParams
from . params                  import DeconvParams
from . params                  import CalibVectors
from . params                  import minmax


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

        csum_blr_py, _, _ = pf.calibrated_pmt_sum(
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

        csum_blr_py_pmt6, _, _ = pf.calibrated_pmt_sum(
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


        wfzs_ene,    wfzs_indx    = cpf.wfzs(csum_blr,    threshold=0.5)
        wfzs_ene_py, wfzs_indx_py =  pf.wfzs(csum_blr_py, threshold=0.5)

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


@fixture(scope="module")
def toy_S1_wf():
    s1      = {}
    indices = [np.arange(125, 130), np.arange(412, 417), np.arange(113, 115)]
    for i, index in enumerate(indices):
        s1[i] = index * 25., np.random.rand(index.size)

    wf = np.random.rand(1000)
    return s1, wf, indices


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

@fixture(scope='module')
def pmaps_electrons(electron_RWF_file):
    """Compute PMAPS for ele of 40 keV. Check that results are consistent."""

    event = 0
    run_number = 0

    s1par = S12Params(time = minmax(min   =  99 * units.mus,
                                    max   = 101 * units.mus),
                      length = minmax(min =   4,
                                      max =  20),
                      stride              =   4,
                      rebin               = False)

    s2par = S12Params(time = minmax(min   =    101 * units.mus,
                                    max   =   1199 * units.mus),
                      length = minmax(min =     80,
                                      max = 200000),
                      stride              =     40,
                      rebin               = True)

    thr = ThresholdParams(thr_s1   =  0.2 * units.pes,
                          thr_s2   =  1   * units.pes,
                          thr_MAU  =  3   * units.adc,
                          thr_sipm =  5   * units.pes,
                          thr_SIPM = 30   * units.pes )


    with tb.open_file(electron_RWF_file,'r') as h5rwf:
        pmtrwf, pmtblr, sipmrwf = tbl.get_vectors(h5rwf)
        DataPMT = load_db.DataPMT(run_number)
        DataSiPM = load_db.DataSiPM(run_number)

        calib = CalibVectors(channel_id = DataPMT.ChannelID.values,
                             coeff_blr = abs(DataPMT.coeff_blr   .values),
                             coeff_c = abs(DataPMT.coeff_c   .values),
                             adc_to_pes = DataPMT.adc_to_pes.values,
                             adc_to_pes_sipm = DataSiPM.adc_to_pes.values,
                             pmt_active = np.nonzero(DataPMT.Active.values)[0].tolist())

        deconv = DeconvParams(n_baseline = 28000,
                              thr_trigger = 5)

        csum, pmp = pf.compute_csum_and_pmaps(pmtrwf,
                                              sipmrwf,
                                              s1par,
                                              s2par,
                                              thr,
                                              event,
                                              calib,
                                              deconv)

        _, pmp2 = pf.compute_csum_and_pmaps(pmtrwf,
                                            sipmrwf,
                                            s1par,
                                            s2par._replace(rebin=False),
                                            thr,
                                            event,
                                            calib,
                                            deconv)

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
        for nsipm in pmp.S2Si[0]:
            assert len(pmp.S2Si[0][nsipm]) == len(pmp.S2[0][0])

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

    t1 =  pf.time_from_index(wfzs_indx)
    t2 = cpf.time_from_index(wfzs_indx)
    npt.assert_allclose(t1, t2)

    t = pf.time_from_index(wfzs_indx)
    e = wfzs_ene
    t1, e1  = pf.rebin_waveform(t, e, stride=10)
    t2, e2 = cpf.rebin_waveform(t, e, stride=10)
    npt.assert_allclose(t1, t2)
    npt.assert_allclose(e1, e2)

    S12L1 = pf.find_S12_py(wfzs_ene, wfzs_indx,
             time   = minmax(0, 1e+6),
             length = minmax(0, 1000000),
             stride=4, rebin=False, rebin_stride=40)

    S12L2 = cpf.find_S12(wfzs_ene, wfzs_indx,
             time   = minmax(0, 1e+6),
             length = minmax(0, 1000000),
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
             time   = minmax(0, 1e+6),
             length = minmax(0, 1000000),
             stride=10, rebin=True, rebin_stride=10)

    E = np.array([155,  200,  155])

    for i in S12L2.keys():
        e = S12L2[i][1]
        npt.assert_allclose(e, E)


def test_cwf_are_empty_for_masked_pmts(csum_zs_blr_cwf):
    assert np.all(csum_zs_blr_cwf.cwf6[6:] == 0.)


def test_correct_S1_ene_returns_correct_energies(toy_S1_wf):
    S1, wf, indices = toy_S1_wf
    corrS1 = cpf.correct_S1_ene(S1, wf)
    for peak_no, (t, E) in corrS1.items():
        assert np.all(E == wf[indices[peak_no]])
