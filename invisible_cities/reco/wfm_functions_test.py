import os
from   os import path

import numpy  as np
import tables as tb

from pytest import mark

from .. database import load_db

from .. core  import tbl_functions    as tbl
from .        import wfm_functions    as wfm
from .. evm.ic_containers import CalibVectors
from .. evm.ic_containers import DeconvParams


@mark.slow
def test_compare_cwf_blr(dbnew, ICDATADIR):
    """Test functions cwf_from_rwf() and compare_cwf_blr().
    The test:
    1) Computes CWF from RWF (function cwf_from_rwf())
    2) Computes the difference between CWF and BLR (compare_cwf_blr())
    3) Asserts that the differences are small.
    For 10 events and 12 PMTs per event, all differences are less than 0.1 %
    Input file (needed in repository): electrons_40keV_z25_RWF.h5
    """

    deconv = DeconvParams(n_baseline  = 45000,
                          thr_trigger =     5)

    run_number = 0
    DataPMT    = load_db.DataPMT (dbnew, run_number)
    DataSiPM   = load_db.DataSiPM(dbnew, run_number)

    calib = CalibVectors(channel_id      =     DataPMT .ChannelID .values ,
                         coeff_blr       = abs(DataPMT .coeff_blr .values),
                         coeff_c         = abs(DataPMT .coeff_c   .values),
                         adc_to_pes      =     DataPMT .adc_to_pes.values ,
                         adc_to_pes_sipm =     DataSiPM.adc_to_pes.values ,
                         pmt_active      = np.nonzero(DataPMT.Active.values)[0])

    RWF_file = path.join(ICDATADIR, 'electrons_40keV_z25_RWF.h5')
    with tb.open_file(RWF_file) as h5rwf:
        pmtrwf, pmtblr, sipmrwf = tbl.get_vectors(h5rwf)
        NEVT, NPMT, PMTWL = pmtrwf.shape

        CWF = wfm.cwf_from_rwf(pmtrwf, range(NEVT), calib, deconv)
        diff = wfm.compare_cwf_blr(CWF, pmtblr,
                                   event_list=range(NEVT), window_size=300)

    assert max(diff) < 0.15


def test_noise_suppression_constant_threshold():
    wfms       = np.array([ np.arange(0, 5)
                          , np.arange(1, 6)
                          , np.arange(2, 7) ])
    threshold  = 3.5
    suppressed = wfm.noise_suppression(wfms, threshold)
    expected   = np.where(wfms<threshold, 0, wfms)

    assert np.all(suppressed == expected)


def test_noise_suppression_different_thresholds():
    wfms       = np.array([ np.arange(0, 5)
                          , np.arange(1, 6)
                          , np.arange(2, 7) ])
    thresholds = [3.5, 4.5, 5.5]
    suppressed = wfm.noise_suppression(wfms, thresholds)
    expected   = wfms.copy()
    expected[:, :-1] = 0

    assert np.all(suppressed == expected)
