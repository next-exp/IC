import os
from   os import path

import numpy  as np
import tables as tb

from pytest import mark
from pytest import fixture

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

@fixture
def sipm_wfs_for_sipm_selection_testing():
    """
    2D array of SiPM waveforms. All but two SiPMs have 100 PEs integrated charge.
    Two outliers:
    - SiPM 2: 150 PEs, more than 3 sigma above the median
    - SiPM 7: 120 PEs, between 1-2 sigma above the median
    """
    n_sipms     = 10
    n_time_bins = 100

    wfs = np.ones((n_sipms, n_time_bins), dtype=np.float32)

    wfs[2, :] = 1.5
    wfs[7, :] = 1.2
    # median ~ 100, std ~ 15.5
    return wfs, [2, 7], [2], [0, 1, 3, 4, 5, 6, 8, 9]


def test_zero_wfs_below_threshold(sipm_wfs_for_sipm_selection_testing):
    """
    Test function zero_wfs_below_threshold(). The test asserts that the function correctly
    sets to zero the entries of the waveforms that are below a specified threshold.
    """
    wfs, passing_wfs_11_ids, _, zeroed_wfs_11_ids  = sipm_wfs_for_sipm_selection_testing

    zeroed_wfs_11 = wfm.zero_wfs_below_threshold(wfs, zeroing_thr=1.1)

    assert np.all(zeroed_wfs_11[zeroed_wfs_11_ids] == 0)
    assert np.all(zeroed_wfs_11[passing_wfs_11_ids] == wfs[passing_wfs_11_ids])


def test_median_std_method(sipm_wfs_for_sipm_selection_testing):
    """
    Test function median_std_method(). The test asserts that the function correctly 
    identifies the outliers based on different standard deviation thresholds. 
    """
    wfs, expected_outliers_1sigma, expected_outliers_3sigma, _ = sipm_wfs_for_sipm_selection_testing

    passing_sipms_1sigma = np.where(wfm.median_std_method(wfs, nsigma=1))[0].tolist()
    passing_sipms_3sigma = np.where(wfm.median_std_method(wfs, nsigma=3))[0].tolist()

    assert passing_sipms_1sigma == expected_outliers_1sigma
    assert passing_sipms_3sigma == expected_outliers_3sigma


def test_threshold_method(sipm_wfs_for_sipm_selection_testing):
    """
    Test function threshold_method(). The test asserts that the function correctly
    kills the SiPMs below a certain charge threshold. 
    """
    wfs, expected_outliers_110pes, expected_outliers_130pes, _ = sipm_wfs_for_sipm_selection_testing

    passing_sipms_110pes, _ = wfm.charge_threshold_method(wfs, zeroing_thr=0, integration_thr=110)
    passing_sipms_130pes, _ = wfm.charge_threshold_method(wfs, zeroing_thr=0, integration_thr=130)

    assert passing_sipms_110pes.tolist() == expected_outliers_110pes
    assert passing_sipms_130pes.tolist() == expected_outliers_130pes


def test_top_n_method(sipm_wfs_for_sipm_selection_testing):
    """
    Test function top_n_method(). The test asserts that the function  selects the correct
    number of SiPMs andcorrectly identifies the top N SiPMs based on their integrated charge. 
    """
    wfs, expected_outliers_top2, expected_outliers_top1, _ = sipm_wfs_for_sipm_selection_testing

    passing_sipms_top2 = np.where(wfm.top_n_method(wfs, n=2))[0].tolist()
    passing_sipms_top1 = np.where(wfm.top_n_method(wfs, n=1))[0].tolist()

    assert len(passing_sipms_top2) == 2
    assert len(passing_sipms_top1) == 1

    assert passing_sipms_top2 == expected_outliers_top2
    assert passing_sipms_top1 == expected_outliers_top1


@fixture
def sipm_grid_for_isolation_and_padding_testing():
    """
    5x5 grid of with 10mm spacing containing 4 SiPMs:
    - SiPM  7 (x=20, y=10) has nearest neighbours 12 and 13
    - SiPM 12 (x=20, y=20) has nearest neighbours 7 and 13
    - SiPM 13 (x=30, y=20) has nearestneighbours 12 and 7
    - SiPM 24 (x=40, y=40) has no nearest neighbours
    With proximity_threshold=15mm SiPMs 7, 12, 13 survive and SiPM 24 is killed.
    With proximity_threshold=5mm, all SiPMs are killed.

    With a padding of 12mm around the cluster of SiPMs 7, 12, 13 you also
    include SiPMs 2, 6, 8, 11, 14, 17, 18.
    """
    spacing = 10
    xs = np.arange(5) * spacing  # [0, 10, 20, 30, 40]
    ys = np.arange(5) * spacing

    grid_x, grid_y = np.meshgrid(xs, ys)
    sipm_x = grid_x.flatten().astype(np.float32)  # shape (25,)
    sipm_y = grid_y.flatten().astype(np.float32)

    # visual representation of the grid with SiPMs plotted as their IDs:
    # X  X  X  X  24
    # X  X  X  X  X
    # X  X  12 13 X
    # X  X  7  X  X
    # X  X  X  X  X
    selected_ids = np.zeros(25, dtype=bool)
    cluster_ids  = [7, 12, 13]
    isolated_id  = [24]
    for idx in cluster_ids + isolated_id:
        selected_ids[idx] = True

    # diagonal SiPM distance ~ 14mm
    # only SiPMs with a nearest neighbour pass assuming proximity_threshold=15mm
    # SiPM 24 should be killed
    expected_survivors_15mm = np.zeros(25, dtype=bool)
    for idx in cluster_ids:
        expected_survivors_15mm[idx] = True

    # given a SiPM distance of 10mm, there should be no survivors with proximity_threshold=5mm
    expected_survivors_5mm = np.zeros(25, dtype=bool)

    # adding a padding of 12mm around the cluster of SiPMs 7, 12, 13 would include 
    # SiPMs 2, 6, 8, 11, 14, 17, 18
    # X  X  X  X  X
    # X  X  17 18 X
    # X  11 12 13 14
    # X  6  7  8  X
    # X  X  2  X  X
    padded_cluster_ids = cluster_ids + [2, 6, 8, 11, 14, 17, 18]
    expected_survivors_padding12 = np.zeros(25, dtype=bool)
    for idx in padded_cluster_ids:
        expected_survivors_padding12[idx] = True

    # adding a padding of 0mm around the cluster of SiPMs 7, 12, 13 would include only the cluster itself
    expected_survivors_padding0 = np.zeros(25, dtype=bool)
    for idx in cluster_ids:
        expected_survivors_padding0[idx] = True

    return (sipm_x, sipm_y, selected_ids, expected_survivors_15mm, expected_survivors_5mm, 
            expected_survivors_padding12, expected_survivors_padding0)


def test_kill_isolated_sipms(sipm_grid_for_isolation_and_padding_testing):
    """"
    Test function kill_isolated_sipms(). The test asserts that the function correctly
    identifies and removes isolated SiPMs based on their proximity to other SiPMs.
    """
    sipm_x, sipm_y, selected_ids, expected_survivors_15mm, expected_survivors_5mm, _, _ = sipm_grid_for_isolation_and_padding_testing
    
    surviving_sipms_15mm = wfm.kill_isolated_sipms(selected_ids, sipm_x, sipm_y, proximity_threshold=15.0)
    surviving_sipms_5mm = wfm.kill_isolated_sipms(selected_ids, sipm_x, sipm_y, proximity_threshold=5.0)

    assert surviving_sipms_15mm.tolist() == expected_survivors_15mm.tolist()
    assert surviving_sipms_5mm.tolist() == expected_survivors_5mm.tolist()


def test_apply_circular_padding(sipm_grid_for_isolation_and_padding_testing):
    """
    Test function apply_circular_padding(). The test asserts that the function correctly applies 
    a circular padding around selected SiPMs to include neighboring SiPMs within the specified radius.
    """
    sipm_x, sipm_y, _, selected_ids, _, expected_survivors_padding12, expected_survivors_padding0 = sipm_grid_for_isolation_and_padding_testing

    padded_sipms_12mm = wfm.apply_circular_padding(selected_ids, sipm_x, sipm_y, padding_radius=12.0)
    padded_sipms_0mm = wfm.apply_circular_padding(selected_ids, sipm_x, sipm_y, padding_radius=0.0)

    assert padded_sipms_12mm.tolist() == expected_survivors_padding12.tolist()
    assert padded_sipms_0mm.tolist() == expected_survivors_padding0.tolist()