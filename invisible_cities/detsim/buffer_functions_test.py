import numpy  as np
import pandas as pd

from pytest import mark

from .. core            import system_of_units as units

from . buffer_functions import       bin_sensors
from . buffer_functions import buffer_calculator
from . buffer_functions import find_signal_start


def test_bin_sensors(mc_waveforms, pmt_ids, sipm_ids):
    max_buffer = 10 * units.minute

    evts, pmt_binwid, sipm_binwid, all_wfs = mc_waveforms

    evt   = evts[0]
    wfs   = all_wfs.loc[evt]

    pmts  = wfs[wfs.index.isin( pmt_ids)]
    sipms = wfs[wfs.index.isin(sipm_ids)]

    ## Assumes pmts the triggering sensors as in new/next-100
    bins_min = pmts.time.min()
    bins_max = pmts.time.max() + pmt_binwid
    pmt_bins ,  pmt_wf = bin_sensors(pmts ,  pmt_binwid,
                                     bins_min, bins_max, max_buffer)
    sipm_bins, sipm_wf = bin_sensors(sipms, sipm_binwid,
                                     bins_min, bins_max, max_buffer)

    assert  pmt_bins[ 0] >= bins_min
    assert  pmt_bins[-1] <= min(bins_max, max_buffer)
    assert np.all(np.diff( pmt_bins) ==  pmt_binwid)

    assert sipm_bins[ 0] >= bins_min
    assert sipm_bins[-1] <= min(bins_max, max_buffer)
    assert np.all(np.diff(sipm_bins) == sipm_binwid)

    ## In current DAQ, pmts have higher sample frequency than SiPMs
    assert pmt_bins[ 0] >= sipm_bins[ 0]
    assert pmt_bins[-1] >= sipm_bins[-1]

    pmt_sum  = pmts .charge.sum()
    sipm_sum = sipms.charge.sum()
    assert pmt_wf .sum().sum() ==  pmt_sum
    assert sipm_wf.sum().sum() == sipm_sum


@mark.parametrize("signal_thresh", (2, 10))
def test_find_signal_start(binned_waveforms, signal_thresh):

    pmt_bins, pmt_wfs, *_ = binned_waveforms

    buffer_length = 800 * units.mus
    bin_width     = np.diff(pmt_bins)[0]
    stand_off     = int(buffer_length / bin_width)

    pmt_sum       = pmt_wfs.sum()
    pulses        = find_signal_start(pmt_wfs, signal_thresh, stand_off)

    assert np.all(pmt_sum[pulses] > signal_thresh)


@mark.parametrize("signal_thresh", (2, 10))
def test_find_signal_start_numpy(binned_waveforms, signal_thresh):

    pmt_bins, pmt_wfs, *_ = binned_waveforms

    buffer_length = 800 * units.mus
    bin_width     = np.diff(pmt_bins)[0]
    stand_off     = int(buffer_length / bin_width)

    pmt_sum       = pmt_wfs.sum()
    pmt_wfs_np    = np.asarray(pmt_wfs.tolist())
    pulses        = find_signal_start(pmt_wfs_np, signal_thresh, stand_off)

    assert np.all(pmt_sum[pulses] > signal_thresh)


def test_find_signal_start_correct_index():

    thresh     = 5
    thr_bin    = 3
    simple_evt = pd.Series([np.zeros(5)] * 12)
    simple_evt[1][thr_bin] = thresh

    pulses     = find_signal_start(simple_evt, thresh, 5)
    assert len(pulses) == 1
    assert pulses[0]   == thr_bin


@mark.parametrize("pre_trigger signal_thresh".split(),
                  ((100 * units.mus,  2),
                   (400 * units.mus, 10)))
def test_buffer_calculator(mc_waveforms, binned_waveforms,
                           pre_trigger ,    signal_thresh):

    _, pmt_binwid, sipm_binwid, _ = mc_waveforms

    pmt_bins, pmt_wfs, sipm_bins, sipm_wfs = binned_waveforms

    buffer_length     = 800 * units.mus
    bin_width         = np.diff(pmt_bins)[0]
    stand_off         = int(buffer_length / bin_width)

    pulses            = find_signal_start(pmt_wfs, signal_thresh, stand_off)

    calculate_buffers = buffer_calculator(buffer_length,
                                          pre_trigger  ,
                                          pmt_binwid   ,
                                          sipm_binwid  )

    buffers           = calculate_buffers(pulses, *binned_waveforms)
    pmt_sum           = pmt_wfs.sum()

    assert len(buffers) == len(pulses)
    for i, (evt_pmt, evt_sipm) in enumerate(buffers):
        sipm_trg_bin = np.where(sipm_bins <= pmt_bins[pulses[i]])[0][-1]
        diff_binedge = pmt_bins[pulses[i]] - sipm_bins[sipm_trg_bin]
        pre_trg_samp = int(pre_trigger / pmt_binwid + diff_binedge)

        assert pmt_wfs .shape[0] == evt_pmt .shape[0]
        assert sipm_wfs.shape[0] == evt_sipm.shape[0]
        assert evt_pmt .shape[1] == int(buffer_length /  pmt_binwid)
        assert evt_sipm.shape[1] == int(buffer_length / sipm_binwid)
        assert np.sum(evt_pmt, axis=0)[pre_trg_samp] == pmt_sum[pulses[i]]


def test_buffer_calculator_pandas_numpy(mc_waveforms, binned_waveforms):

    _, pmt_binwid, sipm_binwid, _ = mc_waveforms

    pmt_bins, pmt_wfs, sipm_bins, sipm_wfs = binned_waveforms

    buffer_length     = 800 * units.mus
    bin_width         = np.diff(pmt_bins)[0]
    stand_off         = int(buffer_length / bin_width)
    pre_trigger       = 100 * units.mus
    signal_thresh     = 2

    pulses_pd         = find_signal_start(pmt_wfs, signal_thresh, stand_off)
    pmt_nparr         = np.asarray(pmt_wfs.tolist())
    pulses_np         = find_signal_start(pmt_nparr, signal_thresh, stand_off)

    calculate_buffers = buffer_calculator(buffer_length,
                                          pre_trigger  ,
                                          pmt_binwid   ,
                                          sipm_binwid  )

    buffers_pd        = calculate_buffers(pulses_pd, *binned_waveforms)
    buffers_np        = calculate_buffers(pulses_np                    ,
                                          pmt_bins                     ,
                                          pmt_nparr                    ,
                                          sipm_bins                    ,
                                          np.asarray(sipm_wfs.tolist()))

    assert len(buffers_pd) == len(buffers_np) == 1
    evtpd_buffers = buffers_pd[0]
    evtnp_buffers = buffers_np[0]
    assert np.all([np.all(evtpd_buffers[0] == evtnp_buffers[0]),
                   np.all(evtpd_buffers[1] == evtnp_buffers[1])])
