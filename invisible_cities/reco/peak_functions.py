"""
code: peak_functions.py

description: functions related to the pmap creation.

credits: see ic_authors_and_legal.rst in /doc

last revised: @abotas & @gonzaponte. Dec 1st 2017
"""

import numpy        as np

from .. core.system_of_units_c import units
from .. evm .new_pmaps         import S1
from .. evm .new_pmaps         import S2
from .. evm .new_pmaps         import PMap
from .. evm .new_pmaps         import PMTResponses
from .. evm .new_pmaps         import SiPMResponses


def indices_and_wf_above_threshold(wf, thr):
    indices_above_thr = np.where(wf > thr)[0]
    wf_above_thr      = wf[indices_above_thr]
    return indices_above_thr, wf_above_thr


def select_wfs_above_time_integrated_thr(wfs, thr):
    selected_ids = np.where(np.sum(wfs, axis=1) >= thr)[0]
    selected_wfs = wfs[selected_ids]
    return selected_ids, selected_wfs


def split_in_peaks(indices, stride):
    where = np.where(np.diff(indices) > stride)[0]
    return np.split(indices, where + 1)


def select_peaks(peaks, time, length):
    def is_valid(indices):
        return (time  .contains(indices[ 0] * 25 * units.ns) and
                time  .contains(indices[-1] * 25 * units.ns) and
                length.contains(indices[-1] + 1 - indices[0]))
    return tuple(filter(is_valid, peaks))


def pick_slice_and_rebin(indices, times, wfs, rebin_stride, pad_zeros=False):
    slice_ = slice(indices[0], indices[-1] + 1)
    times_ = times[   slice_]
    wfs_   = wfs  [:, slice_]
    if pad_zeros:
        n_miss = indices[0] % 40
        n_wfs  = wfs.shape[0]
        times_ = np.concatenate([np.zeros(        n_miss) , times_])
        wfs_   = np.concatenate([np.zeros((n_wfs, n_miss)),   wfs_], axis=1)
    times, wfs = rebin_times_and_waveforms(times_, wfs_, rebin_stride)
    return times, wfs


def build_pmt_responses(indices, times, ccwf, pmt_ids, rebin_stride, pad_zeros):
    pk_times, pmt_wfs = pick_slice_and_rebin(indices, times,
                                             ccwf   , rebin_stride,
                                             pad_zeros = pad_zeros)
    return pk_times, PMTResponses(pmt_ids, pmt_wfs)


def build_sipm_responses(indices, times, sipm_wfs, rebin_stride, thr_sipm_s2):
    _, sipm_wfs_ = pick_slice_and_rebin(indices , times,
                                        sipm_wfs, rebin_stride,
                                        pad_zeros = False)
    (sipm_ids,
     sipm_wfs)   = select_wfs_above_time_integrated_thr(sipm_wfs_,
                                                        thr_sipm_s2)
    return SiPMResponses(sipm_ids, sipm_wfs)


def build_peak(indices, times,
               ccwf, pmt_ids,
               rebin_stride,
               with_sipms, Pk,
               sipm_wfs    = None,
               thr_sipm_s2 = 0):
    (pk_times,
     pmt_r   ) = build_pmt_responses(indices, times,
                                     ccwf, pmt_ids,
                                     rebin_stride, pad_zeros = with_sipms)
    if with_sipms:
        sipm_r = build_sipm_responses(indices // 40, times // 40,
                                      sipm_wfs, rebin_stride // 40,
                                      thr_sipm_s2)
    else:
        sipm_r = SiPMResponses.build_empty_instance()

    return Pk(pk_times, pmt_r, sipm_r)


def find_peaks(ccwfs, index,
               time, length,
               stride, rebin_stride,
               Pk, pmt_ids,
               sipm_wfs=None, thr_sipm_s2=0):
    ccwfs = np.array(ccwfs, ndmin=2)

    peaks           = []
    times           = np.arange     (ccwfs.shape[1]) * 25 * units.ns
    indices_split   = split_in_peaks(index, stride)
    selected_splits = select_peaks  (indices_split, time, length)
    with_sipms      = Pk is S2 and sipm_wfs is not None

    for indices in selected_splits:
        pk = build_peak(indices, times,
                        ccwfs, pmt_ids,
                        rebin_stride,
                        with_sipms, Pk,
                        sipm_wfs, thr_sipm_s2)
        peaks.append(pk)
    return peaks


def get_pmap(ccwf, s1_indx, s2_indx, sipm_zs_wf,
             s1_params, s2_params, thr_sipm_s2, pmt_ids):
    return PMap(find_peaks(ccwf, s1_indx, Pk=S1, pmt_ids=pmt_ids, **s1_params),
                find_peaks(ccwf, s2_indx, Pk=S2, pmt_ids=pmt_ids,
                           sipm_wfs    = sipm_zs_wf,
                           thr_sipm_s2 = thr_sipm_s2,
                           **s2_params))


def rebin_times_and_waveforms(times, waveforms, rebin_stride):
    if rebin_stride < 2: return times, waveforms

    n_bins    = int(np.ceil(len(times) / rebin_stride))
    n_sensors = waveforms.shape[0]

    rebinned_times = np.zeros(            n_bins )
    rebinned_wfs   = np.zeros((n_sensors, n_bins))

    for i in range(n_bins):
        s  = slice(rebin_stride * i, rebin_stride * (i + 1))
        t  = times    [   s]
        e  = waveforms[:, s]
        w  = np.sum(e, axis=0) if np.any(e) else None
        rebinned_times[   i] = np.average(t, weights=w)
        rebinned_wfs  [:, i] = np.sum    (e,    axis=1)
    return rebinned_times, rebinned_wfs
