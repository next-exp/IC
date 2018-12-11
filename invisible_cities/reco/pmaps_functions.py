from ..evm.pmaps           import  PMTResponses
from ..evm.pmaps           import SiPMResponses
from . peak_functions      import rebin_times_and_waveforms
from ..core.core_functions import dict_filter_by_key

import numpy as np


def rebin_peak(peak, rebin_factor):
    if rebin_factor == 1: return peak

    times, widths, pmt_wfs = rebin_times_and_waveforms(peak.times,
                                                       peak.bin_widths,
                                                       peak.pmts.all_waveforms,
                                                       rebin_factor)
    pmt_r  = PMTResponses(peak.pmts.ids, pmt_wfs)

    sipm_r = SiPMResponses.build_empty_instance()
    if peak.sipms.ids.size:
        _, _, sipms = rebin_times_and_waveforms(peak.times,
                                                peak.bin_widths,
                                                peak.sipms.all_waveforms,
                                                rebin_factor)
        sipm_r = SiPMResponses(peak.sipms.ids, sipms)

    return type(peak)(times, widths, pmt_r, sipm_r)


def rebin_peak_threshold(peak, threshold):
    """
    Peak rebinning function based on a minimum
    charge in the pmt sum to form a slice.

    Parameters
    ----------
    peak : object
        The peak object to be rebinned
    threshold : float
        Minimum charge (pe) required in the pmt sum
        to be counted as a slice.

    Returns
    -------
    rebinned_peak : peak object
        The peak rebinned according to the threshold.
    """
    pmt_sum = peak.pmts.sum_over_sensors

    slices     = []
    last_index = 0
    last_sum   = 0
    for i, sum_val in enumerate(np.cumsum(pmt_sum)):
        if sum_val - last_sum >= threshold or i + 1 == len(pmt_sum) - 1:
            slices.append(slice(last_index, i + 1))
            last_index = i + 1
            last_sum   = sum_val
    if pmt_sum[slices[-1]].sum() < threshold:
        last_slice = [slice(slices[-2].start, slices[-1].stop)]
        slices[-2:] = last_slice

    n_bins = len(slices)
    times  = peak.times
    pmts   = peak.pmts.all_waveforms
    sipms  = peak.sipms.all_waveforms
    rebinned_times = np.zeros(                 n_bins )
    rebinned_pmts  = np.zeros((pmts.shape[0] , n_bins))
    rebinned_sipms = np.zeros((sipms.shape[0], n_bins))

    for i, s in enumerate(slices):
        t = times[   s]
        e = pmts [:, s]
        q = sipms[:, s]
        w = np.sum(e, axis=0) if np.any(e) else None
        rebinned_times[   i] = np.average(t, weights=w)
        rebinned_pmts [:, i] = np.sum    (e,    axis=1)
        rebinned_sipms[:, i] = np.sum    (q,    axis=1)
    pmt_r  = PMTResponses(peak.pmts.ids, rebinned_pmts)
    sipm_r = SiPMResponses(peak.sipms.ids, rebinned_sipms)
    return type(peak)(rebinned_times, pmt_r, sipm_r)


def pmap_event_id_selection(data, event_ids):
    """Filter a pmap dictionary by a list of event IDs.
    Parameters
    ----------
    data      : dict
        pmaps to be filtered.
    event_ids : list
        List of event ids that will be kept.
    Returns
    -------
    filterdata : dict
        Filtered pmaps
    """
    return dict_filter_by_key(lambda x: x in event_ids, data)
