from enum import Enum

import numpy as np

from ..evm.pmaps           import  PMTResponses
from ..evm.pmaps           import SiPMResponses
from . peak_functions      import rebin_times_and_waveforms
from ..core.core_functions import dict_filter_by_key


class RebinMethod(Enum):
    stride    = 0
    threshold = 1


def get_even_slices(bins, stride):
    new_bins = int(np.ceil(bins / stride))
    return [slice(stride * i, stride * (i + 1)) for i in range(new_bins)]


def get_theshold_slices(pmt_sum, threshold):
    slices     = []
    last_index = 0
    last_sum   = 0
    for i, sum_val in enumerate(np.cumsum(pmt_sum)):
        if sum_val - last_sum >= threshold or i + 1 == len(pmt_sum):
            slices.append(slice(last_index, i + 1))
            last_index = i + 1
            last_sum   = sum_val
    if pmt_sum[slices[-1]].sum() < threshold and len(slices) > 1:
        slices[-2:] = [slice(slices[-2].start, slices[-1].stop)]
    return slices


def rebin_peak(peak, rebin_factor, model=RebinMethod.stride):
    """
    Rebin a peak either using a set stride of
    a set number of existing bins or a charge
    threshold.

    Parameters
    ----------
    peak : _Peak object
        The Peak to be rebinned
    rebin_factor : int or float
        The rebin stride for RebinMethod.stride (int)
        or the rebin threshold for RebinMethod.threshold (float)
    model : Enum
        The model to be used, stride for set bin width
        threshold for minimum pmt sum charge.

    Returns
    -------
    The rebinned version of the peak
    """
    if model == RebinMethod.threshold:
        slices = get_theshold_slices(peak.pmts.sum_over_sensors,
                                     rebin_factor)
    else:
        if rebin_factor <= 1: return peak
        slices = get_even_slices(peak.times.shape[0], rebin_factor)

    return rebin_peak_to_slices(peak, slices)


def rebin_peak_to_slices(peak, slices):

    times, pmt_wfs = rebin_times_and_waveforms(peak.times,
                                               peak.bin_widths,
                                               peak.pmts.all_waveforms,
                                               slices = slices)
    
    pmt_r  = PMTResponses(peak.pmts.ids, pmt_wfs)

    sipm_r = SiPMResponses.build_empty_instance()
    if peak.sipms.ids.size:
        *_, sipms = rebin_times_and_waveforms(peak.times,
                                              peak.bin_widths,
                                              peak.sipms.all_waveforms,
                                              slices = slices)
        
        sipm_r = SiPMResponses(peak.sipms.ids, sipms)

    return type(peak)(times, widths, pmt_r, sipm_r)


## def rebin_peak_threshold(peak, threshold):
##     """
##     Peak rebinning function based on a minimum
##     charge in the pmt sum to form a slice.

##     Parameters
##     ----------
##     peak : object
##         The peak object to be rebinned
##     threshold : float
##         Minimum charge (pe) required in the pmt sum
##         to be counted as a slice.

##     Returns
##     -------
##     rebinned_peak : peak object
##         The peak rebinned according to the threshold.
##     """
##     pmt_sum = peak.pmts.sum_over_sensors

##     slices     = []
##     last_index = 0
##     last_sum   = 0
##     for i, sum_val in enumerate(np.cumsum(pmt_sum)):
##         if sum_val - last_sum >= threshold or i + 1 == len(pmt_sum) - 1:
##             slices.append(slice(last_index, i + 1))
##             last_index = i + 1
##             last_sum   = sum_val
##     if pmt_sum[slices[-1]].sum() < threshold:
##         last_slice = [slice(slices[-2].start, slices[-1].stop)]
##         slices[-2:] = last_slice

##     n_bins = len(slices)
##     times  = peak.times
##     pmts   = peak.pmts.all_waveforms
##     sipms  = peak.sipms.all_waveforms
##     rebinned_times = np.zeros(                 n_bins )
##     rebinned_pmts  = np.zeros((pmts.shape[0] , n_bins))
##     rebinned_sipms = np.zeros((sipms.shape[0], n_bins))

##     for i, s in enumerate(slices):
##         t = times[   s]
##         e = pmts [:, s]
##         q = sipms[:, s]
##         w = np.sum(e, axis=0) if np.any(e) else None
##         rebinned_times[   i] = np.average(t, weights=w)
##         rebinned_pmts [:, i] = np.sum    (e,    axis=1)
##         rebinned_sipms[:, i] = np.sum    (q,    axis=1)
##     pmt_r  = PMTResponses(peak.pmts.ids, rebinned_pmts)
##     sipm_r = SiPMResponses(peak.sipms.ids, rebinned_sipms)
##     return type(peak)(rebinned_times, pmt_r, sipm_r)


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
