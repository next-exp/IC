from ..evm.pmaps           import  PMTResponses
from ..evm.pmaps           import SiPMResponses
from . peak_functions      import rebin_times_and_waveforms
from ..core.core_functions import dict_filter_by_key

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
