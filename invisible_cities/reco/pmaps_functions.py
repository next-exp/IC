from ..evm.pmaps      import  PMTResponses
from ..evm.pmaps      import SiPMResponses
from . peak_functions import rebin_times_and_waveforms


def rebin_peak(peak, rebin_factor):
    if rebin_factor == 1: return peak

    times, pmt_wfs = rebin_times_and_waveforms(peak.times,
                                               peak.pmts.all_waveforms,
                                               rebin_factor)
    pmt_r  = PMTResponses(peak.pmts.ids, pmt_wfs)

    sipm_r = SiPMResponses.build_empty_instance()
    if peak.sipms.ids.size:
        _, sipms = rebin_times_and_waveforms(peak.times,
                                             peak.sipms.all_waveforms,
                                             rebin_factor)
        sipm_r = SiPMResponses(peak.sipms.ids, sipms)

    return type(peak)(times, pmt_r, sipm_r)
