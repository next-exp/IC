from argparse import Namespace

import numpy  as np
from .. core.system_of_units_c import units
from .. types.ic_types_c import minmax
from .. evm.pmaps import Peak
from .. evm.pmaps import S12
from .. evm.pmaps import S1
from .. evm.pmaps import S2
from .. evm.pmaps import S2Si
from typing import List
#from ..reco.pmaps_functions   import integrate_S2Si_charge
#
# def select_peaks(s1,
#                  Emin, Emax,
#                  Lmin, Lmax,
#                  Hmin, Hmax,
#                  Ethr = -1):
#
#     is_valid = lambda E: (Lmin <= np.size(E) < Lmax and
#                           Hmin <= np.max (E) < Hmax and
#                           Emin <= np.sum (E) < Emax)
#
#     for peak_no in s1.peak_collection():
#         peak = s1.peak_waveform(peak_no)
#         if valid_peak()
#
#     return {peak_no: (t, E) for peak_no, (t, E) in peaks.items() if is_valid(E[E > Ethr])}
#
#
# def select_Si(peaks,
#               Nmin, Nmax):
#     is_valid = lambda sipms: Nmin <= len(sipms) < Nmax
#     return {peak_no: sipms for peak_no, sipms in peaks.items() if is_valid(sipms)}


class S12Selector:
    def __init__(self, **kwds):
        conf = Namespace(**kwds)
        self.s1n = minmax(conf.s1_nmin, conf.s1_nmax)
        self.s1e = minmax(conf.s1_emin, conf.s1_emax)
        self.s1w = minmax(conf.s1_wmin, conf.s1_wmax)
        self.s1h = minmax(conf.s1_hmin, conf.s1_hmax)
        self.s1_ethr = conf.s1_ethr
        self.s2n = minmax(conf.s2_nmin, conf.s2_nmax)
        self.s2e = minmax(conf.s2_emin, conf.s2_emax)
        self.s2w = minmax(conf.s2_wmin, conf.s2_wmax)
        self.s2h = minmax(conf.s2_hmin, conf.s2_hmax)
        self.nsi = minmax(conf.s2_nsipmmin, conf.s2_nsipmmax)
        self.s2_ethr = conf.s2_ethr


        # self.S1_Nmin     = conf.s1_nmin
        # self.S1_Nmax     = conf.s1_nmax
        # self.S1_Emin     = conf.s1_emin
        # self.S1_Emax     = conf.s1_emax
        # self.S1_Lmin     = conf.s1_lmin
        # self.S1_Lmax     = conf.s1_lmax
        # self.S1_Hmin     = conf.s1_hmin
        # self.S1_Hmax     = conf.s1_hmax
        # self.S1_Ethr     = conf.s1_ethr
        #
        # self.S2_Nmin     = conf.s2_nmin
        # self.S2_Nmax     = conf.s2_nmax
        # self.S2_Emin     = conf.s2_emin
        # self.S2_Emax     = conf.s2_emax
        # self.S2_Lmin     = conf.s2_lmin
        # self.S2_Lmax     = conf.s2_lmax
        # self.S2_Hmin     = conf.s2_hmin
        # self.S2_Hmax     = conf.s2_hmax
        # self.S2_NSIPMmin = conf.s2_nsipmmin
        # self.S2_NSIPMmax = conf.s2_nsipmmax
        # self.S2_Ethr     = conf.s2_ethr

    @staticmethod
    def valid_peak(peak : Peak, thr : float,
                   energy : minmax, width: minmax, height : minmax) -> bool :
        """Returns True if the peak energy, width and height
        is contained in the minmax defined by energy, width and
        height.

        """
        #import pdb; pdb.set_trace()
        f1 = energy.contains(peak.total_energy_above_trheshold(thr))
        f2 = width.contains(peak.width_above_trheshold(thr))
        f3 = height.contains(peak.height_above_trheshold(thr))

        return f1 and f2 and f3

    @staticmethod
    def select_valid_peaks(s12 : S12, thr: float,
                           energy : minmax, width: minmax, height : minmax) ->List[int] :
        """Takes an s1/s2 and returns a list of peaks that are contained
        within the boundaries defined by energy, width and height.

        """
        # valid_peaks = 0
        # for peak_no in s1.peak_collection():
        #     peak = s1.peak_waveform(peak_no)
        #     if valid_peak(energy, width, height, peak):
        #         valid_peaks+=1

        # less clear than above
        # valid_peaks = {peak_no: s1.peak_waveform(peak_no) for peak_no in s1.peak_collection()
        # if valid_peak(self.s1e, self.s1w, self.s1h, s1.peak_waveform(peak_no))
        # }
        valid_peaks = [peak_no for peak_no in s12.peak_collection()
                      if S12Selector.valid_peak(s12.peak_waveform(peak_no), thr,
                                    energy, width, height)]

        return valid_peaks

    @staticmethod
    def select_s2si(s2si : S2Si, nsipm : minmax) -> List[int]:
        """Takes an s2si and returns a list of peaks that are contained
        within the boundaries defined by nsipm.

        """
        # valid_peaks = {}
        # for peak_no in s2si.peak_collection():
        #     nof_sipm = s2si.number_of_sipms_in_peak(peak_no)
        #     if nsipm.contains(nof_sipm):
        #         valid_peaks[peak_no] = nof_sipm

        valid_peaks = [peak_no for peak_no in s2si.peak_collection()
                       if nsipm.contains(s2si.number_of_sipms_in_peak(peak_no))]
        return valid_peaks

    def select_s1(self, s1 : S1) -> int:
        """Takes an s1 and returns the number of peaks that pass the filter"""
        peak_list = self.select_valid_peaks(s1, self.s1_ethr,
                                            self.s1e, self.s1w, self.s1h)
        return len(peak_list)

    def select_s2(self, s2 : S2, s2si : S2Si) -> int:
        """Takes an s2/s2si and returns the number of peaks that pass the filter"""
        s2_peak_list   = self.select_valid_peaks(s2, self.s2_ethr,
                                                 self.s2e, self.s2w, self.s2h)
        s2si_peak_list = self.select_s2si(s2si, self.nsi)

        valid_peaks = set(s2_peak_list) & set(s2si_peak_list)
        # s2s = {peak_no: peak for peak_no, peak in s2df.items() if peak_no in valid_peaks}
        # sis = {peak_no: peak for peak_no, peak in sidf.items() if peak_no in valid_peaks}
        return len(valid_peaks)


def s1s2_filter(selector : S12Selector, s1 : S1, s2 : S2, s2si : S2Si) ->bool:
    """Takes the event pmaps (s1, s2 and  s2si)
    and filters the corresponding peaks in terms of the selector.
    1. select_s1 returns the number of s1 peaks whose energy, width and height
       are within the boundaries defined by the selector parameters.
    2. select_s2 returns the number of s2 peaks whose energy, width and height
       are within the boundaries defined by the selector parameters and
       AND which have a number of sipms within boundaries.

    """

    # s1f, s2f and sif are filtered dicts, containining
    # the peaks that pass selection
    n_s1_peaks           = selector.select_s1(s1)
    n_s2_peaks           = selector.select_s2(s2, s2si)


    return (selector.s1n.contains(n_s1_peaks) and
            selector.s2n.contains(n_s2_peaks))

def s2si_filter(s2si : S2Si) -> bool :
    """All peaks must contain at least one non-zero charged sipm"""

    def at_least_one_sipm_with_Q_gt_0(Si):
        return any(q > 0 for q in Si.values())

    def all_peaks_contain_at_least_one_non_zero_charged_sipm(iS2Si):
        return all(at_least_one_sipm_with_Q_gt_0(Si)
                          for Si in iS2Si.values())
    #iS2Si = integrate_S2Si_charge(s2si.s2sid)
    iS2Si = s2si.peak_and_sipm_total_energy_dict()
    return all_peaks_contain_at_least_one_non_zero_charged_sipm(iS2Si)
