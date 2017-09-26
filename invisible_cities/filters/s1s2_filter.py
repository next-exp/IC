from argparse  import Namespace
from functools import partial
from textwrap  import dedent
from typing    import Dict

import numpy as np

from .. types.ic_types_c import minmax
from .. evm  .pmaps      import Peak
from .. evm  .pmaps      import S12
from .. evm  .pmaps      import S1
from .. evm  .pmaps      import S2
from .. evm  .pmaps      import S2Si


class S12SelectorOutput:
    def __init__(self, passed, s1_peaks, s2_peaks):
        self.passed   = passed
        self.s1_peaks = s1_peaks
        self.s2_peaks = s2_peaks

    def __and__(self, other):
        s1_peaks = set(self.s1_peaks) | set(other.s1_peaks)
        s2_peaks = set(self.s2_peaks) | set(other.s2_peaks)

        passed   = self.passed and other.passed

        s1_peaks = {peak_no: ( self.s1_peaks.get(peak_no, False) and
                              other.s1_peaks.get(peak_no, False))
                    for peak_no in s1_peaks}

        s2_peaks = {peak_no: ( self.s2_peaks.get(peak_no, False) and
                              other.s2_peaks.get(peak_no, False))
                    for peak_no in s2_peaks}

        return S12SelectorOutput(passed, s1_peaks, s2_peaks)

    def __or__(self, other):
        s1_peaks = set(self.s1_peaks) | set(other.s1_peaks)
        s2_peaks = set(self.s2_peaks) | set(other.s2_peaks)

        passed   = self.passed or other.passed

        s1_peaks = {peak_no: ( self.s1_peaks.get(peak_no, False) or
                              other.s1_peaks.get(peak_no, False))
                    for peak_no in s1_peaks}

        s2_peaks = {peak_no: ( self.s2_peaks.get(peak_no, False) or
                              other.s2_peaks.get(peak_no, False))
                    for peak_no in s2_peaks}

        return S12SelectorOutput(passed, s1_peaks, s2_peaks)

    def __str__(self):
        return dedent("""
                  S12SelectorOutput:
                      Passed  : {self.passed}
                      s1_peaks: {self.s1_peaks}
                      s2_peaks: {self.s2_peaks}
                      """.format(self = self))

    __repr__ = __str__


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

    @staticmethod
    def valid_peak(peak   : Peak,
                   thr    : float,
                   energy : minmax,
                   width  : minmax,
                   height : minmax) -> bool:
        """Returns True if the peak energy, width and height
        is contained in the minmax defined by energy, width and
        height."""
        f1 = energy.contains(peak.total_energy_above_threshold(thr))
        f2 = width .contains(peak.       width_above_threshold(thr))
        f3 = height.contains(peak.      height_above_threshold(thr))

        return f1 and f2 and f3

    @staticmethod
    def select_valid_peaks(s12    : S12,
                           thr    : float,
                           energy : minmax,
                           width  : minmax,
                           height : minmax) ->Dict[int, bool]:
        """Takes a s1/s2 and returns a dictionary with the outcome of the
        filter for each peak"""
        peak_is_valid = partial(S12Selector.valid_peak,
                                thr    = thr,
                                energy = energy,
                                width  = width,
                                height = height)
        valid_peaks   = {peak_no: peak_is_valid(s12.peak_waveform(peak_no))
                         for peak_no in s12.peak_collection()}
        return valid_peaks

    @staticmethod
    def select_s2si(s2si  : S2Si,
                    nsipm : minmax) -> Dict[int, bool]:
        """Takes a s2si and returns a dictionary with the outcome of the
        filter for each peak"""
        valid_peaks = {peak_no: nsipm.contains(s2si.number_of_sipms_in_peak(peak_no))
                       for peak_no in s2si.peak_collection()}
        return valid_peaks

    def select_s1(self, s1 : S1) -> Dict[int, bool]:
        """Takes a s1 and returns a dictionary with the outcome of the
        filter for each peak"""
        pass_dict = self.select_valid_peaks(s1,
                                            self.s1_ethr,
                                            self.s1e,
                                            self.s1w,
                                            self.s1h)
        return pass_dict

    def select_s2(self, s2 : S2, s2si : S2Si) -> Dict[int, bool]:
        """Takes a s2 and a s2si and returns a dictionary with the
        outcome of the filter for each peak"""
        s2_pass_dict   = self.select_valid_peaks(s2,
                                                 self.s2_ethr,
                                                 self.s2e,
                                                 self.s2w,
                                                 self.s2h)
        s2si_pass_dict = self.select_s2si(s2si, self.nsi)
        combined       = (S12SelectorOutput(None, {},   s2_pass_dict) &
                          S12SelectorOutput(None, {}, s2si_pass_dict))
        return combined.s2_peaks

    def __str__(self):
        return dedent("""
                   S12_selector:
                       s1n     = {self.s1n}
                       s1e     = {self.s1e} pes
                       s1w     = {self.s1w} ns
                       s1h     = {self.s1h} pes
                       s1_ethr = {self.s1_ethr} pes
                       s2n     = {self.s2n}
                       s2e     = {self.s2e} pes
                       s2w     = {self.s2w} ns
                       s2h     = {self.s2h} pes
                       nsipm   = {self.nsi}
                       s2_ethr = {self.s2_ethr} pes""".format(self = self))

    __repr__ = __str__


def s1s2_filter(selector : S12Selector,
                s1       : S1,
                s2       : S2,
                s2si     : S2Si) -> S12SelectorOutput:
    """Takes the event pmaps (s1, s2 and  s2si)
    and filters the corresponding peaks in terms of the selector.
    1. select_s1 returns the s1 peak numbers whose energy, width and height
       are within the boundaries defined by the selector parameters.
    2. select_s2 returns the s2 peak numbers whose energy, width and height
       are within the boundaries defined by the selector parameters and
       AND which have a number of sipms within boundaries.

    """
    selected_s1_peaks = selector.select_s1(s1)
    selected_s2_peaks = selector.select_s2(s2, s2si)

    passed = (selector.s1n.contains(np.count_nonzero(list(selected_s1_peaks.values()))) and
              selector.s2n.contains(np.count_nonzero(list(selected_s2_peaks.values()))))

    return S12SelectorOutput(passed, selected_s1_peaks, selected_s2_peaks)


def s2si_filter(s2si : S2Si) -> S12SelectorOutput:
    """All peaks must contain at least one non-zero charged sipm"""

    def at_least_one_sipm_with_Q_gt_0(Si):
        return any(q > 0 for q in Si.values())

    iS2Si = s2si.peak_and_sipm_total_energy_dict()

    selected_si_peaks = {peak_no: at_least_one_sipm_with_Q_gt_0(peak)
                         for peak_no, peak in iS2Si.items()}
    passed = len(selected_si_peaks) > 0
    return S12SelectorOutput(passed, {}, selected_si_peaks)


def s1s2si_filter(selector : S12Selector,
                  s1       : S1,
                  s2       : S2,
                  s2si     : S2Si) -> S12SelectorOutput:
    """Combine s1s2 and s2si filters"""
    s1s2f = s1s2_filter(selector, s1, s2, s2si)
    s2sif = s2si_filter(s2si)

    # Set s1 peaks to s2si filter so it can be anded.
    s2sif.s1_peaks = s1s2f.s1_peaks
    return s1s2f & s2sif
