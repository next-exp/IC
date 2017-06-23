# import numpy as np

import numpy  as np
from .. core.system_of_units_c import units
#from .. reco                   import peak_functions as pf
from ..reco.pmaps_functions   import integrate_S2Si_charge

def select_peaks(peaks,
                 Emin, Emax,
                 Lmin, Lmax,
                 Hmin, Hmax,
                 Ethr = -1):

    is_valid = lambda E: (Lmin <= np.size(E) < Lmax and
                          Hmin <= np.max (E) < Hmax and
                          Emin <= np.sum (E) < Emax)

    return {peak_no: (t, E) for peak_no, (t, E) in peaks.items() if is_valid(E[E > Ethr])}


def select_Si(peaks,
              Nmin, Nmax):
    is_valid = lambda sipms: Nmin <= len(sipms) < Nmax
    return {peak_no: sipms for peak_no, sipms in peaks.items() if is_valid(sipms)}


class S12Selector:
    def __init__(self,
                 S1_Nmin     = 0,
                 S1_Nmax     = 1000,
                 S1_Emin     = 0,
                 S1_Emax     = np.inf,
                 S1_Lmin     = 0,
                 S1_Lmax     = np.inf,
                 S1_Hmin     = 0,
                 S1_Hmax     = np.inf,
                 S1_Ethr     = 0,

                 S2_Nmin     = 0,
                 S2_Nmax     = 1000,
                 S2_Emin     = 0,
                 S2_Emax     = np.inf,
                 S2_Lmin     = 0,
                 S2_Lmax     = np.inf,
                 S2_Hmin     = 0,
                 S2_Hmax     = np.inf,
                 S2_NSIPMmin = 1,
                 S2_NSIPMmax = np.inf,
                 S2_Ethr     = 0):

        self.S1_Nmin     = S1_Nmin
        self.S1_Nmax     = S1_Nmax
        self.S1_Emin     = S1_Emin
        self.S1_Emax     = S1_Emax
        self.S1_Lmin     = S1_Lmin
        self.S1_Lmax     = S1_Lmax
        self.S1_Hmin     = S1_Hmin
        self.S1_Hmax     = S1_Hmax
        self.S1_Ethr     = S1_Ethr

        self.S2_Nmin     = S2_Nmin
        self.S2_Nmax     = S2_Nmax
        self.S2_Emin     = S2_Emin
        self.S2_Emax     = S2_Emax
        self.S2_Lmin     = S2_Lmin
        self.S2_Lmax     = S2_Lmax
        self.S2_Hmin     = S2_Hmin
        self.S2_Hmax     = S2_Hmax
        self.S2_NSIPMmin = S2_NSIPMmin
        self.S2_NSIPMmax = S2_NSIPMmax
        self.S2_Ethr     = S2_Ethr

    def select_S1(self, s1s):
        return select_peaks(s1s,
                               self.S1_Emin, self.S1_Emax,
                               self.S1_Lmin, self.S1_Lmax,
                               self.S1_Hmin, self.S1_Hmax,
                               self.S1_Ethr)

    def select_S2(self, s2s, sis):
        s2s = select_peaks(s2s,
                              self.S2_Emin, self.S2_Emax,
                              self.S2_Lmin, self.S2_Lmax,
                              self.S2_Hmin, self.S2_Hmax,
                              self.S2_Ethr)
        sis = select_Si(sis,
                           self.S2_NSIPMmin, self.S2_NSIPMmax)

        valid_peaks = set(s2s) & set(sis)
        s2s = {peak_no: peak for peak_no, peak in s2s.items() if peak_no in valid_peaks}
        sis = {peak_no: peak for peak_no, peak in sis.items() if peak_no in valid_peaks}
        return s2s, sis


def s1s2_filter(selector, s1s, s2s, sis):

    S1     = selector.select_S1(s1s)
    S2, Si = selector.select_S2(s2s, sis)

    return (selector.S1_Nmin <= len(S1) <= selector.S1_Nmax and
            selector.S2_Nmin <= len(S2) <= selector.S2_Nmax)

def s2si_filter(S2Si):
    """All peaks must contain at least one non-zero charged sipm"""

    def at_least_one_sipm_with_Q_gt_0(Si):
        return any(q > 0 for q in Si.values())

    def all_peaks_contain_at_least_one_non_zero_charged_sipm(iS2Si):
        return all(at_least_one_sipm_with_Q_gt_0(Si)
                          for Si in iS2Si.values())
    iS2Si = integrate_S2Si_charge(S2Si)
    return all_peaks_contain_at_least_one_non_zero_charged_sipm(iS2Si)
