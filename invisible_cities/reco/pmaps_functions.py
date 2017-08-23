"""Python (private) functions used for testing.

1. _integrate_sipm_charges_in_peak(s2si, peak_number)
Returns (np.array[[nsipm_1 ,     nsipm_2, ...]],
         np.array[[sum(q_1), sum(nsipm_2), ...]])

Last revised, JJGC, July, 2017.

"""
import copy
import numpy  as     np
from .. reco import pmaps_functions_c as pmpc
from .. core import core_functions_c as ccf
from .. core.system_of_units_c      import units
from .. core.exceptions             import NegativeThresholdNotAllowed
from .. evm.pmaps                   import S2, S2Si, Peak

from typing import Dict

def _integrate_sipm_charges_in_peak_as_dict(s2si):
    """Return dict of integrated charges from a SiPM dictionary.

    S2Si = {  peak : Si }
      Si = { nsipm : [ q1, q2, ...] }

    Returns an integrated Si = { nsipm : sum(q_n) }
    """
    return { sipm : sum(qs) for (sipm, qs) in s2si.items() }


def _integrate_sipm_charges_in_peak(s2si):
    """Return arrays of nsipm and integrated charges from SiPM dictionary.

    S2Si = {  peak : Si }
      Si = { nsipm : [ q1, q2, ...] }

    Returns (np.array[[nsipm_1 ,     nsipm_2, ...]],
             np.array[[sum(q_1), sum(nsipm_2), ...]])
    """
    sipms_and_Q_totals = _integrate_sipm_charges_in_peak_as_dict(s2si)
    sipms = np.array(tuple(sipms_and_Q_totals.keys()))
    Qs    = np.array(tuple(sipms_and_Q_totals.values()))
    return sipms, Qs


def _integrate_S2Si_charge(s2sid):
    """Return s2Si containing integrated charges.

    s2sid = { peak_no : {nsipm : [ q1, q2, ...]} }
    Returns s2sid where s2sid = {peak_no: { nsipm : sum([q1, q2, ...])} }
    """
    return { peak_no : _integrate_sipm_charges_in_peak_as_dict(peak)
             for (peak_no, peak) in s2sid.items() }

def rebin_s2si(s2, s2si, rf):
    """given an s2 and a corresponding s2si, rebin them by a factor rf"""
    assert rf >= 1 and rf % 1 == 0
    s2d_rebin   = {}
    s2sid_rebin = {}
    for pn in s2.peaks:
        if pn in s2si.peaks:
            t, e, sipms = rebin_s2si_peak(s2.peaks[pn].t, s2.peaks[pn].E, s2si.s2sid[pn], rf)
            s2sid_rebin[pn] = sipms
        else:
            t, e, _ = rebin_s2si_peak(s2.peaks[pn].t, s2.peaks[pn].E, {}, rf)

        s2d_rebin[pn] = [t, e]


    return S2(s2d_rebin), S2Si(s2d_rebin, s2sid_rebin)


def rebin_s2si_peak(t, e, sipms, stride):
    """rebin: s2 times (taking mean), s2 energies, and s2 sipm qs, by stride"""
    # cython rebin_array is returning memoryview so we need to cast as np array
    return   ccf.rebin_array(t , stride, remainder=True, mean=True), \
             ccf.rebin_array(e , stride, remainder=True)           , \
      {sipm: ccf.rebin_array(qs, stride, remainder=True) for sipm, qs in sipms.items()}


def copy_s2si(s2si_original : S2Si) -> S2Si:
    """ return an identical copy of an s2si. ** note these must be deepcopies, and a deepcopy of
    the s2si itself does not seem to work. """
    return S2Si(copy.deepcopy(s2si_original.s2d),
                copy.deepcopy(s2si_original.s2sid))


def copy_s2si_dict(s2si_dict_original: Dict[int, S2Si]) -> Dict[int, S2Si]:
    """ returns an identical copy of the input s2si_dict """
    return {ev: copy_s2si(s2si) for ev, s2si in s2si_dict_original.items()}


def raise_s2si_thresholds(s2si_dict_original: Dict[int, S2Si],
                         thr_sipm           : float,
                         thr_sipm_s2        : float) -> Dict[int, S2Si]:
    """
    returns s2si_dict after imposing more thr_sipm and/or thr_sipm_s2 thresholds.
    ** NOTE:
        1) thr_sipm IS IMPOSED BEFORE thr_sipm_s2
        2) thresholds cannot be lowered. this function will do nothing if thresholds are set below
           previous values.
    """
    # Ensure thresholds are acceptable values
    if thr_sipm     is None: thr_sipm    = 0
    if thr_sipm_s2  is None: thr_sipm_s2 = 0
    if thr_sipm < 0 or thr_sipm_s2 < 0:
        raise NegativeThresholdNotAllowed('Threshold can be 0 or None, but not negative')
    elif thr_sipm == 0 and thr_sipm_s2 == 0: return s2si_dict_original
    else: s2si_dict = copy_s2si_dict(s2si_dict_original)

    # Impose thresholds
    if thr_sipm    > 0:
        s2si_dict  = pmpc._impose_thr_sipm_destructive   (s2si_dict, thr_sipm   )
    if thr_sipm_s2 > 0:
        s2si_dict  = pmpc._impose_thr_sipm_s2_destructive(s2si_dict, thr_sipm_s2)
    # Get rid of any empty dictionaries
    if thr_sipm > 0 or thr_sipm_s2 > 0:
        s2si_dict  = pmpc._delete_empty_s2si_peaks      (s2si_dict)
        s2si_dict  = pmpc._delete_empty_s2si_dict_events(s2si_dict)
    return s2si_dict


# def select_si_slice(si, slice_no):
#     # This is a temporary fix! The number of slices in the SiPM arrays
#     # must match that of the PMT PMaps.
#     return {sipm_no: (sipm[slice_no] if len(sipm) > slice_no else 0)
#                       for sipm_no, sipm in si.items()}
