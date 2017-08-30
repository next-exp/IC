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


def _sipm_ids_and_charges_in_slice(s2sid_peak, slice_no):
    """Given s2sid_peak = {nsipm : [ q1, q2, ...qn]} and a slice_no
    (running from 1, 2..n) returns:
    Returns (np.array[nsipm_1 , nsipm_2, ...],
             np.array[q_k from nsipm_1, q_k from nsipm_2, ...]]) when slice_no=k
     """
    number_of_sipms = len(s2sid_peak.keys())
    ids      = []
    qs_slice = []
    for i, (nsipm, qs) in enumerate(s2sid_peak.items()):
        if qs[slice_no] > 0:
            ids.append(nsipm)
            qs_slice.append(qs[slice_no])

    return np.array(ids), n.array(qs_slice)


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


def _impose_thr_sipm_destructive(s2si_dict: Dict[int, S2Si],
                                 thr_sipm : float           ) -> Dict[int, S2Si]:
    """imposes a thr_sipm on s2si_dict"""
    for s2si in s2si_dict.values():                   # iter over events
        for si_peak in s2si.s2sid.values():           # iter over peaks
            for sipm in list(si_peak.keys()):         # iter over sipms ** avoid mod while iter
                for i, q in enumerate(si_peak[sipm]): # iter over timebins
                    if q < thr_sipm:                  # impose threshold
                        si_peak[sipm][i] = 0
                if si_peak[sipm].sum() == 0:          # Delete SiPMs with integral
                    del si_peak[sipm]                 # charge equal to 0
    return s2si_dict


def _impose_thr_sipm_s2_destructive(s2si_dict   : Dict[int, S2Si],
                                    thr_sipm_s2 : float           ) -> Dict[int, S2Si]:
    """imposes a thr_sipm_s2 on s2si_dict. deletes keys (sipms) from each s2sid peak if sipm
       integral charge is less than thr_sipm_s2"""
    for s2si in s2si_dict.values():
        for si_peak in s2si.s2sid.values():
            for sipm, qs in list(si_peak.items()): # ** avoid modifying while iterating
                sipm_integral_charge = qs.sum()
                if sipm_integral_charge < thr_sipm_s2:
                    del si_peak[sipm]
    return s2si_dict


def _delete_empty_s2si_peaks(s2si_dict : Dict[int, S2Si]) -> Dict[int, S2Si]:
    """makes sure there are no empty peaks stored in an s2sid
        (s2sid[pn] != {} for all pn in s2sid and all s2sid in s2si_dict)
        ** Also deletes corresponding peak in s2si.s2d! """
    for ev in list(s2si_dict.keys()):
        for pn in list(s2si_dict[ev].s2sid.keys()):
            if len(s2si_dict[ev].s2sid[pn]) == 0:
                del s2si_dict[ev].s2sid[pn]
                del s2si_dict[ev].s2d  [pn]
                # It is not sufficient to just delete the peaks because the S2Si class instance
                # will still think it has peak pn even though its base dictionary does not
                s2si_dict[ev] = S2Si(s2si_dict[ev].s2d, s2si_dict[ev].s2sid)
    return s2si_dict


def _delete_empty_s2si_dict_events(s2si_dict: Dict[int, S2Si]) -> Dict[int, S2Si]:
    """ delete all events from s2si_dict with empty s2sid"""
    for ev in list(s2si_dict.keys()):
        if len(s2si_dict[ev].s2sid) == 0:
            del s2si_dict[ev]
    return s2si_dict


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
