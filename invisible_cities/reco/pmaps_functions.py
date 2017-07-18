"""Python (private) functions used for testing.

1. _integrate_sipm_charges_in_peak(s2si, peak_number)
Returns (np.array[[nsipm_1 ,     nsipm_2, ...]],
         np.array[[sum(q_1), sum(nsipm_2), ...]])

Last revised, JJGC, July, 2017.

"""
import numpy  as np
from .. core import core_functions_c as ccf
from .. core.system_of_units_c      import units
from .. evm.pmaps                   import S2, S2Si, Peak


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
    s2d_rebin = {}
    s2sid_rebin = {}
    for pn in s2.peaks:
        t, e, sipms = rebin_s2si_peak(s2.peaks[pn].t, s2.peaks[pn].E, s2si.s2sid[pn], rf)
        s2d_rebin  [pn] = Peak(t, e)
        s2sid_rebin[pn] = sipms

    return S2(s2d_rebin), S2Si(s2d_rebin, s2sid_rebin)


def rebin_s2si_peak(t, e, sipms, stride):
    """rebin: s2 times (taking mean), s2 energies, s2 sipm qs, by stride"""

    # cython rebin_array is returning memoryview so we need to cast as np array
    return   np.asarray(ccf.rebin_array(t , stride, remainder=True, mean=True)), \
             np.asarray(ccf.rebin_array(e , stride, remainder=True))           , \
      {sipm: np.asarray(ccf.rebin_array(qs, stride, remainder=True)) for sipm, qs in sipms.items()}

# def select_si_slice(si, slice_no):
#     # This is a temporary fix! The number of slices in the SiPM arrays
#     # must match that of the PMT PMaps.
#     return {sipm_no: (sipm[slice_no] if len(sipm) > slice_no else 0)
#                       for sipm_no, sipm in si.items()}
