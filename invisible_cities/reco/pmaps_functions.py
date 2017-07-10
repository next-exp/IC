"""Python (private) functions used for testing.

1. _integrate_sipm_charges_in_peak(s2si, peak_number)
Returns (np.array[[nsipm_1 ,     nsipm_2, ...]],
         np.array[[sum(q_1), sum(nsipm_2), ...]])

Last revised, JJGC, July, 2017.

"""
import numpy  as np
from .. core.system_of_units_c import units
from .. core.core_functions    import rebin_array


def _integrate_sipm_charges_in_peak_as_dict(Si):
    """Return dict of integrated charges from a SiPM dictionary.

    S2Si = {  peak : Si }
      Si = { nsipm : [ q1, q2, ...] }

    Returns an integrated Si = { nsipm : sum(q_n) }
    """
    return { sipm : sum(qs) for (sipm, qs) in Si.items() }


def _integrate_sipm_charges_in_peak(Si):
    """Return arrays of nsipm and integrated charges from SiPM dictionary.

    S2Si = {  peak : Si }
      Si = { nsipm : [ q1, q2, ...] }

    Returns (np.array[[nsipm_1 ,     nsipm_2, ...]],
             np.array[[sum(q_1), sum(nsipm_2), ...]])
    """
    sipms_and_Q_totals = _integrate_sipm_charges_in_peak_as_dict(Si)
    sipms = np.array(tuple(sipms_and_Q_totals.keys()))
    Qs    = np.array(tuple(sipms_and_Q_totals.values()))
    return sipms, Qs


def _integrate_S2Si_charge(S2Si):
    """
    Return S2Si containing integrated charges.

    S2Si = {  peak : Si }
      Si = { nsipm : [ q1, q2, ...] }

    Returns S2Si where Si = { nsipm : sum([q1, q2, ...])}
    """
    return { peak_no : _integrate_sipm_charges_in_peak_as_dict(peak)
             for (peak_no, peak) in S2Si.items() }

def _rebin_S2(t, e, sipms, stride):
    """rebin: s2 times, s2 energies, s2 sipm qs, by stride"""
    return rebin_array(t, stride),
           rebin_array(e, stride),
           {sipm: rebin_array(qs, stride) for sipm, qs in sipms.items()}


# def select_si_slice(si, slice_no):
#     # This is a temporary fix! The number of slices in the SiPM arrays
#     # must match that of the PMT PMaps.
#     return {sipm_no: (sipm[slice_no] if len(sipm) > slice_no else 0)
#                       for sipm_no, sipm in si.items()}
