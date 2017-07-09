"""PMAPS functions.
JJGC December 2016

"""
import numpy  as np
# import pandas as pd
# import tables as tb

from .. core.system_of_units_c import units


def width(times, to_mus=False):
    """
    Compute peak width. Times has to be ordered.
    """

    w = times[-1] - times[0] if len(times) > 0 else 0
    return w * units.ns/units.mus if to_mus else w


def integrate_sipm_charges_in_peak_as_dict(Si):
    """Return dict of integrated charges from a SiPM dictionary.

    S2Si = {  peak : Si }
      Si = { nsipm : [ q1, q2, ...] }

    Returns an integrated Si = { nsipm : sum(q_n) }
    """
    return { sipm : sum(qs) for (sipm, qs) in Si.items() }


def integrate_sipm_charges_in_peak(Si):
    """Return arrays of nsipm and integrated charges from SiPM dictionary.

    S2Si = {  peak : Si }
      Si = { nsipm : [ q1, q2, ...] }

    Returns (np.array[[nsipm_1 ,     nsipm_2, ...]],
             np.array[[sum(q_1), sum(nsipm_2), ...]])
    """
    sipms_and_Q_totals = integrate_sipm_charges_in_peak_as_dict(Si)
    sipms = np.array(tuple(sipms_and_Q_totals.keys()))
    Qs    = np.array(tuple(sipms_and_Q_totals.values()))
    return sipms, Qs


def integrate_S2Si_charge(S2Si):
    """Return S2Si containing integrated charges.

    S2Si = {  peak : Si }
      Si = { nsipm : [ q1, q2, ...] }

    Returns S2Si where Si = { nsipm : sum([q1, q2, ...])}
"""
    return { peak_no : integrate_sipm_charges_in_peak_as_dict(peak)
             for (peak_no, peak) in S2Si.items() }


def select_si_slice(si, slice_no):
    # This is a temporary fix! The number of slices in the SiPM arrays
    # must match that of the PMT PMaps.
    return {sipm_no: (sipm[slice_no] if len(sipm) > slice_no else 0)
                      for sipm_no, sipm in si.items()}
