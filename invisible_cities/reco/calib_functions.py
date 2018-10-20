"""
Contains functions used in calibration.
"""

import numpy  as np
import tables as tb

from .. core.system_of_units_c import units
from .. core.core_functions    import in_range


def bin_waveforms(waveforms, bins):
    """
    A function to bin waveform data. Bins the current event
    data and adds it to the file level bin array.
    """
    def bin_waveform(wf):
        return np.histogram(wf, bins)[0]
    return np.apply_along_axis(bin_waveform, 1, waveforms)


def spaced_integrals(wfs, limits):
    """
    Function to get integrals in certain regions of buffers.
    Returns an array with the integrals between each point
    in the limits array.

    Parameters
    ----------
    wfs: np.ndarray with shape (n, m)
        Buffer waveforms
    limits: np.ndarray with shape (d,)
        Sequence of integration limits

    Returns
    -------
    integrals: np.ndarray with shape (n, d)
        Array with the sum of the waveform between consecutive
        values in the limits array
    """
    if min(limits) < 0 or max(limits) >= np.shape(wfs)[1]:
        raise ValueError(f"Invalid integral limits: {limits}."
                         f" Must be between 0 and {np.shape(wfs)[1]}")
    return np.add.reduceat(wfs, limits, axis=1)


def integral_limits(sample_width, n_integrals, integral_start, integral_width, period):
    """
    Define the integrals to be used for calibration.

    Parameters
    ----------
    sample_width   : float
        Sample width for sensors under study.
    n_integrals    : int
        Number of integrals per buffer
    integral_start : float
        Start in mus of first integral
    integral_width : float
        Width in mus of integrals
    period         : float
        Period in mus between integrals

    Returns
    -------
    corr: np.ndarray
            Correlated limits for integrals
    anti: np.ndarray
        Anticorrelated limits for integrals
    """
    f_int = int(np.floor(integral_start * units.mus / sample_width)) # Position in samples of start of first integral
    w_int = int(np.ceil (integral_width * units.mus / sample_width)) # Width in samples
    p_int = int(np.ceil (        period * units.mus / sample_width)) # Period of repetition in samples
    e_int = f_int + w_int                                            # End first integral

    corr = np.column_stack((f_int + np.arange(0, n_integrals) * p_int,
                            e_int + np.arange(0, n_integrals) * p_int)).flatten()
    anti = corr - w_int - int(2 * units.mus // sample_width)

    return corr, anti


def filter_limits(limits, buffer_length):
    """
    Check that no part of the defined limits falls outside
    the buffer and removes limits if necessary.

    Parameters
    ----------
    limits        : np.ndarray
        Array of integral limits
    buffer_length : int
        Number of waveform samples

    Returns
    -------
    f_limits : np.ndarray
        Filtered limits
    """
    within_buffer = in_range(limits, 0, buffer_length + 1)
    half          = len(limits) // 2

    #check if odd falses at start or end
    n_false_first_half  = np.count_nonzero(~within_buffer[:half])
    n_false_second_half = np.count_nonzero(~within_buffer[half:])

    if n_false_first_half  % 2: within_buffer[  n_false_first_half     ] = False
    if n_false_second_half % 2: within_buffer[- n_false_second_half - 1] = False

    return limits[within_buffer]


def valid_integral_limits(sample_width, n_integrals, integral_start, integral_width, period, buffer_length):
    corr, anti = integral_limits(sample_width, n_integrals, integral_start, integral_width, period)
    return (filter_limits(corr, buffer_length),
            filter_limits(anti, buffer_length))

def copy_sensor_table(h5in, h5out):
    # Copy sensor table if exists (needed for Non DB calibrations)

    with tb.open_file(h5in) as dIn:
        if 'Sensors' not in dIn.root:
            return
        group    = h5out.create_group(h5out.root, "Sensors")

        if 'DataPMT' in dIn.root.Sensors:
            datapmt  = dIn.root.Sensors.DataPMT
            datapmt.copy(newparent=group)

        if 'DataSiPM' in dIn.root.Sensors:
            datasipm = dIn.root.Sensors.DataSiPM
            datasipm.copy(newparent=group)
