import os
import numpy as np

from typing import Callable

from . light_tables import create_lighttable_function

from . simulate_s1 import compute_scintillation_photons
from . simulate_s1 import compute_s1_pes_at_pmts
from . simulate_s1 import generate_s1_times_from_pes


def histogram_s1_times(s1_times      : list,
                       buffer_length : float,
                       bin_width     : float,
                       start_time    : float=0)->np.ndarray:
    """
    s1_times are histogramed in a waveform.
    Parameters
        :s1_times: list
            output of generate_s1_times_from_pes, list of size equal to the
            number of pmts. Each element is an array with the times of the S1
        :buffer_length: float
            waveform buffer lenght in time units
        :bin_width: float
            waveform bin width in time units
        :start_time:
            waveform time of first bin edge in time units
    Returns:
        :wfs: np.ndarray
            waveforms with start_time, buffer_length and bin_width
    """
    bins = np.arange(start_time, start_time + buffer_length + bin_width, bin_width)
    wfs  = np.stack([np.histogram(times, bins=bins)[0] for times in s1_times])
    return wfs


def s1_waveforms_creator(s1_lighttable   : str,
                         ws              : float,
                         wf_pmt_bin_width: float)->Callable:
    """
    Returns a function that creates the S1 waveforms from the hits (x, y, z, time, energy),
    with waveform parameters tmin (start absolute time of the waveform) and buffer_length.
    Parameters:
        :s1_lighttable: str
            the s1 lighttable filename, it can contain env variables
        :ws:
            the inverse scintillation yield
        :wf_pmt_bin_width:
            the waveform bin width that is returned by :create_s1_waveforms_from_hits:
    Returns:
        :create_s1_waveforms_from_hits: function
            see function docstring
    """
    s1_lt = create_lighttable_function(os.path.expandvars(s1_lighttable))

    def create_s1_waveforms_from_hits(x, y, z, time, energy, tmin, buffer_length):
        """
        Computes the s1 waveform from hits
        Parameters:
            :x, y, z, time, energy: np.ndarray
                event track hits
            :tmin: float
                waveform start time
            :buffer_length: float
                waveform buffer_length, waveform will range from tmin to
                tmin + buffer_length
        """
        s1_photons     = compute_scintillation_photons(energy, ws)
        s1_pes_at_pmts = compute_s1_pes_at_pmts(x, y, z, s1_photons, s1_lt)
        s1times = generate_s1_times_from_pes(s1_pes_at_pmts, time)
        s1_wfs  = histogram_s1_times(s1times, buffer_length, wf_pmt_bin_width, tmin)
        return s1_wfs

    return create_s1_waveforms_from_hits
