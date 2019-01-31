""" Classes and functions describing the oscillation in pmt
baselines.
AL, November 2017.
"""


import numpy as np

from functools import partial

from .. database import load_db as DB


def frequency_contribution(rfrequency, magnitude, phase, time_bins):
    """
    Returns the contribution over the whole buffer of
    a frequency, magnitude, phase point.
    
    Parameters
    ----------
    rfrequency : float, rotational frequency
    magnitude  : float, magnitude of oscillation
    phase      : float, phase of oscillation
    """
    return 2 * magnitude * np.cos(rfrequency * time_bins + phase)


def buffer_and_limits(buffer_length, buffer_bin_width, FE_data):
    """
    Makes the basic definitions which are common
    to all calls to the low_frequency_noise function.

    Parameters
    ----------
    buffer_length    : length of buffer to be simulated in no. samples
    buffer_bin_width : sample width in buffer
    FE_data          : np.ndarray with the frequencies to be simulated

    Returns
    -------
    partial wrapped frequency_contribution function for the time bins needed
    lower and upper limits for the frequency bins to be simulated.

    """
    times = buffer_bin_width * np.arange(buffer_length)
    freq_contribution = partial(frequency_contribution, time_bins=times)

    frequency_low  = FE_data - np.diff(FE_data)[0] / 2
    frequency_high = FE_data + np.diff(FE_data)[0] / 2

    return freq_contribution, frequency_low, frequency_high


def low_frequency_noise(detector_db, run_number, buffer_length, buffer_bin_width=25e-9):
    """
    Randomises frequencies, magnitudes and phases and
    returns a function that can be used to get the
    simulated low frequency noise for a particular PMT
    """

    FE_mapping, FE_data = DB.PMTLowFrequencyNoise(detector_db, run_number)

    ## Need to protect for old runs where PMT indx != sensorID
    sens_id       = DB.DataPMT(detector_db, run_number).SensorID.values

    n_febox       = FE_data.shape[1] - 1
    n_frequencies = FE_data.shape[0]

    freq_contrib, freq_low, freq_high = buffer_and_limits(buffer_length,
                                                          buffer_bin_width,
                                                          FE_data[:, 0])

    ## Randomise frequencies
    rot_frequencies = 2 * np.pi * np.random.uniform(freq_low, freq_high)
    rot_frequencies = np.tile(rot_frequencies, n_febox)

    ## Randomise magnitudes and phases. mag_rms ~ 0.5 * mag_mean
    magnitudes = np.array(tuple(map(np.random.normal    ,
                                    FE_data[:, 1:]      ,
                                    FE_data[:, 1:] * 0.5)))
    ## Reshape to ease mapping
    magnitudes = magnitudes.T.reshape(magnitudes.size,)

    phases     = np.random.uniform(-np.pi, np.pi, magnitudes.shape)

    noise = np.array(tuple(map(freq_contrib   ,
                               rot_frequencies,
                               magnitudes     ,
                               phases         )))
    noise = noise.reshape(n_febox, n_frequencies, buffer_length).sum(axis=1)

    def get_low_frequency_noise(indx_pmt):
        """ Returns the appropriate vector """

        febox = FE_mapping.FEBox[FE_mapping.SensorID==sens_id[indx_pmt]].values
        return noise[febox[0]]

    return get_low_frequency_noise
