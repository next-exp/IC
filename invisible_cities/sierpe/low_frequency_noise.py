""" Classes and functions describing the oscillation in pmt
baselines.
AL, November 2017.
"""


import numpy as np

from functools import partial

from .. database import load_db as DB


def frequency_contribution(rfrequency, magnitude, phase, time_bins):

    return 2 * magnitude * np.cos(rfrequency * time_bins + phase)


def low_frequency_noise(run_number, buffer_length, buffer_bin_width=25e-9):

    FE_mapping, FE_data = DB.PMTLowFrequencyNoise(run_number)

    ## Need to protect for old runs where PMT indx != sensorID
    sensor_id     = DB.DataPMT(run_number).SensorID.values

    n_febox       = FE_data.shape[1] - 1
    n_frequencies = FE_data.shape[0]

    times = buffer_bin_width * np.arange(buffer_length)
    freq_contribution = partial(frequency_contribution, time_bins=times)

    ## Randomise frequencies
    frequency_low  = FE_data[:, 0] - np.diff(FE_data[:, 0])[0] / 2
    frequency_high = FE_data[:, 0] + np.diff(FE_data[:, 0])[0] / 2
    rot_frequencies   = 2 * np.pi * np.random.uniform(frequency_low, frequency_high)
    rot_frequencies   = np.tile(rot_frequencies, n_febox)

    ## Randomise magnitudes and phases
    magnitudes = np.array(tuple(map(np.random.normal    ,
                                    FE_data[:, 1:]      ,
                                    FE_data[:, 1:] * 0.5)))
    ## Reshape to ease mapping
    magnitudes = magnitudes.T.reshape(magnitudes.size,)

    phases     = np.random.uniform(-np.pi, np.pi, magnitudes.shape)

    noise = np.array(tuple(map(freq_contribution,
                               rot_frequencies  ,
                               magnitudes       ,
                               phases           )))
    noise = noise.reshape(n_febox, n_frequencies, buffer_length).sum(axis=1)

    def get_low_frequency_noise(indx_pmt):
        """ Returns the appropriate vector """

        febox = FE_mapping.FEBox[FE_mapping.SensorID==sensor_id[indx_pmt]].values[0]
        return noise[febox]

    return get_low_frequency_noise
