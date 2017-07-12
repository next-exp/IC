import numpy as np

from . random_sampling  import NoiseSampler
from ..database.load_db import DataSiPM

def test_noise_sampler_masked_sensors():
    run_number = 0

    sampler = NoiseSampler(run_number, 1000, False)
    sample  = sampler.Sample()

    datasipm = DataSiPM(run_number)
    masked_sensors = datasipm[datasipm.Active==0].index.values
    assert not np.any(sample[masked_sensors])
