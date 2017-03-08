import pytest
import os

@pytest.fixture(scope = 'session')
def ICDIR():
    return os.environ['ICDIR']

@pytest.fixture(scope = 'session')
def irene_diomira_chain_tmpdir(tmpdir_factory):
    return tmpdir_factory.mktemp('irene_diomira_tests')

@pytest.fixture(scope = 'session')
def config_tmpdir(tmpdir_factory):
    return tmpdir_factory.mktemp('configure_tests')

@pytest.fixture(scope  = 'session',
                params = ['electrons_40keV_z250_RWF.h5',
                          'electrons_511keV_z250_RWF.h5',
                          'electrons_1250keV_z250_RWF.h5',
                          'electrons_2500keV_z250_RWF.h5'])
def electron_RWF_file(request, ICDIR):
    return os.path.join(ICDIR,
                        'database/test_data',
                        request.param)



from invisible_cities.reco.pmaps_functions import df_to_pmaps_dict, df_to_s2si_dict
from pandas import DataFrame, Series
import numpy as np

@pytest.fixture(scope='session')
def s12_dataframe_converted():
    evs  = [   0,     0,     0,     0,     0,      3,     3]
    peak = [   0,     0,     1,     1,     1,      0,     0]
    time = [1000., 1025., 2050., 2075., 2100., 5000., 5025.]
    ene  = [123.4, 140.8, 730.0, 734.2, 732.7, 400.0, 420.3]
    df = DataFrame.from_dict(dict(
        event  = Series(evs , dtype=np.  int32),
        evtDaq = evs,
        peak   = Series(peak, dtype=np.   int8),
        time   = Series(time, dtype=np.float32),
        ene    = Series(ene , dtype=np.float32),
    ))
    return df_to_pmaps_dict(df), df

@pytest.fixture(scope='session')
def s2si_dataframe_converted():
    evs  = [    0,     0,     0,     0,     0,     3,     3]
    peak = [    0,     0,     1,     1,     1,     0,     0]
    sipm = [    1,     2,     3,     4,     5,     5,     6]
    samp = [    0,     2,     0,     1,     2,     3,     4]
    ene  = [123.4, 140.8, 730.0, 734.2, 732.7, 400.0, 420.3]
    df = DataFrame.from_dict(dict(
        event   = Series(evs , dtype=np.int32),
        evtDaq  = evs,
        peak    = Series(peak, dtype=np.   int8),
        nsipm   = Series(sipm, dtype=np.  int16),
        nsample = Series(samp, dtype=np.  int16),
        ene     = Series(ene , dtype=np.float32),
    ))
    return df_to_s2si_dict(df), df

