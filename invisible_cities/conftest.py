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

from invisible_cities.reco.pmaps_functions import (df_to_pmaps_dict,
                                                   df_to_s2si_dict,
                                                   read_pmaps,
                                                   load_pmaps)
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


@pytest.fixture(scope='session')
def KrMC_pmaps(ICDIR):
    # Input file was produced to contain exactly 15 S1 and 50 S2.
    test_file = ICDIR + "/database/test_data/KrMC_pmaps.h5"
    S1_evts   = list(filter(lambda x: x not in [48, 50], range(23, 52)))
    S2_evts   = list(range(31, 41))
    S2Si_evts = list(range(31, 41))
    s1s, s2s, s2sis = read_pmaps(test_file)
    s1, s2, s2si    = load_pmaps(test_file)
    return (test_file,
            (s1s, s2s, s2sis),
            (S1_evts, S2_evts, S2Si_evts),
            (s1, s2, s2si))



@pytest.fixture(scope='session')
def Kr_dst_data(ICDIR):
    data = {}
    data["event"] = np.array   ([  1] * 3 + [2  ] + [6   ] * 2)
    data["time" ] = np.array   ([1e7] * 3 + [2e7] + [3e7 ] * 2)
    data["peak" ] = np.array   ([0, 1, 2] + [0  ] + [0, 1]    )
    data["nS2"  ] = np.array   ([  3] * 3 + [1  ] + [2   ] * 2)
    data["S1w"  ] = np.array   ([100] * 3 + [160] + [180 ] * 2)
    data["S1h"  ] = np.array   ([ 10] * 3 + [ 50] + [ 60 ] * 2)
    data["S1e"  ] = np.array   ([  5] * 3 + [  2] + [  8 ] * 2)
    data["S1t"  ] = np.array   ([100] * 3 + [200] + [700 ] * 2)

    data["S2w"  ] = np.linspace( 10,  17, 6)
    data["S2h"  ] = np.linspace(150, 850, 6)
    data["S2e"  ] = np.linspace(1e3, 8e3, 6)
    data["S2q"  ] = np.linspace(  0, 700, 6)
    data["S2t"  ] = np.linspace(200, 900, 6)

    data["Nsipm"] = np.arange  (  1,   7, 1)
    data["DT"   ] = np.linspace(100, 107, 6)
    data["Z"    ] = np.linspace(200, 207, 6)
    data["X"    ] = np.linspace(-55, +55, 6)
    data["Y"    ] = np.linspace(-95, +95, 6)
    data["R"    ] = (data["X"]**2 + data["Y"]**2)**0.5
    data["Phi"  ] = np.arctan2 (data["Y"], data["X"])
    data["Xrms" ] = np.linspace( 10,  70, 6)
    data["Yrms" ] = np.linspace( 20,  90, 6)

    cols = ("event", "time", "peak", "nS2",
            "S1w", "S1h", "S1e", "S1t", "S2w", "S2h", "S2e", "S2q", "S2t",
            "Nsipm", "DT", "Z", "X", "Y", "R", "Phi", "Xrms", "Yrms")

    df = DataFrame(data, columns = cols)

    return (ICDIR + "/database/test_data/Kr_dst.h5", "DST", "data"), df


