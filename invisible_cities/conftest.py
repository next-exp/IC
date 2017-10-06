import os
import pytest
from pandas import DataFrame, Series
import numpy as np

# from . io.pmap_io   import df_to_pmaps_dict
from . io.pmap_io   import df_to_s1_dict
from . io.pmap_io   import df_to_s2_dict
from . io.pmap_io   import df_to_s2si_dict
from . io.pmap_io   import read_pmaps
from . io.pmap_io   import load_pmaps
from . io.pmap_io   import load_ipmt_pmaps
from . io.pmap_io   import load_pmaps_with_ipmt
from . io.pmap_io   import ipmt_pmap_writer
from . io.hits_io   import load_hits
from . io.hits_io   import load_hits_skipping_NN

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


@pytest.fixture(scope  = 'session',
                params = ['electrons_40keV_z250_MCRD.h5'])
def electron_MCRD_file(request, ICDIR):
    return os.path.join(ICDIR,
                        'database/test_data',
                        request.param)

@pytest.fixture(scope  = 'session',
                params = ['dst_NEXT_v1_00_05_Tl_ACTIVE_140_0_7bar_PMP_2.h5'])
def thallium_DST_file(request, ICDIR):
    return os.path.join(ICDIR,
                        'database/test_data',
                        request.param)

@pytest.fixture(scope='session')
def mc_all_hits_data(thallium_DST_file):
    number_of_hits = 18
    evt_number = 5600000
    efile = thallium_DST_file
    return efile, number_of_hits, evt_number

@pytest.fixture(scope='session')
def mc_particle_and_hits_data(electron_MCRD_file):
    X = [-0.10718990862369537, -0.16415221989154816, -0.18664051592350006, -0.19431403279304504]
    Y = [-0.01200979482382536, -0.07335199415683746, -0.09059777110815048, -0.09717071801424026,]
    Z = [25.12295150756836, 25.140811920166016, 25.11968994140625, 25.115009307861328]
    E = [0.006218845956027508, 0.014433029107749462, 0.010182539001107216, 0.009165585972368717]
    t = [0.0009834024822339416, 0.0018070531077682972, 0.002247565658763051, 0.002446305239573121]
    name = b'e-'
    pdg = 11
    vi = np.array([  3.06151588e-14,   0.00000000e+00,  2.51000004e+01])
    vf = np.array([ -0.19431403,  -0.09717072,  25.11500931])
    p =  np.array([-0.20033967, -0.0224465,   0.0428962 ])
    Ep = 0.03999999910593033
    nhits = 4

    efile = electron_MCRD_file
    return efile, name, pdg, vi, vf, p, Ep, nhits, X, Y, Z, E, t


@pytest.fixture(scope='session')
def s1_dataframe_converted():
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
    return df_to_s1_dict(df), df


@pytest.fixture(scope='session')
def s2_dataframe_converted():
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

    return df_to_s2_dict(df), df


@pytest.fixture(scope='session')
def s2si_dataframe_converted():
    evs  = [  0,   0,  0,  0,   0,  0,  0,  3,  3,  3,  3]
    peak = [  0,   0,  0,  0,   1,  1,  1,  0,  0,  0,  0]
    sipm = [  0,   0,  1,  1,   3,  3,  3, 10, 10, 15, 15]
    ene  = [1.5, 2.5, 15, 25, 5.5, 10, 20,  8,  9, 17, 18]

    dfs2si = DataFrame.from_dict(dict(
        event   = Series(evs , dtype=np.int32),
        evtDaq  = evs,
        peak    = Series(peak, dtype=np.   int8),
        nsipm   = Series(sipm, dtype=np.  int16),
        ene     = Series(ene , dtype=np.float32),
    ))
    _,  dfs2 =  s2_dataframe_converted()
    return df_to_s2si_dict(dfs2, dfs2si), dfs2si


@pytest.fixture(scope='session')
def KrMC_pmaps(ICDIR):
    # Input file was produced to contain exactly 15 S1 and 50 S2.
    test_file = ICDIR + "/database/test_data/KrMC_pmaps.h5"
    S1_evts   = list(filter(lambda x: x not in [48, 50], range(23, 52)))
    S2_evts   = list(range(31, 41))
    S2Si_evts = list(range(31, 41))
    s1t, s2t, s2sit = read_pmaps(test_file)
    s1, s2, s2si    = load_pmaps(test_file)
    return (test_file,
            (s1t, s2t, s2sit),
            (S1_evts, S2_evts, S2Si_evts),
            (s1, s2, s2si))


@pytest.fixture(scope='session')
def TlMC_hits(ICDIR):
    # Input file was produced to contain exactly 15 S1 and 50 S2.
    hits_file_name = ICDIR + "/database/test_data/dst_NEXT_v1_00_05_Tl_ACTIVE_100_0_7bar_DST_10.h5"
    hits = load_hits(hits_file_name)
    return hits


@pytest.fixture(scope='session')
def TlMC_hits_skipping_NN(ICDIR):
    # Input file was produced to contain exactly 15 S1 and 50 S2.
    hits_file_name = ICDIR + "/database/test_data/dst_NEXT_v1_00_05_Tl_ACTIVE_100_0_7bar_DST_10.h5"
    hits = load_hits_skipping_NN(hits_file_name)
    return hits


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


@pytest.fixture(scope='session')
def corr_toy_data(ICDIR):
    x = np.arange( 100, 200)
    y = np.arange(-200,   0)
    E = np.arange( 1e4, 1e4 + x.size*y.size).reshape(x.size, y.size)
    U = np.arange( 1e2, 1e2 + x.size*y.size).reshape(x.size, y.size)
    N = np.ones_like(U)

    corr_filename = os.path.join(ICDIR, "database/test_data/toy_corr.h5")
    return corr_filename, (x, y, E, U, N)


@pytest.fixture(scope='session')
def hits_toy_data(ICDIR):
    npeak = np.array   ([0]*25 + [1]*30 + [2]*35 + [3]*10)
    nsipm = np.arange  (1000, 1100)
    x     = np.linspace( 150,  250, 100)
    y     = np.linspace(-280, -180, 100)
    xrms  = np.linspace(   1,   80, 100)
    yrms  = np.linspace(   2,   40, 100)
    z     = np.linspace(   0,  515, 100)
    q     = np.linspace( 1e3,  1e3, 100)
    e     = np.linspace( 2e3,  1e4, 100)

    hits_filename = os.path.join(ICDIR, "database/test_data/toy_hits.h5")
    return hits_filename, (npeak, nsipm, x, y, xrms, yrms, z, q, e)


@pytest.fixture(scope='session')
def Kr_MC_4446_load_s1_s2_s2si(ICDIR):
    ipmt_pmap_path = ICDIR + 'database/test_data/Kr_MC_ipmt_pmaps_5evt.h5'
    Kr_MC_4446_load_pmaps = load_pmaps(ipmt_pmap_path)
    return Kr_MC_4446_load_pmaps


@pytest.fixture(scope='session')
def Kr_MC_4446_load_pmaps_with_ipmt(ICDIR):
    ipmt_pmap_path = ICDIR + 'database/test_data/Kr_MC_ipmt_pmaps_5evt.h5'
    Kr_MC_4446_load_pmaps_with_ipmt = load_pmaps_with_ipmt(ipmt_pmap_path)
    return Kr_MC_4446_load_pmaps_with_ipmt


@pytest.fixture(scope='session')
def Kr_pmaps_run4628(ICDIR):
    filename = ICDIR + 'database/test_data/Kr_pmaps_run4628.h5'
    return filename
