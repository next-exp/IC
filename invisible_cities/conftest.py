import os
import pytest
import numpy as np

from pandas      import DataFrame
from pandas      import Series

from . io.pmap_io   import df_to_s1_dict
from . io.pmap_io   import df_to_s2_dict
from . io.pmap_io   import df_to_s2si_dict
from . io.pmap_io   import read_pmaps
from . io.pmap_io   import load_pmaps
from . io.pmap_io   import load_ipmt_pmaps
from . io.pmap_io   import load_pmaps_with_ipmt
from . io.pmap_io   import ipmt_pmap_writer
from . io. dst_io   import load_dst
from . io.hits_io   import load_hits
from . io.hits_io   import load_hits_skipping_NN

from . core.system_of_units_c import units


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
    test_file = os.path.join(ICDIR,
                             "database",
                             "test_data",
                             "dst_NEXT_v1_00_05_Kr_ACTIVE_0_0_7bar_PMP_10evt.h5")
    S1_evts   = [15, 17, 19, 25, 27]
    S2_evts   = [15, 17, 19, 21, 23, 25, 27, 29, 31, 33]
    S2Si_evts = [15, 17, 19, 21, 23, 25, 27, 29, 31, 33]
    s1t, s2t, s2sit = read_pmaps(test_file)
    s1, s2, s2si    = load_pmaps(test_file)
    return (test_file,
            (s1t, s2t, s2sit),
            (S1_evts, S2_evts, S2Si_evts),
            (s1, s2, s2si))


@pytest.fixture(scope='session')
def KrMC_kdst(ICDIR):
    test_file = os.path.join(ICDIR,
                             "database",
                             "test_data",
                             "dst_NEXT_v1_00_05_Kr_ACTIVE_0_0_7bar_KDST_10evt.h5")
    group = "DST"
    node  = "Events"

    configuration = dict(run_number  =  -4446,
                         drift_v     =      2 * units.mm / units.mus,
                         s1_nmin     =      1,
                         s1_nmax     =      1,
                         s1_emin     =      0 * units.pes,
                         s1_emax     =     30 * units.pes,
                         s1_wmin     =    100 * units.ns,
                         s1_wmax     =    500 * units.ns,
                         s1_hmin     =    0.5 * units.pes,
                         s1_hmax     =     10 * units.pes,
                         s1_ethr     =   0.37 * units.pes,
                         s2_nmin     =      1,
                         s2_nmax     =      2,
                         s2_emin     =    1e3 * units.pes,
                         s2_emax     =    1e8 * units.pes,
                         s2_wmin     =      1 * units.mus,
                         s2_wmax     =     20 * units.mus,
                         s2_hmin     =    500 * units.pes,
                         s2_hmax     =    1e5 * units.pes,
                         s2_ethr     =      1 * units.pes,
                         s2_nsipmmin =      2,
                         s2_nsipmmax =   1000)

    event = [ 15, 17, 19, 25, 27]
    time  = [  0,  0,  0,  0,  0]
    peak  = [  0,  0,  0,  0,  0]
    nS2   = [  1,  1,  1,  1,  1]

    S1w   = [   125         ,    250         ,    250         ,    200         ,    150         ]
    S1h   = [     1.56272364,      3.25255418,      2.44831991,      1.75402236,      1.12322664]
    S1e   = [     6.198125  ,     19.414172  ,     16.340692  ,     10.747040  ,      6.176510  ]
    S1t   = [100200         , 100150         , 100100         , 100100         , 100225         ]

    S2w   = [    11.2625    ,     11.2625    ,     11.3625    ,      8.6375    ,      8.5375    ]
    S2h   = [   885.675964  ,    881.007202  ,    851.110901  ,   1431.818237  ,   1970.844116  ]
    S2e   = [  4532.44612527,   5008.20850897,   4993.90345001,   5979.88975143,   7006.23293018]
    S2q   = [   263.91775739,    185.49846303,    184.47896039,    311.68750608,    385.77873647]
    S2t   = [528500         , 554500         , 625500         , 338500         , 283500         ]
    Nsipm = [     7         ,      5         ,      6         ,      7         ,      7         ]

    DT    = [ 428.300     , 454.350     ,  525.400    , 238.400     , 183.275     ]
    Z     = [ 856.6       , 908.7       , 1050.8      , 476.8       , 366.55      ]
    X     = [-177.29208558, 120.08634576,    2.6156617, 154.08992723,  77.7168822 ]
    Y     = [   4.565395  , 106.715016  ,  147.882146 ,  72.077534  , -77.904470  ]
    R     = [ 177.350857  , 160.651253  ,  147.905276 , 170.114304  , 110.040993  ]
    Phi   = [   3.11584764,   0.7265102 ,    1.5531107,   0.43752688,  -0.78660357]
    Xrms  = [   6.14794   ,   4.999254  ,    5.906495 ,   6.198707  ,   6.42701   ]
    Yrms  = [   6.802864  ,   5.797995  ,    6.467924 ,   5.600673  ,   5.670117  ]

    df_true = DataFrame({"event": event,
                         "time" : time ,
                         "peak" : peak ,
                         "nS2"  : nS2  ,

                         "S1w"  : S1w,
                         "S1h"  : S1h,
                         "S1e"  : S1e,
                         "S1t"  : S1t,

                         "S2w"  : S2w,
                         "S2h"  : S2h,
                         "S2e"  : S2e,
                         "S2q"  : S2q,
                         "S2t"  : S2t,
                         "Nsipm": Nsipm,

                         "DT"   : DT,
                         "Z"    : Z,
                         "X"    : X,
                         "Y"    : Y,
                         "R"    : R,
                         "Phi"  : Phi,
                         "Xrms" : Xrms,
                         "Yrms" : Yrms})

    df_read = load_dst(test_file,
                       group = group,
                       node  = node)

    return ((test_file, group, node),
            configuration,
            df_read,
            df_true)


@pytest.fixture(scope='session')
def KrMC_hdst(ICDIR):
    test_file = os.path.join(ICDIR,
                             "database",
                             "test_data",
                             "dst_NEXT_v1_00_05_Kr_ACTIVE_0_0_7bar_HDST_10evt.h5")
    group = "RECO"
    node  = "Events"

    configuration = dict(run_number    =  -4446,
                         rebin         =      2,
                         drift_v       =      2 * units.mm / units.mus,
                         s1_nmin       =      1,
                         s1_nmax       =      1,
                         s1_emin       =      0 * units.pes,
                         s1_emax       =     30 * units.pes,
                         s1_wmin       =    100 * units.ns,
                         s1_wmax       =    500 * units.ns,
                         s1_hmin       =    0.5 * units.pes,
                         s1_hmax       =     10 * units.pes,
                         s1_ethr       =   0.37 * units.pes,
                         s2_nmin       =      1,
                         s2_nmax       =      2,
                         s2_emin       =    1e3 * units.pes,
                         s2_emax       =    1e8 * units.pes,
                         s2_wmin       =      1 * units.mus,
                         s2_wmax       =     20 * units.mus,
                         s2_hmin       =    500 * units.pes,
                         s2_hmax       =    1e5 * units.pes,
                         s2_ethr       =      1 * units.pes,
                         s2_nsipmmin   =      2,
                         s2_nsipmmax   =   1000,
                         msipm         =      1,
                         qthr          =      2 * units.pes,
                         qlm           =      5 * units.pes,
                         lm_radius     =      0 * units.mm,
                         new_lm_radius =     15 * units.mm)

    event = [15] * 7 + [17] * 7 + [19] * 8 + [25] * 5 + [27] * 6
    time  = [0]  * 33
    peak  = [0]  * 33
    nsipm = [0, 6, 7, 7, 4, 0, 0,
             0, 4, 5, 5, 0, 0, 0,
             0, 6, 4, 1, 6, 3, 0, 0,
             3, 7, 7, 6, 0,
             0, 7, 5, 2, 6, 0]

    X     = [ 0.00000000e+00, -1.79940785e+02, -1.77493026e+02, -1.76512358e+02, -1.75334056e+02,
              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.19908346e+02,  1.20280265e+02,
              1.19897288e+02,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
              3.11653915e+00,  1.68025927e+00,  5.00000000e+00,  1.96339537e+00,  5.79493732e+00,
              0.00000000e+00,  0.00000000e+00,  1.56440365e+02,  1.54312276e+02,  1.54017158e+02,
              1.52760628e+02,  0.00000000e+00,  0.00000000e+00,  7.66086466e+01,  7.98235545e+01,
              6.50000000e+01,  7.75782656e+01,  0.00000000e+00]

    Y     = [ 0.00000000e+00,  4.86151945e+00,  5.64318878e+00,  4.57161366e+00,  8.34641810e-01,
              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.08774814e+02,  1.05106046e+02,
              1.07573831e+02,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
              1.47104004e+02,  1.50782271e+02,  1.35000000e+02,  1.47748654e+02,  1.48782929e+02,
              0.00000000e+00,  0.00000000e+00,  7.20626968e+01,  7.19827231e+01,  7.17844856e+01,
              7.35674548e+01,  0.00000000e+00,  0.00000000e+00, -7.83527723e+01, -7.78888272e+01,
             -7.75947914e+01, -7.69924054e+01,  0.00000000e+0] 

    Xrms  = [ 0.00000000e+00,  4.99964935e+00,  6.16285499e+00,  6.39776071e+00,  4.73166675e+00,
              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  4.99915988e+00,  4.99213896e+00,
              4.99894491e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
              6.67177274e+00,  4.70921743e+00,  0.00000000e+00,  5.96738923e+00,  7.84466590e+00,
              0.00000000e+00,  0.00000000e+00,  3.51126694e+00,  6.25898165e+00,  6.46279288e+00,
              5.39667925e+00,  0.00000000e+00,  0.00000000e+00,  6.39273276e+00,  4.99688573e+00,
              0.00000000e+00,  5.60389979e+00,  0.00000000e+00]

    Yrms  = [ 0.00000000e+00,  6.99179462e+00,  7.16712038e+00,  6.57909879e+00,  4.92984513e+00,
              0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  4.84756843e+00,  6.17415930e+00,
              5.11706448e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
              6.33785683e+00,  4.93842608e+00,  0.00000000e+00,  6.05342196e+00,  4.84961221e+00,
              0.00000000e+00,  0.00000000e+00,  4.55469889e+00,  5.70961845e+00,  5.52716097e+00,
              5.86810454e+00,  0.00000000e+00,  0.00000000e+00,  5.58579870e+00,  5.76180862e+00,
              4.38348850e+00,  5.76341550e+00,  0.00000000e+00]

    Z     = [ 8.47850000e+02,  8.51600000e+02,  8.55600000e+02,  8.59600000e+02,  8.63600000e+02,
              8.67600000e+02,  8.69625000e+02,  8.99950000e+02,  9.03700000e+02,  9.07700000e+02,
              9.11700000e+02,  9.15700000e+02,  9.19700000e+02,  9.21725000e+02,  1.04010000e+03,
              1.04380000e+03,  1.04780000e+03,  1.04780000e+03,  1.05180000e+03,  1.05580000e+03,
              1.05980000e+03,  1.06212500e+03,  4.69962500e+02,  4.73800000e+02,  4.77800000e+02,
              4.81800000e+02,  4.85600000e+02,  3.57875000e+02,  3.61550000e+02,  3.65550000e+02,
              3.65550000e+02,  3.69550000e+02,  3.73412500e+02]

    Q     = [-9.99999000e+05,  4.62482347e+01,  9.17025954e+01,  9.13810852e+01,  3.07611203e+01,
             -9.99999000e+05, -9.99999000e+05, -9.99999000e+05,  3.66840464e+01,  7.52080684e+01,
              5.74266894e+01, -9.99999000e+05, -9.99999000e+05, -9.99999000e+05, -9.99999000e+05,
              3.34944267e+01,  4.60593218e+01,  7.88968563e+00,  7.13625143e+01,  1.53295889e+01,
             -9.99999000e+05, -9.99999000e+05,  1.38965549e+01,  1.02213222e+02,  1.59417052e+02,
              3.42498608e+01, -9.99999000e+05, -9.99999000e+05,  1.06680569e+02,  1.88616520e+02,
              2.29496250e+01,  6.45566070e+01, -9.99999000e+05]

    E     = [ 6.40848265e+01,  4.54434708e+02,  1.47234515e+03,  1.58452844e+03,  7.91471527e+02,
              1.64559525e+02,  1.02194333e+00,  1.42587963e+02,  7.93532227e+02,  1.67873566e+03,
              1.56285858e+03,  6.89836090e+02,  1.39569141e+02,  1.08884954e+00,  1.18094801e+02,
              8.13183167e+02,  1.26648796e+03,  2.16941793e+02,  1.56585492e+03,  8.15261322e+02,
              1.84432705e+02,  1.36467876e+01,  1.67134626e+02,  1.54898758e+03,  2.80964453e+03,
              1.30383865e+03,  1.50284361e+02,  8.28040495e+01,  1.24097345e+03,  3.28156814e+03,
              3.99279757e+02,  1.81999011e+03,  1.81617418e+02]

    df_true = DataFrame({"event": event,
                         "time" : time ,
                         "npeak": peak ,
                         "nsipm": nsipm,
                         "X"    : X,
                         "Y"    : Y,
                         "Xrms" : Xrms,
                         "Yrms" : Yrms,
                         "Z"    : Z,
                         "Q"    : Q,
                         "E"    : E})

    df_read = load_dst(test_file,
                       group = group,
                       node  = node)

    return ((test_file, group, node),
            configuration,
            df_read,
            df_true)



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
