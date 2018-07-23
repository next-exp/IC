import os
import pytest
import numpy  as np
import tables as tb

from pandas      import DataFrame
from collections import namedtuple

from . core.system_of_units_c import units
from . evm . pmaps_test       import pmaps
from . io  . pmaps_io         import load_pmaps_as_df
from . io  . pmaps_io         import load_pmaps
from . io  . pmaps_io         import pmap_writer
from . io  .   dst_io         import load_dst
from . io  .  hits_io         import load_hits
from . io  .  hits_io         import load_hits_skipping_NN
from . io  . mcinfo_io        import load_mchits


tbl_data = namedtuple('tbl_data', 'filename group node')
dst_data = namedtuple('dst_data', 'file_info config read true')
pmp_dfs  = namedtuple('pmp_dfs' , 's1 s2 si, s1pmt, s2pmt')
pmp_data = namedtuple('pmp_data', 's1 s2 si')
mcs_data = namedtuple('mcs_data', 'pmap hdst')


@pytest.fixture(scope = 'session')
def ICDIR():
    return os.environ['ICDIR']


@pytest.fixture(scope = 'session')
def ICDATADIR(ICDIR):
    return os.path.join(ICDIR, "database/test_data/")


@pytest.fixture(scope = 'session')
def irene_diomira_chain_tmpdir(tmpdir_factory):
    return tmpdir_factory.mktemp('irene_diomira_tests')


@pytest.fixture(scope = 'session')
def config_tmpdir(tmpdir_factory):
    return tmpdir_factory.mktemp('configure_tests')


@pytest.fixture(scope = 'session')
def output_tmpdir(tmpdir_factory):
    return tmpdir_factory.mktemp('output_files')


@pytest.fixture(scope='session')
def example_blr_wfs_filename(ICDATADIR):
    return os.path.join(ICDATADIR, "blr_examples.h5")


@pytest.fixture(scope  = 'session',
                params = ['electrons_40keV_z250_RWF.h5',
                          'electrons_511keV_z250_RWF.h5',
                          'electrons_1250keV_z250_RWF.h5',
                          'electrons_2500keV_z250_RWF.h5'])
def electron_RWF_file(request, ICDATADIR):
    return os.path.join(ICDATADIR, request.param)


@pytest.fixture(scope  = 'session',
                params = ['electrons_40keV_z250_MCRD.h5'])
def electron_MCRD_file(request, ICDATADIR):
    return os.path.join(ICDATADIR, request.param)


@pytest.fixture(scope  = 'session',
                params = ['Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts.MCRD.h5'])
def krypton_MCRD_file(request, ICDATADIR):
    return os.path.join(ICDATADIR, request.param)


@pytest.fixture(scope='session')
def mc_all_hits_data(krypton_MCRD_file):
    number_of_hits = 8
    evt_number     = 2
    efile          = krypton_MCRD_file
    return efile, number_of_hits, evt_number

@pytest.fixture(scope='session')
def mc_particle_and_hits_nexus_data(ICDATADIR):
    X     = [ -4.37347144e-02,  -2.50248108e-02, -3.25887166e-02, -3.25617939e-02 ]
    Y     = [ -1.37645766e-01,  -1.67959690e-01, -1.80502057e-01, -1.80206522e-01 ]
    Z     = [  2.49938721e+02,   2.49911240e+02,  2.49915543e+02,  2.49912308e+02 ]
    E     = [ 0.02225098,  0.00891293,  0.00582698,  0.0030091 ]
    t     = [ 0.00139908,  0.00198319,  0.00226054,  0.00236114 ]

    vi    = np.array([        0.,          0.,          250.,          0.])
    vf    = np.array([-3.25617939e-02,  -1.80206522e-01,   2.49912308e+02,   2.36114440e-03])

    p     = np.array([-0.05745485, -0.18082699, -0.08050126])

    efile = os.path.join(ICDATADIR, 'electrons_40keV_z250_MCRD.h5')
    Ep    = 0.04
    name  = 'e-'
    nhits = 4

    return efile, name, vi, vf, p, Ep, nhits, X, Y, Z, E, t


@pytest.fixture(scope='session')
def mc_sensors_nexus_data(ICDATADIR):
    pmt0_first       = (0, 1)
    pmt0_last        = (670, 1)
    pmt0_tot_samples = 54

    sipm_id = 13016
    sipm    = [(63, 3), (64, 2), (65, 1)]

    efile = os.path.join(ICDATADIR, 'Kr83_full_nexus_v5_03_01_ACTIVE_7bar_1evt.sim.h5')

    return efile, pmt0_first, pmt0_last, pmt0_tot_samples, sipm_id, sipm


def _get_pmaps_dict_and_event_numbers(filename):
    dict_pmaps = load_pmaps(filename)

    def peak_contains_sipms(s2 ): return bool(s2.sipms.ids.size)
    def  evt_contains_sipms(evt): return any (map(peak_contains_sipms, dict_pmaps[evt].s2s))
    def  evt_contains_s1s  (evt): return bool(dict_pmaps[evt].s1s)
    def  evt_contains_s2s  (evt): return bool(dict_pmaps[evt].s2s)

    s1_events  = tuple(filter(evt_contains_s1s  , dict_pmaps))
    s2_events  = tuple(filter(evt_contains_s2s  , dict_pmaps))
    si_events  = tuple(filter(evt_contains_sipms, dict_pmaps))

    return dict_pmaps, pmp_data(s1_events, s2_events, si_events)


@pytest.fixture(scope='session')
def KrMC_pmaps_filename(ICDATADIR):
    test_file = "Kr83_nexus_v5_03_00_ACTIVE_7bar_10evts_PMP.h5"
    test_file = os.path.join(ICDATADIR, test_file)
    return test_file


@pytest.fixture(scope='session')
def KrMC_pmaps_without_ipmt_filename(ICDATADIR):
    test_file = "dst_NEXT_v1_00_05_Kr_ACTIVE_0_0_7bar_PMP_10evt_new_wo_ipmt.h5"
    test_file = os.path.join(ICDATADIR, test_file)
    return test_file


@pytest.fixture(scope='session')
def KrMC_pmaps_dfs(KrMC_pmaps_filename):
    s1df, s2df, sidf, s1pmtdf, s2pmtdf = load_pmaps_as_df(KrMC_pmaps_filename)
    return pmp_dfs(s1df, s2df, sidf, s1pmtdf, s2pmtdf)


@pytest.fixture(scope='session')
def KrMC_pmaps_without_ipmt_dfs(KrMC_pmaps_without_ipmt_filename):
    s1df, s2df, sidf, s1pmtdf, s2pmtdf = load_pmaps_as_df(KrMC_pmaps_without_ipmt_filename)
    return pmp_dfs(s1df, s2df, sidf, s1pmtdf, s2pmtdf)


@pytest.fixture(scope='session')
def KrMC_pmaps_dict(KrMC_pmaps_filename):
    dict_pmaps, evt_numbers = _get_pmaps_dict_and_event_numbers(KrMC_pmaps_filename)
    return dict_pmaps, evt_numbers


@pytest.fixture(scope='session')
def KrMC_pmaps_without_ipmt_dict(KrMC_pmaps_without_ipmt_filename):
    dict_pmaps, evt_numbers = _get_pmaps_dict_and_event_numbers(KrMC_pmaps_without_ipmt_filename)
    return dict_pmaps, evt_numbers


@pytest.fixture(scope='session')
def KrMC_pmaps_example(output_tmpdir):
    output_filename  = os.path.join(output_tmpdir, "test_pmap_file.h5")
    number_of_events = 3
    event_numbers    = np.random.choice(2 * number_of_events, size=number_of_events, replace=False)
    event_numbers    = sorted(event_numbers)
    pmt_ids          = np.arange(3) * 2
    pmap_generator   = pmaps(pmt_ids)
    true_pmaps       = [pmap_generator.example()[1] for _ in range(number_of_events)]

    with tb.open_file(output_filename, "w") as output_file:
        write = pmap_writer(output_file)
        list(map(write, true_pmaps, event_numbers))
    return output_filename, dict(zip(event_numbers, true_pmaps))


@pytest.fixture(scope='session')
def KrMC_kdst(ICDATADIR):
    test_file = "Kr83_nexus_v5_03_00_ACTIVE_7bar_10evts_KDST.h5"
    test_file = os.path.join(ICDATADIR, test_file)

    wrong_file = "kdst_5881_map_lt.h5"
    wrong_file = os.path.join(ICDATADIR, wrong_file)

    group = "DST"
    node  = "Events"

    configuration = dict(run_number  =  -4734,
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

    event    = [3, 4, 5, 9]
    time     = [  0,  0,  0, 0]
    s1_peak  = [  0,  0,  0, 0]
    nS1      = [  1,  1,  1, 1]
    s2_peak  = [  0,  0,  0, 0]
    nS2      = [  1,  1,  1, 1]

    S1w      = [   175.,       125.,        150.,       125.]
    S1h      = [     1.25865412, 1.4069742,   1.92166746, 0.90697569]
    S1e      = [     6.62232113, 5.84485674, 10.12352943, 4.90731287]
    S1t      = [100150.,    100125.,     100175.,    100200.]

    S2w      = [     6.44796875,  9.4400625,         6.4500625,      4.65923437]
    S2h      = [   650.96478271,  646.59228516,    677.49707031,   892.47644043]
    S2e      = [  2720.47900391,  2865.42333984,  2736.49707031,  3156.47216797]
    S2q      = [   586.34741211,  562.96801758,    535.01025391,   616.89324951]
    S2t      = [392470.5,      277481.625,      373514.59375,   263485.46875]
    Nsipm    = [13, 11, 12, 13]

    DT       = [ 292.3205    ,  177.356625  ,  273.33959375,  163.28546875]
    Z        = [ 584.641     ,  354.71325   ,  546.6791875,   326.5709375]
    Zrms     = [   1.49153943,    2.39387988,    1.41887842,    1.16239893]
    X        = [  45.59453442,  -55.71659201,  110.4270649 ,  111.86722506]
    Y        = [-136.06568541,  -91.44119328,   56.71870461,  -43.21484769]
    R        = [ 143.50168053,  107.07861809,  124.14164537,  119.92413896]
    Phi      = [  -1.24746373,   -2.11803776,    0.47449242,   -0.36864472]
    Xrms     = [   7.33091719,    6.30531617,    7.40606092,    7.24923445]
    Yrms     = [   7.23771972,    6.96642957,    7.60636805,    6.46853221]

    df_true = DataFrame({"event"  : event,
                         "time"   : time ,
                         "nS1"    : nS1  ,
                         "s1_peak": s1_peak ,
                         "nS2"    : nS2  ,
                         "s2_peak": s2_peak ,

                         "S1w"    : S1w,
                         "S1h"    : S1h,
                         "S1e"    : S1e,
                         "S1t"    : S1t,

                         "S2w"    : S2w,
                         "S2h"    : S2h,
                         "S2e"    : S2e,
                         "S2q"    : S2q,
                         "S2t"    : S2t,
                         "Nsipm"  : Nsipm,

                         "DT"     : DT,
                         "Z"      : Z,
                         "Zrms"   : Zrms,
                         "X"      : X,
                         "Y"      : Y,
                         "R"      : R,
                         "Phi"    : Phi,
                         "Xrms"   : Xrms,
                         "Yrms"   : Yrms})

    df_read = load_dst(test_file,
                       group = group,
                       node  = node)

    return dst_data(tbl_data(test_file, group, node),
                    configuration,
                    df_read,
                    df_true), tbl_data(wrong_file, group, node)


@pytest.fixture(scope='session')
def KrMC_hdst(ICDATADIR):
    test_file = "Kr83_nexus_v5_03_00_ACTIVE_7bar_10evts_HDST.h5"
    test_file = os.path.join(ICDATADIR, test_file)

    group = "RECO"
    node  = "Events"

    configuration = dict(run_number    =  -4734,
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

    event = [3] * 4 + [4] * 7 + [5] * 7 + [9] * 5
    time  = [0]  * 23
    peak  = [0]  * 23
    nsipm = [8, 9, 9, 8, 8, 9, 2, 7, 6, 8, 0, 7, 8, 2, 2, 8, 2, 5, 9, 1, 9, 2, 6]

    X     = [ 43.40294538,   46.47361995,   45.25429999,   45.23969681,
             -56.7783662 ,  -56.28419189,  -58.95283555,  -55.71850501,
             -54.84400966,  -54.52265854,    0.        ,  109.08928307,
             110.92084168,   95.        ,  110.60298616,  109.8447486 ,
             125.        ,  112.45975154,  113.05630365,   95.        ,
             113.06740096,   95.        ,  111.70545868]

    Y     = [-135.25268225, -135.77726939, -135.34085102, -136.3205644 ,
             -92.14774585,  -92.12689227,  -75.        ,  -92.01383524,
             -91.65213449,  -92.69232958,    0.        ,   56.55189485,
              55.76663453,   51.70236775,   75.        ,   55.9215318 ,
              57.13390328,   60.70777889,  -43.59306889,  -45.        ,
             -44.06857595,  -41.25946865,  -42.77815792]

    Xrms  = [ 6.48582136,  6.57229469,  6.75297825,  6.61735328,  6.12250653,
              6.16999461,  4.88911512,  6.16091263,  6.44104397,  6.31737004,
              0.        ,  6.17085002,  6.31399258,  0.        ,  4.9635076 ,
              6.20856522,  0.        ,  6.83960577,  6.5243526 ,  0.        ,
              6.0996107 ,  0.        ,  5.65269751]

    Yrms  = [ 6.89078637,  6.87813429,  6.74202862,  6.50750598,  5.84491542,
              6.34939153,  0.        ,  5.54236517,  4.71915793,  6.55253452,
              0.        ,  6.88855879,  6.72115655,  4.70127047,  0.        ,
              7.19014738,  4.09700984,  4.94965141,  6.02196165,  0.        ,
              6.11025905,  4.83877451,  5.2691056 ]

    Z     = [ 578.5915625 ,  581.949     ,  585.512875  ,  588.933875  ,
              350.6590625 ,  353.8185625 ,  353.8185625 ,  357.1928125 ,
              361.98175   ,  365.587125  ,  368.0826875 ,  542.4195    ,
              545.850875  ,  545.850875  ,  545.850875  ,  549.325875  ,
              549.325875  ,  552.6724375 ,  322.31784375,  322.31784375,
              325.630875  ,  325.630875  ,  328.81375   ]

    Q     = [   6.87585220e+01,   2.01821630e+02,   2.15834586e+02,
                7.03598893e+01,   8.58836713e+01,   2.14698522e+02,
                9.82612514e+00,   8.06264758e+01,   6.66430998e+01,
                7.62628095e+01,  -9.99999000e+05,   6.78010640e+01,
                2.39987609e+02,   1.70596542e+01,   9.05966711e+00,
                1.30985524e+02,   1.20733635e+01,   2.90001264e+01,
                1.81413164e+02,   5.71870232e+00,   3.13697742e+02,
                2.02247858e+01,   7.54415753e+01]

    E     = [  214.13093281,   971.6635704 ,  1162.77239227,   371.91184807,
               294.58552933,  1154.2397466 ,    52.82618663,   718.05093002,
               205.45460224,   403.17513466,    37.09131622,   310.53118706,
              1043.68731557,    74.19110003,    39.39978264,   986.06121642,
                90.88848298,   191.73790073,   643.57917544,    20.28760007,
              1651.86650255,   106.4994795 ,   734.23902893]

    Xpeak = [ 45.5945344 ,   45.5945344 ,   45.5945344 ,   45.5945344 ,
             -55.71659212,  -55.71659212,  -55.71659212,  -55.71659212,
             -55.71659212,  -55.71659212,  -55.71659212,  110.42706483,
             110.42706483,  110.42706483,  110.42706483,  110.42706483,
             110.42706483,  110.42706483,  111.86722516,  111.86722516,
             111.86722516,  111.86722516,  111.86722516]

    Ypeak = [-136.06568539, -136.06568539, -136.06568539, -136.06568539,
              -91.44119333,  -91.44119333,  -91.44119333,  -91.44119333,
              -91.44119333,  -91.44119333,  -91.44119333,   56.71870467,
               56.71870467,   56.71870467,   56.71870467,   56.71870467,
               56.71870467,   56.71870467,  -43.21484771,  -43.21484771,
              -43.21484771,  -43.21484771,  -43.21484771]

    df_true = DataFrame({"event": event,
                         "time" : time ,
                         "npeak": peak ,
                         "Xpeak": Xpeak,
                         "Ypeak": Ypeak,
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

    return dst_data(tbl_data(test_file, group, node),
                    configuration,
                    df_read,
                    df_true)


@pytest.fixture(scope='session')
def KrMC_true_hits(KrMC_pmaps_filename, KrMC_hdst):
    pmap_filename = KrMC_pmaps_filename
    hdst_filename = KrMC_hdst .file_info.filename
    pmap_mctracks = load_mchits(pmap_filename)
    hdst_mctracks = load_mchits(hdst_filename)
    return mcs_data(pmap_mctracks, hdst_mctracks)


@pytest.fixture(scope='session')
def TlMC_hits(ICDATADIR):
    # Input file was produced to contain exactly 15 S1 and 50 S2.
    hits_file_name = "dst_NEXT_v1_00_05_Tl_ACTIVE_100_0_7bar_DST_10.h5"
    hits_file_name = os.path.join(ICDATADIR, hits_file_name)
    hits = load_hits(hits_file_name)
    return hits


@pytest.fixture(scope='session')
def TlMC_hits_skipping_NN(ICDATADIR):
    # Input file was produced to contain exactly 15 S1 and 50 S2.
    hits_file_name = "dst_NEXT_v1_00_05_Tl_ACTIVE_100_0_7bar_DST_10.h5"
    hits_file_name = os.path.join(ICDATADIR, hits_file_name)
    hits = load_hits_skipping_NN(hits_file_name)
    return hits


@pytest.fixture(scope='session')
def corr_toy_data(ICDATADIR):
    x = np.arange( 100, 200)
    y = np.arange(-200,   0)
    E = np.arange( 1e4, 1e4 + x.size*y.size).reshape(x.size, y.size)
    U = np.arange( 1e2, 1e2 + x.size*y.size).reshape(x.size, y.size)
    N = np.ones_like(U)

    corr_filename = os.path.join(ICDATADIR, "toy_corr.h5")
    return corr_filename, (x, y, E, U, N)


@pytest.fixture(scope='session')
def hits_toy_data(ICDATADIR):
    npeak  = np.array   ([0]*25 + [1]*30 + [2]*35 + [3]*10)
    nsipm  = np.arange  (1000, 1100)
    x      = np.linspace( 150,  250, 100)
    y      = np.linspace(-280, -180, 100)
    xrms   = np.linspace(   1,   80, 100)
    yrms   = np.linspace(   2,   40, 100)
    z      = np.linspace(   0,  515, 100)
    q      = np.linspace( 1e3,  1e3, 100)
    e      = np.linspace( 2e3,  1e4, 100)
    x_peak = np.array([(x * e).sum() / e.sum()] * 100)
    y_peak = np.array([(y * e).sum() / e.sum()] * 100)

    hits_filename = os.path.join(ICDATADIR, "toy_hits.h5")
    return hits_filename, (npeak, nsipm, x, y, xrms, yrms, z, q, e, x_peak, y_peak)


@pytest.fixture(scope='session')
def Kr_pmaps_run4628_filename(ICDATADIR):
    filename = os.path.join(ICDATADIR, "Kr_pmaps_run4628.h5")
    return filename

@pytest.fixture(scope='session')
def voxels_toy_data(ICDATADIR):
    event = np.zeros(100)
    X     = np.linspace( 150,  250, 100)
    Y     = np.linspace(-280, -180, 100)
    Z     = np.linspace(   0,  100, 100)
    E     = np.linspace( 1e3,  1e3, 100)
    size  = np.reshape(np.repeat([10,10,10],100),(100,3))

    voxels_filename = os.path.join(ICDATADIR, "toy_voxels.h5")
    return voxels_filename, (event, X, Y, Z, E, size)
