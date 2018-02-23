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
from . io  .mchits_io         import load_mchits_nexus


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
                params = ['kr.3evts.MCRD.h5'])
def krypton_DST_file(request, ICDATADIR):
    return os.path.join(ICDATADIR, request.param)


@pytest.fixture(scope='session')
def mc_all_hits_data(krypton_DST_file):
    number_of_hits = 7
    evt_number = 2
    efile = krypton_DST_file
    return efile, number_of_hits, evt_number


#@pytest.fixture(scope='session')
#def mc_particle_and_hits_data(electron_MCRD_file):
#    X     = [ -0.10718990862369537,  -0.16415221989154816, -0.18664051592350006, -0.19431403279304504]
#    Y     = [ -0.01200979482382536,  -0.07335199415683746, -0.09059777110815048, -0.09717071801424026]
#    Z     = [    25.12295150756836,    25.140811920166016,    25.11968994140625,   25.115009307861328]
#    E     = [ 0.006218845956027508,  0.014433029107749462, 0.010182539001107216, 0.009165585972368717]
#    t     = [0.0009834024822339416, 0.0018070531077682972, 0.002247565658763051, 0.002446305239573121]

#    vi    = np.array([3.06151588e-14,         0.0,  25.1000004])
#    vf    = np.array([   -0.19431403, -0.09717072, 25.11500931])
#    p     = np.array([   -0.20033967,  -0.0224465,   0.0428962])

#    efile = electron_MCRD_file
#    Ep    = 0.03999999910593033
#    name  = b'e-'
#    pdg   = 11
#    nhits = 4

#    return efile, name, pdg, vi, vf, p, Ep, nhits, X, Y, Z, E, t

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
    pmt0_first = (  0, 2)
    pmt0_last  = (378, 1)
    pmt0_tot_samples = 53

    sipm12013  = [(0, 1), (34, 1), (35, 1)]

    efile = os.path.join(ICDATADIR, 'Kr83_full.h5')

    return efile, pmt0_first, pmt0_last, pmt0_tot_samples, sipm12013

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
    test_file = "Kr_7bar_pmaps.h5"
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
    test_file = "Kr_7bar_KDST.h5"
    test_file = os.path.join(ICDATADIR, test_file)

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

    event = [3, 5, 8]
    time  = [  0,  0,  0]
    peak  = [  0,  0,  0]
    nS2   = [  1,  1,  1]

    S1w   = [150.0, 150.0, 150.0]
    S1h   = [1.97846091,  2.23351383,  1.20616078]
    S1e   = [9.53253078,  12.09763336,   6.44876432]
    S1t   = [100150.0, 100150.0, 100150.0]

    S2w   = [6.2746875 ,  5.68278125,  4.53234375]
    S2h   = [654.15313721,   683.74475098,  1188.52636719]
    S2e   = [2691.78320312,  2667.24829102,  3462.50341797]
    S2q   = [542.06463623,  526.95635986,  669.19622803]
    S2t   = [392479.4375,  373490.78125,  204476.515625]
    Nsipm = [15, 14, 12]

    DT    = [292.3294375 ,  273.34078125,  104.32651562]
    Z     = [584.658875  ,  546.6815625 ,  208.65303125]
    Zrms  = [1.43789648,  1.40609082,  0.99248303]
    X     = [46.31293282,  110.01751273,  -72.51959983]
    Y     = [-135.76067998,   56.26875304,   64.76540529]
    R     = [143.44284567,  123.57194535,   97.22988266]
    Phi   = [-1.24203938,  0.47276773,  2.412617]
    Xrms  = [7.55973859,  7.53298881,  6.23957521]
    Yrms  = [7.7685894 ,  7.39070164,  6.36868596]

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
                         "Zrms" : Zrms,
                         "X"    : X,
                         "Y"    : Y,
                         "R"    : R,
                         "Phi"  : Phi,
                         "Xrms" : Xrms,
                         "Yrms" : Yrms})

    df_read = load_dst(test_file,
                       group = group,
                       node  = node)

    return dst_data(tbl_data(test_file, group, node),
                    configuration,
                    df_read,
                    df_true)


@pytest.fixture(scope='session')
def KrMC_hdst(ICDATADIR):
    test_file = "Kr_7bar_HDST.h5"
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

    event = [3] * 5 + [5] * 6 + [8] * 5
    time  = [0]  * 16
    peak  = [0]  * 16
    nsipm = [6, 9, 9, 2, 6, 7, 8, 3, 9, 2, 0, 8, 9, 1, 1, 7]

    X     = [ 45.92089955,   46.12343447,   45.57183749,   51.92473694,
              46.616915  ,  109.90659715,  108.7973678 ,  125.        ,
             109.34261391,  125.        ,    0.        ,  -72.42377351,
             -73.02665708,  -55.        ,  -65.        ,  -73.23631779]

    Y     = [-135.33323446, -135.37371883, -135.1989478 , -155.        ,
             -134.73777219,   55.1502882 ,   56.93560653,   55.94482217,
              55.23480278,   52.25441809,    0.        ,   64.74595981,
              64.61950978,   65.        ,   85.        ,   65.05241448]

    Xrms  = [6.59632036,  6.96052693,  6.74357901,  4.61469259,  6.6189527 ,
             5.69963793,  6.26789152,  0.        ,  6.25831163,  0.        ,
             0.        ,  5.84845232,  5.92286867,  0.        ,  0.        ,
             5.45787191]

    Yrms  = [ 6.18217317,  6.73045979,  6.85202893,  0.        ,  5.70502447,
              7.59934684,  6.59871948,  7.26956706,  6.34538562,  4.46291375,
              0.        ,  5.68982819,  5.92711033,  0.        ,  0.        ,
              4.75869096]

    Z     = [ 578.5401875 ,  581.963375  ,  585.50275   ,  585.50275   ,
              588.665375  ,  542.27125   ,  545.883375  ,  545.883375  ,
              549.4025625 ,  549.4025625 ,  552.253625  ,  204.6431875 ,
              207.797     ,  207.797     ,  207.797     ,  210.73215625]

    Q     = [ 4.95296574e+01,   1.80898233e+02,   2.19138371e+02,
             9.61981177e+00,   4.53317676e+01,   7.95299282e+01,
             2.10233315e+02,   2.15513377e+01,   1.58320073e+02,
             1.52069054e+01,  -9.99999000e+05,   1.50914664e+02,
             4.11809814e+02,   6.81977272e+00,   5.71307802e+00,
             7.94903576e+01]

    E     = [  221.0178175 ,   982.84460449,  1122.37334476,    49.270332  ,
               316.27702713,   343.18898582,  1069.10244006,   109.59532148,
               913.54655235,    87.74765999,   144.0670352 ,   444.02672958,
              2123.30748505,    35.16301447,    29.45685338,   830.54934311]

    Xpeak = [ 46.3129329 ,   46.3129329 ,   46.3129329 ,   46.3129329 ,
              46.3129329 ,  110.01751265,  110.01751265,  110.01751265,
             110.01751265,  110.01751265,  110.01751265,  -72.51960002,
             -72.51960002,  -72.51960002,  -72.51960002,  -72.51960002]
    Ypeak = [-135.7606799 , -135.7606799 , -135.7606799 , -135.7606799 ,
             -135.7606799 ,   56.26875289,   56.26875289,   56.26875289,
               56.26875289,   56.26875289,   56.26875289,   64.76540542,
               64.76540542,   64.76540542,   64.76540542,   64.76540542]

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
    pmap_mctracks = load_mchits_nexus(pmap_filename)
    hdst_mctracks = load_mchits_nexus(hdst_filename)
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
