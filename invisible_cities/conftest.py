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
from . io  .mchits_io         import load_mchits


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
                params = ['dst_NEXT_v1_00_05_Tl_ACTIVE_140_0_7bar_PMP_2.h5'])
def thallium_DST_file(request, ICDATADIR):
    return os.path.join(ICDATADIR, request.param)


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
    test_file = "dst_NEXT_v1_00_05_Kr_ACTIVE_0_0_7bar_PMP_10evt_new.h5"
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
    test_file = "dst_NEXT_v1_00_05_Kr_ACTIVE_0_0_7bar_KDST_10evt_new.h5"
    test_file = os.path.join(ICDATADIR, test_file)

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

    event = [17, 19, 25]
    time  = [  0,  0,  0]
    peak  = [  0,  0,  0]
    nS2   = [  1,  1,  1]

    S1w   = [250.0, 225.0, 200.0]
    S1h   = [3.2523329257965088, 2.4484190940856934, 1.7538540363311768]
    S1e   = [19.411735534667969, 15.692093849182129, 10.745523452758789]
    S1t   = [100150.0, 100100.0, 100100.0]

    S2w   = [11.228562500000001, 11.358750000000001, 8.5991874999999993]
    S2h   = [880.9984130859375, 851.11505126953125, 1431.811279296875]
    S2e   = [5008.1044921875, 4993.95166015625, 5979.82373046875]
    S2q   = [157.06857299804688, 146.64302062988281, 263.83566284179688]
    S2t   = [554493.0625, 625482.0, 338503.65625]
    Nsipm = [5, 6, 6]

    DT    = [454.34306250000003, 525.38200000000006, 238.40365625000001]
    Z     = [908.68612500000006, 1050.7640000000001, 476.80731250000002]
    Zrms  = [2.082009521484375, 2.19840771484375, 1.5321597900390624]
    X     = [120.2356729453311, 1.0275025311778865, 154.77121933901094]
    Y     = [106.26682866748105, 146.81638355271943, 71.503424708151769]
    R     = [160.46574688593327, 146.8199790251681, 170.49008792501272]
    Phi   = [0.7238042531182578, 1.5637978860814405, 0.43278351817373956]
    Xrms  = [4.9944427379677672, 4.8932850467168816, 5.7780285188196077]
    Yrms  = [5.3678579292067852, 7.1442047998378158, 4.7686176347992912]

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
    test_file = "dst_NEXT_v1_00_05_Kr_ACTIVE_0_0_7bar_HDST_10evt_new.h5"
    test_file = os.path.join(ICDATADIR, test_file)

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

    event = [17] * 7 + [19] * 8 + [25] * 5
    time  = [0]  * 20
    peak  = [0]  * 20
    nsipm = [0, 4, 5, 4, 0, 0, 0, 0, 3, 4, 2, 6, 0, 0, 0, 1, 6, 6, 1, 0]

    X     = [0.0,
             120.19301521736676,
             120.2417342322328,
             119.8876430125747,
             0.0,
             0.0,
             0.0,
             0.0,
             1.4141658648702744,
             2.0059661341323785,
             -0.44500029002100205,
             0.71450467122860617,
             0.0,
             0.0,
             0.0,
             155.0,
             155.30091118461135,
             154.42891365610399,
             155.0,
             0.0]

    Y     = [0.0, 108.12404705116668, 104.42646749780661, 107.77236935123307,  0.0,  0.0, 0.0, 0.0,
             142.75961853042116, 150.74664509647295, 135.0, 147.01393806830407, 0.0, 0.0, 0.0,
             75.0, 71.165499244103017, 71.181017825460103, 75.0, 0.0]

    Xrms  = [0.0,
            4.9962731236257358,
            4.9941530373995162,
            4.9987374313297064,
            0.0,
            0.0,
            0.0,
            0.0,
            4.7958455882811437,
            4.5799672344585609,
            4.9801581041048513,
            4.9486849843966123,
            0.0,
            0.0,
            0.0,
            0.0,
            5.5135164273344754,
            6.3081332082454828,
            0.0,
            0.0]

    Yrms  = [0.0,
             4.6347384536523304,
             5.7924982219597982,
             4.4763446798335611,
             0.0,
             0.0,
             0.0,
             0.0,
             4.1694730562213946,
             4.9439378131114182,
             0.0,
             7.0751297093843268,
             0.0,
             0.0,
             0.0,
             0.0,
             4.8622640314975998,
             4.8585179732039174,
             0.0,
             0.0]

    Z     = [900.48312499999997,
             904.04050000000007,
             907.76300000000003,
             911.51125000000002,
             915.24025000000006,
             919.07112500000005,
             921.70000000000005,
             1040.7703750000001,
             1044.0696250000001,
             1047.867125,
             1047.867125,
             1051.664,
             1055.4748750000001,
             1059.220875,
             1062.1355000000001,
             470.79374999999999,
             474.22925000000004,
             477.73925000000003,
             481.31693749999999,
             484.77218750000003]

    Q     = [-999999.0,
             29.235990524291992,
             70.374324321746826,
             53.386228561401367,
             -999999.0,
             -999999.0,
             -999999.0,
             -999999.0,
             17.607906818389893,
             41.205404996871948,
             10.87603235244751,
             68.89520788192749,
             -999999.0,
             -999999.0,
             -999999.0,
             9.8147153854370117,
             88.746064186096191,
             152.45538997650146,
             12.819503784179688,
             -999999.0]

    E     = [142.57600879669189,
             793.51572418212891,
             1678.7182083129883,
             1562.8402099609375,
             689.81671142578125,
             139.54881477355957,
             1.0885893274098635,
             118.0997519493103,
             813.1905517578125,
             1173.6551782119202,
             309.78245911718147,
             1565.8631744384766,
             815.27002334594727,
             184.44183254241943,
             13.648312956094742,
             167.12385177612305,
             1548.9741058349609,
             2809.6304168701172,
             1303.8236541748047,
             150.27177047729492]

    Xpeak = [120.23567275745617, 120.23567275745617, 120.23567275745617, 120.23567275745617, 120.23567275745617, 120.23567275745617, 120.23567275745617,
             1.0275026443161928, 1.0275026443161928, 1.0275026443161928, 1.0275026443161928, 1.0275026443161928, 1.0275026443161928, 1.0275026443161928, 1.0275026443161928,
             154.7712193078262, 154.7712193078262, 154.7712193078262, 154.7712193078262, 154.7712193078262]
    Ypeak = [106.26682866707235, 106.26682866707235, 106.26682866707235, 106.26682866707235, 106.26682866707235, 106.26682866707235, 106.26682866707235,
             146.81638358971426, 146.81638358971426, 146.81638358971426, 146.81638358971426, 146.81638358971426, 146.81638358971426, 146.81638358971426, 146.81638358971426,
             71.50342482013173, 71.50342482013173, 71.50342482013173, 71.50342482013173, 71.50342482013173]

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
