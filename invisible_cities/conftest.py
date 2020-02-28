import os
import pytest
import numpy  as np
import tables as tb

from pandas      import DataFrame
from collections import namedtuple

from . core              import system_of_units as units
from . evm  . pmaps_test import pmaps
from . io   . pmaps_io   import load_pmaps_as_df
from . io   . pmaps_io   import load_pmaps
from . io   . pmaps_io   import pmap_writer
from . io   .   dst_io   import load_dst
from . io   .  hits_io   import load_hits
from . io   .  hits_io   import load_hits_skipping_NN
from . io   .mcinfo_io   import load_mchits_df
from . types.ic_types    import NN

tbl_data = namedtuple('tbl_data', 'filename group node')
dst_data = namedtuple('dst_data', 'file_info config read true')
pmp_dfs  = namedtuple('pmp_dfs' , 's1 s2 si, s1pmt, s2pmt')
pmp_data = namedtuple('pmp_data', 's1 s2 si')
mcs_data = namedtuple('mcs_data', 'pmap hdst')
db_data  = namedtuple('db_data', 'detector npmts nsipms feboxes nfreqs')


@pytest.fixture(scope = 'session')
def ICDIR():
    return os.environ['ICDIR']


@pytest.fixture(scope = 'session')
def ICDATADIR(ICDIR):
    return os.path.join(ICDIR, "database/test_data/")

@pytest.fixture(scope = 'session')
def PSFDIR(ICDIR):
    return os.path.join(ICDIR, "database/test_data/PSF_dst_sum_collapsed.h5")

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
                params = ['electrons_40keV_z250_MCRD.h5'])
def electron_MCRD_file(request, ICDATADIR):
    return os.path.join(ICDATADIR, request.param)


@pytest.fixture(scope  = 'session',
                params = ['Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts.MCRD.h5'])
def krypton_MCRD_file(request, ICDATADIR):
    return os.path.join(ICDATADIR, request.param)

@pytest.fixture(scope  = 'session',
                params = ['mcfile_nohits.sim.h5'])
def nohits_sim_file(request, ICDATADIR):
    return os.path.join(ICDATADIR, request.param)

@pytest.fixture(scope  = 'session',
                params = ['mcfile_sns_only.sim.h5'])
def sns_only_sim_file(request, ICDATADIR):
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
def correction_map_filename(ICDATADIR):
    test_file = "kr_emap_xy_100_100_r_6573_time.h5"
    test_file = os.path.join(ICDATADIR, test_file)
    return test_file

@pytest.fixture(scope='session')
def correction_map_MC_filename(ICDATADIR):
    test_file = "kr_emap_xy_100_100_mc.h5"
    test_file = os.path.join(ICDATADIR, test_file)
    return test_file


@pytest.fixture(scope='session')
def KrMC_hdst_filename(ICDATADIR):
    test_file = "Kr83_nexus_v5_03_00_ACTIVE_7bar_10evts_HDST.h5"
    test_file = os.path.join(ICDATADIR, test_file)
    return test_file


@pytest.fixture(scope='session')
def KrMC_hdst_filename_toy(ICDATADIR):
    test_file = "toy_MC_HDST.h5"
    test_file = os.path.join(ICDATADIR, test_file)
    return test_file


@pytest.fixture(scope='session')
def esmeralda_tracks(ICDATADIR):
    test_file = "esmeralda_tracks.hdf5"
    test_file = os.path.join(ICDATADIR, test_file)
    return test_file


@pytest.fixture(scope='session')
def data_hdst(ICDATADIR):
    test_file = "test_hits_th_1pes.h5"
    test_file = os.path.join(ICDATADIR, test_file)
    return test_file

@pytest.fixture(scope='session')
def data_hdst_deconvolved(ICDATADIR):
    test_file = "test_hits_th_1pes_deconvolution.npz"
    test_file = os.path.join(ICDATADIR, test_file)
    return test_file

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

    configuration = dict(run_number    =  -4734,
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
                         global_reco_params =   dict(
                             Qthr           =   1 * units.pes,
                             Qlm            =   0 * units.pes,
                             lm_radius      =  -1 * units.mm,
                             new_lm_radius  =  -1 * units.mm,
                             msipm          =   1           )     )

    event    = [0, 1, 2, 3, 4, 5, 6, 7, 9]
    time     = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    s1_peak  = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    nS1      = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    s2_peak  = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    nS2      = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    S1w      = [200., 200., 175., 175.,
                200., 175., 175., 250., 200.]
    S1h      = [2.20967579, 4.76641989, 2.19646263, 2.28649235,
                2.10444999, 2.80842876, 1.66933489, 2.51325226, 1.45906901]
    S1e      = [11.79215431, 24.58425140, 11.20991993, 10.81333828,
                10.09474468, 15.06069851,  7.42862463, 18.16711235, 9.39478493]
    S1t      = [100225, 100100, 100150, 100125,
                100100, 100150, 100150, 100175, 100150]

    S2w      = [ 8.300, 8.775, 9.575,  8.550,
                11.175, 8.175, 6.800, 11.025,
                 6.600]
    S2h      = [ 709.73730469,  746.38555908,  717.25158691, 1004.44152832,
                 972.86987305, 1064.57861328, 1101.23962402,  532.64318848,
                1389.43591309]
    S2e      = [3333.79321289, 3684.11181641, 3815.17578125, 4563.97412109,
                4995.36083984, 4556.64892578, 4572.18945313, 3301.98510742,
                5057.96923828]
    S2q      = [502.42642212, 520.74383545, 507.32211304, 607.92211914,
                585.70526123, 578.10437012, 595.45251465, 479.56573486,
                633.57226563]
    S2t      = [424476.3, 495500.0, 542483.1, 392466.5,
                277481.8, 373512.6, 322489.5, 562502.9,
                263483.5]
    Nsipm    = [13, 14, 15, 13, 12, 14, 14, 16, 12]

    DT       = [324.25134277, 395.40005493, 442.33322144, 292.34155273,
                177.38185120, 273.36264038, 222.33953857, 462.32797241,
                163.33348083]
    Z        = [648.50268555, 790.80010986, 884.66644287, 584.68310547,
                354.76370239, 546.72528076, 444.67907715, 924.65594482,
                326.66696167]
    Zrms     = [1.71384448, 1.79561206, 1.95215662, 1.69040076,
                2.67407129, 1.57506763, 1.39490784, 2.27981592,
                1.25179053]
    X        = [137.83250525, -137.61807146,  98.89289503,   45.59115903,
                -55.56241288,  108.79125729, -65.40441501,  146.19707698,
                111.82481626]
    Y        = [124.29040384, -99.36910382,  97.30389804, -136.05131238,
                -91.45151520,  56.89100575, 130.07815715,   85.99513413,
                -43.18583024]
    R        = [185.59607752, 169.74378453, 138.73663273, 143.48697984,
                107.00729581, 122.76857985, 145.59555099, 169.61352662,
                119.87412342]
    Phi      = [0.733780880, -2.51621137, 0.77729934, -1.24745421,
                -2.11675716, 0.481828610, 2.03668828,  0.53170808,
                -0.36854640]
    Xrms     = [7.46121721,  8.06076669, 7.81822080, 7.57937056,
                6.37266772, 15.45983120, 7.42686347, 8.19314573,
                7.17110860]
    Yrms     = [7.32118281, 7.63813330, 8.35199371, 6.93334565,
                7.22325362, 7.49551110, 7.32949621, 8.06191793,
                6.26888356]

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

    ZANODE = -9.425 * units.mm

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
                         global_reco_params =   dict(
                             Qthr           =   1  * units.pes,
                             Qlm            =   0  * units.pes,
                             lm_radius      =  -1  * units.mm ,
                             new_lm_radius  =  -1  * units.mm ,
                             msipm          =   1             ),
                         slice_reco_params  =   dict(
                             Qthr           =   2  * units.pes,
                             Qlm            =   5  * units.pes,
                             lm_radius      =   0  * units.mm ,
                             new_lm_radius  =   15 * units.mm ,
                             msipm          =   1             ))

    event = [0] * 7 + [1] * 10 + [2] * 8 + [3] * 5 + [4] * 8 + [5] * 9 + [6] * 7 + [7] * 7 + [9] * 6
    time  = [0]  * 67
    peak  = [0]  * 67
    nsipm = [8, 9, 2, 9, 7, 1, 0,
             7, 2, 8, 2, 9, 1, 2, 8, 1, 0,
             5, 9, 2, 2, 9, 2, 8, 4,
             8, 9, 9, 7, 0,
             7, 9, 3, 7, 7, 2, 9, 0,
             5, 9, 2, 8, 3, 7, 2, 1, 0,
             8, 2, 9, 3, 1, 9, 3,
             6, 9, 9, 8, 2, 8, 0,
             7, 9, 2, 7, 2, 4]

    X     = [137.02335083, 136.05361214, 155, 137.50739563,
             139.44340629, 135, 0.,
             -136.91523372, -138.31812254, -137.40236334, -139.36096871,
             -136.614188  , -135.        , -155.        , -137.13894421,
             -135.        ,    0.,
             96.69981723,  97.48205345,  96.84759934, 115.        ,
             96.87320369, 115.        , 101.92102321,  97.29559249,
             43.04750178, 46.37438754, 45.18567591, 45.30916066,  0,
             -56.51582522, -56.16605278, -54.00613888, -55.77326839,
             -54.43059006, -52.08436199, -54.76790039,   0.,
             106.9129152 , 108.02149081, 125.        , 111.23591207,
             95.        , 109.36687514, 125.        , -35.        ,
             0,
             -66.81829731, -60.99443543, -65.21616666, -56.82709265,
             -85.        , -66.38902019, -67.16981495,
             146.62612941, 145.98052086, 144.56275403, 146.57670316,
             165.        , 143.75471749,   0.,
             112.68567761, 113.05236723,  95.        , 112.18326138,
             95.        , 111.13948562]

    Y     = [123.3554201 , 124.21305414, 130.32028088, 125.13468301,
             125.723091  , 105.        ,   0.,
             -102.20454509,  -85.      , -101.19527484,  -85.    ,
             -98.18372292, -115.       ,  -97.50736874,  -97.65435292,
             -115.        ,    0.,
             97.83535504,  96.71162732, 115.        ,  99.94110169,
             96.96828871,  91.9964654 ,  96.47442012,  97.41483565,
             -135.46032693, -135.78741482, -135.45706672, -136.84192389,
             0,
             -91.93474203, -92.22587474, -75.        , -91.79293263,
             -92.05845326, -75.        , -93.17777252,   0,
             60.37973152, 55.49979777, 58.7016448 , 56.05515802,
             52.82336998, 57.04430991, 61.02136179, 65.        ,
             0.,
             131.35172348, 115.        , 131.55083652, 116.93600131,
             125.        , 131.38450477, 127.55032531,
             86.4185202 , 85.50276576, 85.08681108, 85.63406367,
             92.12569852, 87.37079899,  0.,
             -41.26775923, -43.95642973, -42.01248503, -43.6128179 ,
             -39.43319563, -41.83429664]

    Xrms  = [6.82943353, 6.54561049, 0.        , 6.57516493,
            6.10884172, 0.        , 0.,
            6.34641453, 4.70863974, 6.99252219, 4.95899577,
            6.99286177, 0.        , 0.        , 7.31434091,
            0.        , 0.,
            5.46065169, 6.95289236, 3.88102693, 0.        ,
            7.08872666, 0.        , 6.12004979, 4.20549403,
            6.50762165, 6.53200551, 6.72634463, 6.63232452,
            0.,
            6.01927117, 6.15010511, 7.21567172, 6.18867493,
            6.68684705, 4.54482509, 6.49772479, 0,
            6.93471034, 6.73472262, 0.        , 6.23230238,
            0., 6.60554616, 0.        , 0.        ,
            0.,
            6.5619163 , 4.90011206, 6.62430967, 7.32479932,
            0., 6.44742043, 4.12189915,
            5.39988073, 6.60345367, 6.72834925, 6.74313437,
            0., 7.29624974, 0.,
            6.91160248, 6.11943089, 0.        , 5.61844221,
            0., 4.86842608]

    Yrms  = [7.53561556, 6.7267195 , 4.98973147, 6.66971537,
             6.72639992, 0.        , 0,
             7.6755039 , 0.        , 6.32848615, 0.        ,
             6.5346365 , 0.        , 4.33437301, 6.08928588,
             0.        , 0,
             4.50714013, 7.16164783, 0.        , 4.99965309,
             6.85733991, 4.5841167 , 6.9783623 , 4.27982772,
             6.75410279, 6.79005446, 6.72968451, 6.11981085,
             0,
             5.70842513, 6.38557803, 0.        , 5.52636093,
             5.17674174, 0.        , 6.9095478 , 0.,
             4.98555954, 6.83197289, 4.82848566, 6.69411227,
             6.65765325, 7.0682494 , 4.89457047, 0.        ,
             0,
             6.26325683, 0.        , 6.33195824, 3.95119122,
             0., 6.41624414, 4.35879501,
             5.88461762, 7.49736328, 6.59322568, 6.52422553,
             4.52563872, 6.98159176, 0.,
             5.49357201, 6.10943873, 4.57710651, 5.55130713,
             4.9677694 , 4.65138214]

    Z     = [642.291    ,  645.7665  , 645.7665   , 649.3726875,
             653.05675  , 653.05675  , 656.1485   ,
             784.297    , 784.297    , 788.0405   , 788.0405   ,
             791.746    , 791.746    , 791.746    , 795.323    ,
             795.323    , 798.587875 ,
             876.165125 , 879.974125 , 879.974125 , 879.974125,
             883.695    , 883.695    , 887.402625 , 891.02675 ,
             578.47125  , 582.0140625, 585.5928125, 589.239125,
             592.4990625,
             350.6575   , 353.8748125, 353.8748125, 357.335875,
             361.9025   , 361.9025   , 365.6785   , 368.9885  ,
             540.8020625, 544.093625 , 544.093625 , 547.667375,
             547.667375 , 551.231625 , 551.231625 , 551.231625,
             554.582,
             440.6051875, 440.6051875, 443.814375 , 443.814375,
             443.814375 , 447.4028125, 450.6166875,
             916.4715   , 919.979    , 923.716    , 927.529125,
             927.529125 , 931.291125 , 935.062625,
             320.8508125, 324.0374375, 324.0374375, 327.4530625,
             327.4530625, 330.653625]

    Q     = [6.21135249e+01,  1.93207452e+02,  1.17878785e+01,  1.52947767e+02,
             5.92023814e+01,  5.98997307e+00, NN,
             5.11176143e+01,  7.79754877e+00,  1.74522912e+02,  1.52560458e+01,
             1.51785870e+02,  9.12431908e+00,  1.20135555e+01,  6.86140332e+01,
             5.19579268e+00,  NN,
             38.49134254, 130.57549524,  12.6504097 ,  12.36747885,
             158.67850065,  13.2809248 ,  80.46449661,  18.56184649,
             7.04285767e+01,  2.03529454e+02,  2.23206003e+02,  7.24489250e+01,
             NN,
             8.47140183e+01,  2.18426581e+02,  1.07217135e+01,  8.33113148e+01,
             6.86556058e+01,  8.63394284e+00,  7.88299217e+01,  NN,
             2.79961238e+01,  1.58822554e+02,  1.55958052e+01,  2.37206266e+02,
             1.47270021e+01,  7.56110237e+01,  1.20107818e+01,  5.01965523e+00,
             NN,
             119.14831066,  13.13961554, 247.54202271,  26.03989458,
             6.0433917 , 145.98711324,  21.32093287,
             3.22954066e+01,  7.83973203e+01,  1.35020741e+02,  1.22896467e+02,
             7.75498104e+00,  4.41479754e+01,  NN,
             61.85911345, 297.65047169,  14.66699696, 208.5009613 ,
             15.51440573,  14.46035504]

    Ql    = [-1] * len(Q)

    Qc    = [-1] * len(Q)

    E     = [300.67897224, 1031.45098773,   62.93038282, 1264.9021225 ,
             574.87652458,   58.16480385,   40.7894727,
             261.72873756,   39.92444923, 1007.64861959,   88.08432872,
             1275.08688025,   76.64942446,  100.92063957,  663.7729013 ,
             50.26415467,  120.03185987,
             255.05263805,  751.72643276,   72.82872899,   71.19988891,
             1293.0319564 ,  108.22297982, 1000.51422119,  262.59910393,
             358.72694397, 1477.87328339, 1833.91390991,  799.69952011,
             93.7603054,
             435.34708595, 1719.97730608,   84.42701369, 1209.82822418,
             439.89744612,   55.32031015,  815.31221771,  235.25139713,
             129.01727104,  988.40456541,   97.05778352, 1980.85363294,
             122.98172472,  876.96921053,  139.30621875,   58.22012279,
             163.83844948,
             434.5464781 ,   47.92156618, 1774.64676229,  186.6818979 ,
             43.32551461, 1761.74893188,  323.31822968,
             127.10046768,  485.85840988, 1011.98860931,  919.25156787,
             58.00637433,  525.53125763,  174.24835968,
             230.65227795, 1978.67551363,   97.50103056, 2193.99010015,
             163.2532165 ,  393.89710426]

    El    = [-1] * len(E)

    Ec    = [-1] * len(E)

    Zc    = [ZANODE] * len(E)

    Xpeak = [137.83250525] * 7 + [-137.61807157] * 10 + [98.89289499] * 8 + [45.59115902] * 5 + [-55.56241282] * 8 + [108.79125728] * 9 + [-65.40441505] * 7 + [146.19707713] * 7 + [111.82481627] * 6

    Ypeak = [124.29040397] * 7 + [-99.36910381] * 10 + [97.3038981] * 8 + [-136.05131239] * 5 + [-91.45151521] * 8 + [56.89100565] * 9 + [130.07815717] * 7 + [85.995134] * 7 + [-43.18583022] * 6

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
                         "E"    : E,
                         "Ql"   : Ql,
                         "El"   : El,
                         "Qc"   : Qc,
                         "Ec"   : Ec,
                         "Zc"   : Zc})

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
    pmap_mctracks = load_mchits_df(pmap_filename)
    hdst_mctracks = load_mchits_df(hdst_filename)
    return mcs_data(pmap_mctracks, hdst_mctracks)


@pytest.fixture(scope='session')
def TlMC_hits(ICDATADIR):
    hits_file_name = "dst_NEXT_v1_00_05_Tl_ACTIVE_100_0_7bar_DST_10.h5"
    hits_file_name = os.path.join(ICDATADIR, hits_file_name)
    hits = load_hits(hits_file_name)
    return hits


@pytest.fixture(scope='session')
def TlMC_hits_skipping_NN(ICDATADIR):
    hits_file_name = "dst_NEXT_v1_00_05_Tl_ACTIVE_100_0_7bar_DST_10.h5"
    hits_file_name = os.path.join(ICDATADIR, hits_file_name)
    hits = load_hits_skipping_NN(hits_file_name)
    return hits

@pytest.fixture(scope='session')
def TlMC_hits_merged(ICDATADIR):
    hits_file_name = "dst_NEXT_v1_00_05_Tl_ACTIVE_100_0_7bar_DST_10_merged.h5"
    hits_file_name = os.path.join(ICDATADIR, hits_file_name)
    hits = load_hits(hits_file_name)
    return hits


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
def Xe2nu_pmaps_mc_filename(ICDATADIR):
    filename = os.path.join(ICDATADIR, "Xe2nu_NEW_v1_05_02_nexus_v5_03_08_ACTIVE_10.2bar_run4.0_0_pmaps.h5")
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


@pytest.fixture(scope='session')
def dbdemopp():
    return 'demopp'

@pytest.fixture(scope='session')
def dbnew():
    return 'new'

@pytest.fixture(scope='session')
def dbnext100():
    return 'next100'

@pytest.fixture(scope='session',
                params=[db_data('demopp' ,  3,  256, 3, 79),
                        db_data('new'    , 12, 1792, 3, 79),
                        db_data('next100', 60, 6848, 8, 79)],
               ids=["demo", "new", "next100"])
def db(request):
    return request.param

@pytest.fixture(scope='function') # Needs to be function as the config dict is modified when running
def deconvolution_config(ICDIR, ICDATADIR, PSFDIR, config_tmpdir):
    PATH_IN     = os.path.join(ICDATADIR    ,    "test_Xe2nu_NEW_v1.2.0_cdst.5_62.h5")
    PATH_OUT    = os.path.join(config_tmpdir,                       "beersheba_MC.h5")
    nevt_req    = 3
    conf        = dict(files_in      = PATH_IN ,
                       file_out      = PATH_OUT,
                       event_range   = nevt_req,
                       compression   = 'ZLIB4',
                       print_mod     = 1000,
                       run_number    = 0,
                       deconv_params = dict(q_cut         =         10,
                                            drop_dist     = [10., 10.],
                                            psf_fname     =     PSFDIR,
                                            e_cut         =       1e-3,
                                            n_iterations  =         10,
                                            iteration_tol =       0.01,
                                            sample_width  = [10., 10.],
                                            bin_size      = [ 1.,  1.],
                                            energy_type   =        'E',
                                            diffusion     = (1.0, 1.0),
                                            deconv_mode   =    'joint',
                                            n_dim         =          2,
                                            cut_type      =      'abs',
                                            inter_method  =    'cubic'))

    return conf, PATH_OUT
