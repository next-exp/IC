import os
import pytest
import shutil

import numpy  as np
import tables as tb

from pandas      import DataFrame
from collections import namedtuple

from . core              import system_of_units as units
from . io   . pmaps_io   import load_pmaps_as_df
from . io   . pmaps_io   import load_pmaps
from . io   .   dst_io   import load_dst
from . io   .  hits_io   import load_hits
from . io   .  hits_io   import load_hits_skipping_NN
from . io   .mcinfo_io   import load_mchits_df
from . types.ic_types    import NN
from . types.symbols     import XYReco
from . types.symbols     import RebinMethod
from . types.symbols     import HitEnergy
from . types.symbols     import DeconvolutionMode
from . types.symbols     import CutType
from . types.symbols     import SiPMCharge
from . types.symbols     import InterpolationMethod
from . types.symbols     import NormStrategy

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
def PSFDIR(ICDATADIR):
    return os.path.join(ICDATADIR, "NEXT100_PSF_kr83m.h5")

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

@pytest.fixture(scope  = 'session',
                params = ['Kr83_full_nexus_v5_03_01_ACTIVE_7bar_1evt.sim.h5'])
def full_sim_file(request, ICDATADIR):
    return os.path.join(ICDATADIR, request.param)


@pytest.fixture(scope='session')
def mc_all_hits_data(krypton_MCRD_file):
    number_of_hits = 8
    evt_number     = 2
    efile          = krypton_MCRD_file
    return efile, number_of_hits, evt_number


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
                         global_reco_algo   = XYReco.barycenter,
                         global_reco_params = dict(Qthr = 1 * units.pes))

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
    qmax     = [134.07653809, 100.63265228, 109.73874664, 171.19009399,
                190.54495239, 116.1299057 , 146.56044006, 118.97045135,
                212.43554688]
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
                         "qmax"   : qmax,

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


@pytest.fixture(scope="session")
def Th228_pmaps(ICDATADIR):
    filename = "228Th_10evt_pmaps.h5"
    filename = os.path.join(ICDATADIR, filename)
    return filename


@pytest.fixture(scope="session")
def Th228_hits(ICDATADIR):
    filename = "228Th_10evt_hits.h5"
    filename = os.path.join(ICDATADIR, filename)
    return filename


@pytest.fixture(scope="session")
def Th228_tracks(ICDATADIR):
    filename = "228Th_10evt_tracks.h5"
    filename = os.path.join(ICDATADIR, filename)
    return filename


@pytest.fixture(scope="session")
def Th228_deco(ICDATADIR):
    filename = "228Th_10evt_deco.h5"
    filename = os.path.join(ICDATADIR, filename)
    return filename


@pytest.fixture(scope="session")
def Th228_deco_separate(ICDATADIR):
    filename = "228Th_10evt_deco_separate.h5"
    filename = os.path.join(ICDATADIR, filename)
    return filename


@pytest.fixture(scope="session")
def Th228_hits_missing(Th228_hits, config_tmpdir):
    """Copy input file and remove the hits from the first event"""
    outpath = os.path.basename(Th228_hits).replace(".h5", "_missing_hits.h5")
    outpath = os.path.join(config_tmpdir, outpath)
    shutil.copy(Th228_hits, outpath)
    with tb.open_file(outpath, "r+") as file:
        first_evt = file.root.Run.events[0][0]
        evt_rows  = [row[0] == first_evt for row in file.root.RECO.Events]
        n_delete  = sum(evt_rows)
        file.root.RECO.Events.remove_rows(0, n_delete)
    return outpath


@pytest.fixture(scope="session")
def next100_mc_krmap(ICDATADIR):
    filename = "map_NEXT100_MC.h5"
    filename = os.path.join(ICDATADIR, filename)
    return filename


@pytest.fixture(scope="function")
def sophronia_config(Th228_pmaps, next100_mc_krmap):
    config   = dict( files_in    = Th228_pmaps
                   , compression = "ZLIB4"
                   , event_range = 10
                   , run_number  = 0
                   , detector_db = "next100"
                   , print_mod   = 1
                   , drift_v     = 0.84 * units.mm / units.mus
                   , s1_params   = dict(
                        s1_nmin     =    1            ,
                        s1_nmax     =    5            ,
                        s1_emin     =    5 * units.pes,
                        s1_emax     =  1e4 * units.pes,
                        s1_wmin     =   75 * units.ns ,
                        s1_wmax     =    2 * units.mus,
                        s1_hmin     =    2 * units.pes,
                        s1_hmax     =  1e4 * units.pes,
                        s1_ethr     =    0 * units.pes,
                   )
                   , s2_params   = dict(
                        s2_nmin     =    1            ,
                        s2_nmax     =    5            ,
                        s2_emin     =  1e2 * units.pes,
                        s2_emax     =  1e9 * units.pes,
                        s2_wmin     =  0.5 * units.mus,
                        s2_wmax     =  1e3 * units.ms ,
                        s2_hmin     =  1e2 * units.pes,
                        s2_hmax     =  1e9 * units.pes,
                        s2_nsipmmin =    1            ,
                        s2_nsipmmax = 3000            ,
                        s2_ethr     =    0 * units.pes,
                   )
                   , rebin              = 1
                   , rebin_method       = RebinMethod.stride
                   , sipm_charge_type   = SiPMCharge.raw
                   , q_thr              = 5 * units.pes
                   , global_reco_algo   = XYReco.barycenter
                   , global_reco_params = dict(Qthr = 20 * units.pes)
                   , same_peak          = True
                   , corrections        = dict(
                       filename   = next100_mc_krmap,
                       apply_temp =            False,
                       norm_strat =  NormStrategy.kr)
                   )
    return config



@pytest.fixture(scope="function")
def esmeralda_config(Th228_hits, next100_mc_krmap):
    config = dict( files_in    = Th228_hits
                 , compression = "ZLIB4"
                 , event_range = 8
                 , run_number  = 0
                 , detector_db = "next100"
                 , print_mod   = 1
                 , threshold   = 30 * units.pes
                 , same_peak   = True
                 , fiducial_r  = 474 * units.mm
                 , paolina_params  = dict(
                      vox_size         = [15 * units.mm] * 3,
                      strict_vox_size  = True               ,
                      energy_threshold = 20 * units.keV     ,
                      min_voxels       = 3                  ,
                      blob_radius      = 21 * units.mm      ,
          	      max_num_hits     = 30000              )
                 , corrections = dict(
                      filename   = next100_mc_krmap,
                      apply_temp =            False,
                      norm_strat =  NormStrategy.kr)
                 )

    return config


@pytest.fixture(scope='session')
def hits_toy_data(ICDATADIR):
    npeak  = np.array   ([0]*25 + [1]*30 + [2]*35 + [3]*10)
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
    return hits_filename, (npeak, x, y, xrms, yrms, z, q, e, x_peak, y_peak)


@pytest.fixture(scope='session')
def random_hits_toy_data():

    n = 50
    xs = np.random.uniform(-10, 10, n)
    ys = np.random.uniform(-10, 10, n)
    zs = np.random.uniform( 10, 50, n)
    es = np.random.uniform(  0, 90, n)
    qs = np.random.uniform(  0, 20, n)

    hits = DataFrame(dict( event=0, time=1e9, npeak=0, Xpeak=-1., Ypeak=-2.
                         , X=xs, Y=ys, Z=zs, E=es, Q=qs))
    return hits


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


@pytest.fixture(scope='session')
def dbdemopp():
    return 'demopp'

@pytest.fixture(scope='session')
def dbnew():
    return 'new'

@pytest.fixture(scope='session')
def dbnext100():
    return 'next100'

@pytest.fixture(scope='session')
def dbflex100():
    return 'flex100'

@pytest.fixture(scope='session',
                params=[db_data('demopp' ,  3,  256, 3, 79),
                        db_data('new'    , 12, 1792, 3, 79),
                        db_data('next100', 60, 3584, 0, 0),
                        db_data('flex100', 60, 3093, 0, 0)],
               ids=["demo", "new", "next100", "flex100"])
def db(request):
    return request.param


@pytest.fixture(scope='function')
def beersheba_config(Th228_hits, PSFDIR, next100_mc_krmap):
    config = dict( files_in      = Th228_hits
                 , event_range   = 80
                 , compression   = 'ZLIB4'
                 , detector_db   = "next100"
                 , print_mod     = 1
                 , run_number    = 0
                 , threshold     = 5 * units.pes
                 , same_peak     = True
                 , deconv_params = dict( q_cut         = 10
                                       , drop_dist     = [16.0] * 2
                                       , psf_fname     = PSFDIR
                                       , e_cut         = 12e-3
                                       , n_iterations  = 100
                                       , iteration_tol = 1e-10
                                       , sample_width  = [15.55] * 2
                                       , bin_size      = [ 1.,  1.]
                                       , diffusion     = (1.0, 0.2)
                                       , n_dim         = 2
                                       , energy_type   = HitEnergy.Ec
                                       , deconv_mode   = DeconvolutionMode.joint
                                       , cut_type      = CutType.abs
                                       , inter_method  = InterpolationMethod.cubic)
                 , satellite_params = None
                 , corrections   = dict( filename   = next100_mc_krmap
                                       , apply_temp = False
                                       , norm_strat = NormStrategy.kr))
    return config


@pytest.fixture(scope='function')
def beersheba_config_separate(beersheba_config):
    beersheba_config["deconv_params"].update(dict( deconv_mode    = DeconvolutionMode.separate
                                                 , n_iterations   = 50
                                                 , n_iterations_g = 50))

    return beersheba_config


## To make very slow tests only run with specific option
def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
        )


def pytest_configure(config):
    config.addinivalue_line("markers", "veryslow: mark test as very slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "veryslow" in item.keywords:
            item.add_marker(skip_slow)
