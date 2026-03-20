import os
import warnings
import numpy  as np
import tables as tb
import pandas as pd

from pytest import mark

from .. core                 import system_of_units as units
from .. core.core_functions  import in_range
from .. core.testing_utils   import assert_dataframes_close
from .. core.testing_utils   import assert_tables_equality
from .. core.testing_utils   import ignore_warning
from .. core.configure       import configure
from .. io                   import dst_io as dio
from .. io.mcinfo_io         import load_mchits_df
from .. io.mcinfo_io         import load_mcparticles_df
from .. types.symbols        import all_events
from .. types.symbols        import RebinMethod
from .. types.symbols        import SiPMCharge

from .  penthesilea          import penthesilea


#in order not to fail direct comparation tests when changing hit attribute we compare only the columns that penthesilea is using
columns = ['event', 'time', 'npeak', 'Xpeak', 'Ypeak', 'X', 'Y', 'Z', 'Q', 'E']

@ignore_warning.no_config_group
def test_penthesilea_KrMC(KrMC_pmaps_filename, KrMC_hdst, KrMC_kdst, config_tmpdir):
    PATH_IN   = KrMC_pmaps_filename
    PATH_OUT  = os.path.join(config_tmpdir,'Kr_HDST.h5')
    conf      = configure('dummy invisible_cities/config/penthesilea.conf'.split())
    nevt_req  = 10

    DF_TRUE_RECO =  KrMC_hdst   .true
    DF_TRUE_DST  =  KrMC_kdst[0].true

    conf.update(dict(files_in      = PATH_IN,
                     file_out      = PATH_OUT,
                     event_range   = nevt_req,
                     **KrMC_hdst.config))

    cnt = penthesilea(**conf)
    assert cnt.events_in  == nevt_req
    assert cnt.events_out == len(set(DF_TRUE_RECO.event))
    assert cnt.events_out == len(set(DF_TRUE_DST .event))

    df_penthesilea_reco = dio.load_dst(PATH_OUT , 'RECO', 'Events')
    df_penthesilea_dst  = dio.load_dst(PATH_OUT , 'DST' , 'Events')
    assert len(set(df_penthesilea_dst .event)) == cnt.events_out
    assert len(set(df_penthesilea_reco.event)) == cnt.events_out

    assert_dataframes_close(df_penthesilea_reco[columns], DF_TRUE_RECO[columns],
                            check_dtype=False           , rtol=1e-4            )
    assert_dataframes_close(df_penthesilea_dst          , DF_TRUE_DST          ,
                            check_dtype=False           , rtol=1e-4            )


@ignore_warning.no_config_group
def test_penthesilea_filter_events(config_tmpdir, Kr_pmaps_run4628_filename):
    PATH_IN =  Kr_pmaps_run4628_filename

    PATH_OUT = os.path.join(config_tmpdir, 'KrDST_4628.h5')
    nrequired = 50
    conf = configure('dummy invisible_cities/config/penthesilea.conf'.split())
    conf.update(dict(run_number = 4628,
                     files_in   = PATH_IN,
                     file_out   = PATH_OUT,

                     drift_v     =      2 * units.mm / units.mus,
                     s1_nmin     =      1,
                     s1_nmax     =      1,
                     s1_emin     =      1 * units.pes,
                     s1_emax     =     30 * units.pes,
                     s1_wmin     =    100 * units.ns,
                     s1_wmax     =    300 * units.ns,
                     s1_hmin     =      1 * units.pes,
                     s1_hmax     =      5 * units.pes,
                     s1_ethr     =    0.5 * units.pes,
                     s2_nmin     =      1,
                     s2_nmax     =      2,
                     s2_emin     =    1e3 * units.pes,
                     s2_emax     =    1e4 * units.pes,
                     s2_wmin     =      2 * units.mus,
                     s2_wmax     =     20 * units.mus,
                     s2_hmin     =    1e3 * units.pes,
                     s2_hmax     =    1e5 * units.pes,
                     s2_ethr     =      1 * units.pes,
                     s2_nsipmmin =      5,
                     s2_nsipmmax =     30,
                     event_range = (0, nrequired)))

    events_pass_reco = ([ 1]*21 + [ 4]*15 + [10]*16 + [15]*19 +
                        [19]*17 + [20]*19 + [21]*15 + [26]*23 +
                        [29]*22 + [33]*14 + [41]*18 + [43]*18 +
                        [45]*13 + [46]*18)
    peak_pass_reco   = [int(in_range(i, 138, 145))
                        for i in range(248)]

    cnt      = penthesilea(**conf)
    nevt_in  = cnt.events_in
    nevt_out = cnt.events_out
    assert nrequired    == nevt_in
    assert nevt_out     == len(set(events_pass_reco))

    df_penthesilea_reco = dio.load_dst(PATH_OUT , 'RECO', 'Events')
    assert len(set(df_penthesilea_reco.event.values)) ==   nevt_out
    assert  np.all(df_penthesilea_reco.event.values   == events_pass_reco)
    assert  np.all(df_penthesilea_reco.npeak.values   ==   peak_pass_reco)


    events_pass_dst = [ 1,  4, 10, 15, 19, 20, 21, 26,
                        26, 29, 33, 41, 43, 45, 46]
    s1_peak_pass_dst = [ 0,  0,  0,  0,  0,  0,  0, 0,
                         0,  0,  0,  0,  0,  0,  0]
    s2_peak_pass_dst = [ 0,  0,  0,  0,  0,  0,  0, 0,
                         1,  0,  0,  0,  0,  0,  0]
    assert nevt_out     == len(set(events_pass_dst))
    df_penthesilea_dst  = dio.load_dst(PATH_OUT , 'DST' , 'Events')
    assert len(set(df_penthesilea_dst.event.values)) == nevt_out

    assert np.all(df_penthesilea_dst.event  .values ==  events_pass_dst)
    assert np.all(df_penthesilea_dst.s1_peak.values == s1_peak_pass_dst)
    assert np.all(df_penthesilea_dst.s2_peak.values == s2_peak_pass_dst)


@ignore_warning.no_config_group
def test_penthesilea_threshold_rebin(ICDATADIR, output_tmpdir):
    file_in     = os.path.join(ICDATADIR    ,            "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts.PMP.h5")
    file_out    = os.path.join(output_tmpdir,                   "exact_result_penthesilea_rebin4000.h5")
    true_output = os.path.join(ICDATADIR    , "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts_rebin4000.HDST.h5")

    conf        = configure('dummy invisible_cities/config/penthesilea.conf'.split())
    rebin_thresh = 4000

    conf.update(dict(run_number   =                 -6340,
                     files_in     =               file_in,
                     file_out     =              file_out,
                     event_range  =            all_events,
                     rebin        =          rebin_thresh,
                     rebin_method = RebinMethod.threshold))

    penthesilea(**conf)

    output_dst   = dio.load_dst(file_out   , 'RECO', 'Events')
    expected_dst = dio.load_dst(true_output, 'RECO', 'Events')

    assert len(set(output_dst.event)) == len(set(expected_dst.event))
    assert_dataframes_close(output_dst[columns], expected_dst[columns], check_dtype=False)


@ignore_warning.no_config_group
def test_penthesilea_signal_to_noise(ICDATADIR, output_tmpdir):
    file_in     = os.path.join(ICDATADIR    ,     "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts.PMP.h5")
    file_out    = os.path.join(output_tmpdir,                   "exact_result_penthesilea_SN.h5")
    true_output = os.path.join(ICDATADIR    , "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts_SN.HDST.h5")

    conf        = configure('dummy invisible_cities/config/penthesilea.conf'.split())

    reco_params = dict(Qthr          =  2           ,
                       Qlm           =  6           ,
                       lm_radius     =  0 * units.mm,
                       new_lm_radius = 15 * units.mm,
                       msipm         =  1           )

    conf.update(dict(run_number        =                      -6340,
                     files_in          =                    file_in,
                     file_out          =                   file_out,
                     event_range       =                 all_events,
                     rebin             =                          2,
                     sipm_charge_type  = SiPMCharge.signal_to_noise,
                     slice_reco_params =                reco_params))

    penthesilea(**conf)

    output_dst   = dio.load_dst(file_out   , 'RECO', 'Events')
    expected_dst = dio.load_dst(true_output, 'RECO', 'Events')

    assert len(set(output_dst.event)) == len(set(expected_dst.event))
    assert_dataframes_close(output_dst[columns], expected_dst[columns], check_dtype=False)


@ignore_warning.no_config_group
@mark.serial
def test_penthesilea_produces_mcinfo(KrMC_pmaps_filename, KrMC_hdst, config_tmpdir):
    PATH_IN   = KrMC_pmaps_filename
    PATH_OUT  = os.path.join(config_tmpdir, "Kr_HDST_with_MC.h5")
    conf      = configure('dummy invisible_cities/config/penthesilea.conf'.split())
    nevt_req  = 10

    conf.update(dict(files_in        = PATH_IN,
                     file_out        = PATH_OUT,
                     event_range     = nevt_req,
                     **KrMC_hdst.config))

    penthesilea(**conf)

    with tb.open_file(PATH_OUT) as h5out:
        assert "MC"           in h5out.root
        assert "MC/hits"      in h5out.root
        assert "MC/particles" in h5out.root


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
def KrMC_true_hits(KrMC_pmaps_filename, KrMC_hdst):
    pmap_filename = KrMC_pmaps_filename
    hdst_filename = KrMC_hdst .file_info.filename
    pmap_mctracks = load_mchits_df(pmap_filename)
    hdst_mctracks = load_mchits_df(hdst_filename)
    return mcs_data(pmap_mctracks, hdst_mctracks)

@pytest.fixture(scope='session')
def KrMC_hdst(ICDATADIR):
    test_file = "Kr83_nexus_v5_03_00_ACTIVE_7bar_10evts_HDST.h5"
    test_file = os.path.join(ICDATADIR, test_file)

    # ZANODE = -9.425 * units.mm # unsure if this is somehow useful

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
                         global_reco_algo   = XYReco.barycenter,
                         global_reco_params = dict(Qthr = 1 * units.pes),
                         slice_reco_algo    = XYReco.corona,
                         slice_reco_params  =   dict(
                             Qthr           =   2  * units.pes,
                             Qlm            =   5  * units.pes,
                             lm_radius      =   0  * units.mm ,
                             new_lm_radius  =   15 * units.mm ,
                             msipm          =   1             ))

    event = [0] * 7 + [1] * 10 + [2] * 8 + [3] * 5 + [4] * 8 + [5] * 9 + [6] * 7 + [7] * 7 + [9] * 6
    time  = [0]  * 67
    peak  = [0]  * 67

    X     = [137.02335083, 136.05361214, 155, 137.50739563,
             139.44340629, 135, NN,
             -136.91523372, -138.31812254, -137.40236334, -139.36096871,
             -136.614188  , -135.        , -155.        , -137.13894421,
             -135.        ,    NN,
             96.69981723,  97.48205345,  96.84759934, 115.        ,
             96.87320369, 115.        , 101.92102321,  97.29559249,
             43.04750178, 46.37438754, 45.18567591, 45.30916066,  NN,
             -56.51582522, -56.16605278, -54.00613888, -55.77326839,
             -54.43059006, -52.08436199, -54.76790039,   NN,
             106.9129152 , 108.02149081, 125.        , 111.23591207,
             95.        , 109.36687514, 125.        , -35.        ,
             NN,
             -66.81829731, -60.99443543, -65.21616666, -56.82709265,
             -85.        , -66.38902019, -67.16981495,
             146.62612941, 145.98052086, 144.56275403, 146.57670316,
             165.        , 143.75471749,   NN,
             112.68567761, 113.05236723,  95.        , 112.18326138,
             95.        , 111.13948562]

    Y     = [123.3554201 , 124.21305414, 130.32028088, 125.13468301,
             125.723091  , 105.        ,   NN,
             -102.20454509,  -85.      , -101.19527484,  -85.    ,
             -98.18372292, -115.       ,  -97.50736874,  -97.65435292,
             -115.        ,   NN,
             97.83535504,  96.71162732, 115.        ,  99.94110169,
             96.96828871,  91.9964654 ,  96.47442012,  97.41483565,
             -135.46032693, -135.78741482, -135.45706672, -136.84192389,
             NN,
             -91.93474203, -92.22587474, -75.        , -91.79293263,
             -92.05845326, -75.        , -93.17777252,   NN,
             60.37973152, 55.49979777, 58.7016448 , 56.05515802,
             52.82336998, 57.04430991, 61.02136179, 65.        ,
             NN,
             131.35172348, 115.        , 131.55083652, 116.93600131,
             125.        , 131.38450477, 127.55032531,
             86.4185202 , 85.50276576, 85.08681108, 85.63406367,
             92.12569852, 87.37079899,  NN,
             -41.26775923, -43.95642973, -42.01248503, -43.6128179 ,
             -39.43319563, -41.83429664]

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

    Ec    = [-1] * len(E)

    Xpeak = [137.83250525] * 7 + [-137.61807157] * 10 + [98.89289499] * 8 + [45.59115902] * 5 + [-55.56241282] * 8 + [108.79125728] * 9 + [-65.40441505] * 7 + [146.19707713] * 7 + [111.82481627] * 6

    Ypeak = [124.29040397] * 7 + [-99.36910381] * 10 + [97.3038981] * 8 + [-136.05131239] * 5 + [-91.45151521] * 8 + [56.89100565] * 9 + [130.07815717] * 7 + [85.995134] * 7 + [-43.18583022] * 6

    df_true = DataFrame({"event": event,
                         "time" : time ,
                         "npeak": peak ,
                         "Xpeak": Xpeak,
                         "Ypeak": Ypeak,
                         "X"    : X,
                         "Y"    : Y,
                         "Z"    : Z,
                         "Q"    : Q,
                         "E"    : E,
                         "Ec"   : Ec,
                         })

    df_read = load_dst(test_file,
                       group = group,
                       node  = node)

    return dst_data(tbl_data(test_file, group, node),
                    configuration,
                    df_read,
                    df_true)



@mark.serial
def test_penthesilea_true_hits_are_correct(KrMC_true_hits, config_tmpdir):
    penthesilea_output_path = os.path.join(config_tmpdir,'Kr_HDST_with_MC.h5')
    penthesilea_evts        = load_mchits_df(penthesilea_output_path)
    true_evts               = KrMC_true_hits.hdst

    assert_dataframes_close(penthesilea_evts, true_evts)


@ignore_warning.no_config_group
def test_penthesilea_read_multiple_files(ICDATADIR, output_tmpdir):
    file_in     = os.path.join(ICDATADIR                                       ,
                               "Tl_v1_00_05_nexus_v5_02_08_7bar_pmaps_5evts_*.h5")
    file_out    = os.path.join(output_tmpdir                            ,
                               "Tl_v1_00_05_nexus_v5_02_08_7bar_hits.h5")

    nrequired = 10
    conf = configure('dummy invisible_cities/config/penthesilea.conf'.split())
    conf.update(dict(run_number = -4735,
                 files_in       = file_in,
                 file_out       = file_out,
                 event_range    = (0, nrequired)))

    penthesilea(**conf)

    first_file  = os.path.join(ICDATADIR    ,
                               "Tl_v1_00_05_nexus_v5_02_08_7bar_pmaps_5evts_0.h5")
    second_file = os.path.join(ICDATADIR    ,
                               "Tl_v1_00_05_nexus_v5_02_08_7bar_pmaps_5evts_1.h5")

    particles_in1 = load_mcparticles_df( first_file)
    hits_in1      = load_mchits_df     ( first_file)
    particles_in2 = load_mcparticles_df(second_file)
    hits_in2      = load_mchits_df     (second_file)
    particles_out = load_mcparticles_df(   file_out)
    hits_out      = load_mchits_df     (   file_out)

    # back to front beacause that's how the events are.
    evt_in  = np.concatenate([hits_in2.index.levels[0],
                              hits_in1.index.levels[0]])
    first_event_out = 4
    evt_out = hits_out.index.levels[0]
    assert all(evt_in[first_event_out:] == evt_out)

    all_hit_in      = pd.concat([hits_in1     ,      hits_in2])
    assert_dataframes_close(all_hit_in.loc[evt_in[first_event_out:]],
                            hits_out                                )
    all_particle_in = pd.concat([particles_in1, particles_in2])
    assert_dataframes_close(all_particle_in.loc[evt_in[first_event_out:]],
                            particles_out                                )


@ignore_warning.no_config_group
def test_penthesilea_exact_result(ICDATADIR, output_tmpdir):
    file_in     = os.path.join(ICDATADIR                                     ,
                               "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts.PMP.h5")
    file_out    = os.path.join(output_tmpdir                ,
                               "exact_result_penthesilea.h5")
    true_output = os.path.join(ICDATADIR                                      ,
                               "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts.NEWMC.HDST.h5")

    conf = configure("penthesilea invisible_cities/config/penthesilea.conf".split())
    conf.update(dict(run_number   = -6340,
                     files_in     = file_in,
                     file_out     = file_out,
                     event_range  = all_events))

    penthesilea(**conf)

    tables = ("RECO/Events"     , "DST/Events"   , "Filters/s12_selector",
              "MC/event_mapping", "MC/generators", "MC/hits", "MC/particles")
    with tb.open_file(true_output)  as true_output_file:
        with tb.open_file(file_out) as      output_file:
            for table in tables:
                assert hasattr(output_file.root, table)
                got      = getattr(     output_file.root, table)
                expected = getattr(true_output_file.root, table)
                assert_tables_equality(got, expected)

@ignore_warning.no_config_group
def test_penthesilea_exact_result_noS1(ICDATADIR, output_tmpdir):
    file_in     = os.path.join(ICDATADIR                                      ,
                               "Kr83_nexus_v5_03_00_ACTIVE_7bar_10evts_PMP.h5")
    file_out    = os.path.join(output_tmpdir                     ,
                               "exact_result_penthesilea_noS1.h5")
    true_output = os.path.join(ICDATADIR                                     ,
                               "Kr83_nexus_v5_03_00_ACTIVE_7bar_noS1.NEWMC.HDST.h5")

    conf = configure("penthesilea invisible_cities/config/penthesilea.conf".split())
    conf.update(dict(run_number   =      -6340,
                     files_in     =    file_in,
                     file_out     =   file_out,
                     event_range  = all_events,
                     s1_nmin      =          0,
                     s1_emin      =         10 * units.pes))

    penthesilea(**conf)

    tables = ("RECO/Events"     , "DST/Events"   , "Filters/s12_selector",
              "MC/event_mapping", "MC/generators", "MC/hits", "MC/particles")
    with tb.open_file(true_output)  as true_output_file:
        with tb.open_file(file_out) as      output_file:
            for table in tables:
                assert hasattr(output_file.root, table)
                got      = getattr(     output_file.root, table)
                expected = getattr(true_output_file.root, table)
                assert_tables_equality(got, expected)

@pytest.fixture(scope='session')
def Xe2nu_pmaps_mc_filename(ICDATADIR):
    filename = os.path.join(ICDATADIR, "Xe2nu_NEW_v1_05_02_nexus_v5_03_08_ACTIVE_10.2bar_run4.0_0_pmaps.h5")
    return filename

# test for PR 628
@ignore_warning.no_config_group
def test_penthesilea_xyrecofail(config_tmpdir, Xe2nu_pmaps_mc_filename):
    PATH_IN =  Xe2nu_pmaps_mc_filename

    PATH_OUT = os.path.join(config_tmpdir, 'Xe2nu_2nu_hdst.h5')
    conf = configure('dummy invisible_cities/config/penthesilea.conf'.split())
    conf.update(dict(run_number = -6000,
                     files_in   = PATH_IN,
                     file_out   = PATH_OUT,

                     drift_v     =      1 * units.mm / units.mus,
                     s1_nmin     =      1,
                     s1_nmax     =      1,
                     s1_emin     =      0 * units.pes,
                     s1_emax     =   1e+6 * units.pes,
                     s1_wmin     =      1 * units.ns,
                     s1_wmax     =   1.e6 * units.ns,
                     s1_hmin     =      0 * units.pes,
                     s1_hmax     =   1e+6 * units.pes,
                     s1_ethr     =    0.5 * units.pes,
                     s2_nmin     =      1,
                     s2_nmax     =    100,
                     s2_emin     =      0 * units.pes,
                     s2_emax     =    1e8 * units.pes,
                     s2_wmin     =    2.5 * units.mus,
                     s2_wmax     =     10 * units.ms,
                     s2_hmin     =      0 * units.pes,
                     s2_hmax     =    1e6 * units.pes,
                     s2_ethr     =      1 * units.pes,
                     s2_nsipmmin =      1,
                     s2_nsipmmax =   2000,
                     rebin       =      2,
                     global_reco_params = dict(),
                     slice_reco_params = dict(
                         Qthr            =  30 * units.pes,
                         Qlm             =  30 * units.pes,
                         lm_radius       =  0 * units.mm ,
                         new_lm_radius   =  0 * units.mm ,
                         msipm = 1 ),
                     event_range = (826, 827)))

    # check it runs
    penthesilea(**conf)


@ignore_warning.no_config_group
def test_penthesilea_empty_input_file(config_tmpdir, ICDATADIR):
    # Penthesilea must run on an empty file without raising any exception
    # The input file has the complete structure of a PMAP but no events.

    PATH_IN  = os.path.join(ICDATADIR    , 'empty_pmaps.h5')
    PATH_OUT = os.path.join(config_tmpdir, 'empty_hdst.h5')

    conf = configure('dummy invisible_cities/config/penthesilea.conf'.split())
    conf.update(dict(files_in      = PATH_IN,
                     file_out      = PATH_OUT))

    # Warning expected since no MC tables present.
    # Suppress since irrelevant in test.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        penthesilea(**conf)


@ignore_warning.no_config_group
def test_penthesilea_global_xyreco_bias(config_tmpdir, ICDATADIR):
    # Make sure the global_reco_params are used for the columns Xpeak/Ypeak
    # This is done by running over an event that has many spurious sipm hits
    # and only one sipm with signal over the threshold. If the threshold
    # is used, the reconstructed position will match the sipm position.
    # If the threshold is not used the position will be biased.

    PATH_IN  = os.path.join(ICDATADIR    , 'fake_pmap_nothreshold_barycenter_bias.h5')
    PATH_OUT = os.path.join(config_tmpdir, 'fake_hdst_nothreshold_barycenter_bias.h5')

    conf = configure('dummy invisible_cities/config/penthesilea.conf'.split())
    conf.update(dict( files_in    = PATH_IN
                    , file_out    = PATH_OUT
                    , rebin       =  100
                    , run_number  = 7000
                    , s2_nsipmmax =  200
                    ))

    conf["global_reco_params"].update(dict(Qthr=5))

    penthesilea(**conf)

    output_dst = dio.load_dst(PATH_OUT, 'RECO', 'Events')
    assert np.all(output_dst.Xpeak == -5)
    assert np.all(output_dst.Ypeak == -5)


@ignore_warning.no_config_group
@mark.parametrize('flags value counter'.split(),
                  (('-e all'   , 10, 'events_in'), # 10 events in the file
                   ('-e   9'   ,  9, 'events_in'), # [ 0,  9) -> 9
                   ('-e 5 9'   ,  4, 'events_in'), # [ 5,  9) -> 4
                   ('-e 2 last',  8, 'events_in'), # events [2, 10) -> 8
                  ))
def test_config_penthesilea_counters(config_tmpdir, KrMC_pmaps_filename, flags, value, counter):
    input_filename  = KrMC_pmaps_filename
    config_filename = 'invisible_cities/config/penthesilea.conf'
    flags_wo_spaces = flags.replace(" ", "_")
    output_filename = os.path.join(config_tmpdir, f'penthesilea_counters_{flags_wo_spaces}.h5')

    argv = f'penthesilea {config_filename} -i {input_filename} -o {output_filename} {flags}'.split()
    counters = penthesilea(**configure(argv))
    assert getattr(counters, counter) == value
