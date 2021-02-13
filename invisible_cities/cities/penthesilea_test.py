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
from .. core.configure       import configure
from .. core.configure       import all as all_events
from .. io                   import dst_io as dio
from .. io.mcinfo_io         import load_mchits_df
from .. io.mcinfo_io         import load_mcparticles_df

from .  penthesilea          import penthesilea


#in order not to fail direct comparation tests when changing hit attribute we compare only the columns that penthesilea is using
columns = ['event', 'time', 'npeak', 'Xpeak', 'Ypeak', 'nsipm', 'X', 'Y', 'Xrms', 'Yrms', 'Z', 'Q', 'E']

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
                            check_types=False           , rtol=1e-4            )
    assert_dataframes_close(df_penthesilea_dst          , DF_TRUE_DST          ,
                            check_types=False           , rtol=1e-4            )


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

    events_pass_reco = ([ 1]*21 + [ 4]*15 + [10]*16 + [19]*17 +
                        [20]*19 + [21]*15 + [26]*23 + [29]*22 +
                        [33]*14 + [41]*18 + [43]*18 + [45]*13 +
                        [46]*18)
    peak_pass_reco   = [int(in_range(i, 119, 126))
                        for i in range(229)]

    cnt      = penthesilea(**conf)
    nevt_in  = cnt.events_in
    nevt_out = cnt.events_out
    assert nrequired    == nevt_in
    assert nevt_out     == len(set(events_pass_reco))

    df_penthesilea_reco = dio.load_dst(PATH_OUT , 'RECO', 'Events')
    assert len(set(df_penthesilea_reco.event.values)) ==   nevt_out
    assert  np.all(df_penthesilea_reco.event.values   == events_pass_reco)
    assert  np.all(df_penthesilea_reco.npeak.values   ==   peak_pass_reco)


    events_pass_dst = [ 1,  4, 10, 19, 20, 21, 26,
                        26, 29, 33, 41, 43, 45, 46]
    s1_peak_pass_dst = [ 0,  0,  0,  0,  0,  0,  0,
                         0,  0,  0,  0,  0,  0,  0]
    s2_peak_pass_dst = [ 0,  0,  0,  0,  0,  0,  0,
                         1,  0,  0,  0,  0,  0,  0]
    assert nevt_out     == len(set(events_pass_dst))
    df_penthesilea_dst  = dio.load_dst(PATH_OUT , 'DST' , 'Events')
    assert len(set(df_penthesilea_dst.event.values)) == nevt_out

    assert np.all(df_penthesilea_dst.event  .values ==  events_pass_dst)
    assert np.all(df_penthesilea_dst.s1_peak.values == s1_peak_pass_dst)
    assert np.all(df_penthesilea_dst.s2_peak.values == s2_peak_pass_dst)


def test_penthesilea_threshold_rebin(ICDATADIR, output_tmpdir):
    file_in     = os.path.join(ICDATADIR    ,            "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts.PMP.h5")
    file_out    = os.path.join(output_tmpdir,                   "exact_result_penthesilea_rebin4000.h5")
    true_output = os.path.join(ICDATADIR    , "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts_rebin4000.HDST.h5")

    conf        = configure('dummy invisible_cities/config/penthesilea.conf'.split())
    rebin_thresh = 4000

    conf.update(dict(run_number   =        -6340,
                     files_in     =      file_in,
                     file_out     =     file_out,
                     event_range  =   all_events,
                     rebin        = rebin_thresh,
                     rebin_method =  'threshold'))

    penthesilea(**conf)

    output_dst   = dio.load_dst(file_out   , 'RECO', 'Events')
    expected_dst = dio.load_dst(true_output, 'RECO', 'Events')

    assert len(set(output_dst.event)) == len(set(expected_dst.event))
    assert_dataframes_close(output_dst[columns], expected_dst[columns], check_types=False)


def test_penthesilea_signal_to_noise(ICDATADIR, output_tmpdir):
    file_in     = os.path.join(ICDATADIR    ,            "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts.PMP.h5")
    file_out    = os.path.join(output_tmpdir,                   "exact_result_penthesilea_SN.h5")
    true_output = os.path.join(ICDATADIR    , "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts_SN.HDST.h5")

    conf        = configure('dummy invisible_cities/config/penthesilea.conf'.split())

    reco_params = dict(Qthr          =  2           ,
                       Qlm           =  6           ,
                       lm_radius     =  0 * units.mm,
                       new_lm_radius = 15 * units.mm,
                       msipm         =  1           )

    conf.update(dict(run_number        =             -6340,
                     files_in          =           file_in,
                     file_out          =          file_out,
                     event_range       =        all_events,
                     rebin             =                 2,
                     sipm_charge_type  = 'signal_to_noise',
                     slice_reco_params =      reco_params))

    penthesilea(**conf)

    output_dst   = dio.load_dst(file_out   , 'RECO', 'Events')
    expected_dst = dio.load_dst(true_output, 'RECO', 'Events')

    assert len(set(output_dst.event)) == len(set(expected_dst.event))
    assert_dataframes_close(output_dst[columns], expected_dst[columns], check_types=False)


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


@mark.serial
def test_penthesilea_true_hits_are_correct(KrMC_true_hits, config_tmpdir):
    penthesilea_output_path = os.path.join(config_tmpdir,'Kr_HDST_with_MC.h5')
    penthesilea_evts        = load_mchits_df(penthesilea_output_path)
    true_evts               = KrMC_true_hits.hdst

    assert_dataframes_close(penthesilea_evts, true_evts)


@mark.skip(reason='Pandas API change at 1.1.0 interfering with Nix adoption')
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

# test for PR 628
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
