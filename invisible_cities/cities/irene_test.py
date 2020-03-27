import os

import tables as tb
import numpy  as np
import pandas as pd

from pytest import raises
from pytest import mark
from pytest import fixture

from .. core                import system_of_units as units
from .. core.configure      import             all as all_events
from .. core.configure      import configure
from .. core.testing_utils  import exactly
from .. core.testing_utils  import assert_tables_equality
from .. types.ic_types      import minmax
from .. io.run_and_event_io import read_run_and_event
from .. evm.ic_containers   import S12Params as S12P
from .. io.mcinfo_io        import load_mcparticles_df
from .. io.mcinfo_io        import load_mchits_df

from .. database.load_db    import DetDB
from .. io      .pmaps_io   import load_pmaps

from .  irene import irene


@fixture(scope='module')
def s12params():
    s1par = S12P(time   = minmax(min =  99 * units.mus,
                                 max = 101 * units.mus),
                 length = minmax(min =   4,
                                 max =  20,),
                 stride              =   4,
                 rebin_stride        =   1)

    s2par = S12P(time   = minmax(min =    101 * units.mus,
                                 max =   1199 * units.mus),
                 length = minmax(min =     80,
                                 max = 200000),
                 stride              =     40,
                 rebin_stride        =     40)
    return s1par, s2par


def unpack_s12params(s12params):
    s1par, s2par = s12params
    return dict(s1_tmin         = s1par.time.min,
                s1_tmax         = s1par.time.max,
                s1_lmin         = s1par.length.min,
                s1_lmax         = s1par.length.max,

                s2_tmin         = s2par.time.min,
                s2_tmax         = s2par.time.max,
                s2_rebin_stride = s2par.rebin_stride,
                s2_lmin         = s2par.length.min,
                s2_lmax         = s2par.length.max)


@mark.slow
@mark.parametrize("thr_sipm_type thr_sipm_value".split(),
                  (("common"    , 3.5 ),
                   ("individual", 0.99)))
def test_irene_electrons_40keV(config_tmpdir, ICDATADIR, s12params,
                               thr_sipm_type, thr_sipm_value):
    # NB: avoid taking defaults for PATH_IN and PATH_OUT
    # since they are in general test-specific
    # NB: avoid taking defaults for run number (test-specific)

    PATH_IN  = os.path.join(ICDATADIR                             ,
                            'electrons_40keV_ACTIVE_10evts_RWF.h5')
    PATH_OUT = os.path.join(config_tmpdir                         ,
                            'electrons_40keV_ACTIVE_10evts_CWF.h5')

    nrequired  = 2

    conf = configure('dummy invisible_cities/config/irene.conf'.split())
    conf.update(dict(detector_db = DetDB.new,
                     run_number    = 0,
                     files_in      = PATH_IN,
                     file_out      = PATH_OUT,
                     event_range   = (0, nrequired),
                     thr_sipm_type = thr_sipm_type,
                     thr_sipm      = thr_sipm_value,
                     **unpack_s12params(s12params)))

    cnt = irene(**conf)

    nactual = cnt.events_in
    assert nrequired == nactual

    mcparticles_in  = load_mcparticles_df( PATH_IN)
    mcparticles_out = load_mcparticles_df(PATH_OUT)
    pd.testing.assert_frame_equal(mcparticles_in, mcparticles_out)
    with tb.open_file(PATH_IN , mode='r') as h5in, \
         tb.open_file(PATH_OUT, mode='r') as h5out:

            # check events numbers & timestamps
            evts_in  = h5in .root.Run.events[:nactual].astype([('evt_number', '<i4'), ('timestamp', '<u8')])
            evts_out = h5out.root.Run.events[:nactual]
            np.testing.assert_array_equal(evts_in, evts_out)


@mark.slow
@mark.serial
def test_irene_run_2983(config_tmpdir, ICDIR, s12params):
    """Run Irene. Write an output file."""

    # NB: the input file has 5 events. The maximum value for 'n'
    # in the IRENE parameters is 5, but it can run with a smaller values
    # (eg, 2) to speed the test.

    PATH_IN  = os.path.join(ICDIR, 'database/test_data/', 'run_2983.h5')
    PATH_OUT = os.path.join(config_tmpdir, 'run_2983_pmaps.h5')

    nrequired = 2

    conf = configure('dummy invisible_cities/config/irene.conf'.split())
    conf.update(dict(run_number  = 2983,
                     files_in    = PATH_IN,
                     file_out    = PATH_OUT,
                     event_range = (0, nrequired),
                     **unpack_s12params(s12params)))

    cnt = irene(**conf)
    assert nrequired == cnt.events_in


@mark.slow # not slow itself, but depends on a slow test
@mark.serial
def test_irene_runinfo_run_2983(config_tmpdir, ICDATADIR):
    """Read back the file written by previous test. Check runinfo."""

    # NB: the input file has 5 events. The maximum value for 'n'
    # in the IRENE parameters is 5, but it can run with a smaller values
    # (eg, 2) to speed the test. BUT NB, this has to be propagated to this
    # test, eg. h5in .root.Run.events[0:3] if one has run 2 events.

    PATH_IN  = os.path.join(ICDATADIR    , 'run_2983.h5')
    PATH_OUT = os.path.join(config_tmpdir, 'run_2983_pmaps.h5')


    with tb.open_file(PATH_IN, mode='r') as h5in:
        valid_events = (0,)

        evts_in = h5in.root.Run.events.cols.evt_number[valid_events]
        ts_in   = h5in.root.Run.events.cols.timestamp [valid_events]

        rundf, evtdf = read_run_and_event(PATH_OUT)
        evts_out     = evtdf.evt_number.values
        ts_out       = evtdf.timestamp.values

        np.testing.assert_array_equal(evts_in, evts_out)
        np.testing.assert_array_equal(  ts_in,   ts_out)

        run_number_in  = h5in.root.Run.runInfo[:][0][0]
        run_number_out = rundf.run_number[0]

        assert run_number_in == run_number_out


@mark.serial
@mark.slow
def test_irene_output_file_structure(config_tmpdir):
    PATH_OUT = os.path.join(config_tmpdir, 'run_2983_pmaps.h5')

    with tb.open_file(PATH_OUT) as h5out:
        assert "PMAPS"        in h5out.root
        assert "Run"          in h5out.root
        #assert "DeconvParams" in h5out.root # Not there in liquid city
        assert "S1"           in h5out.root.PMAPS
        assert "S2"           in h5out.root.PMAPS
        assert "S2Si"         in h5out.root.PMAPS
        assert "events"       in h5out.root.Run
        assert "runInfo"      in h5out.root.Run


def test_empty_events_issue_81(config_tmpdir, ICDATADIR, s12params):
    # NB: explicit PATH_OUT
    PATH_IN  = os.path.join(ICDATADIR    , 'irene_bug_Kr_ACTIVE_7bar_RWF.h5')
    PATH_OUT = os.path.join(config_tmpdir, 'irene_bug_Kr_ACTIVE_7bar_CWF.h5')

    nrequired = 10

    conf = configure('dummy invisible_cities/config/irene.conf'.split())
    conf.update(dict(run_number   = 0,
                     files_in     = PATH_IN,
                     file_out     = PATH_OUT,
                     event_range = (0, nrequired),
                     **unpack_s12params(s12params)))

    cnt = irene(**conf)

    assert cnt.events_in           == 1
    assert cnt.events_out          == 0
    assert cnt.  over_thr.n_failed == 1


def test_irene_empty_pmap_output(ICDATADIR, output_tmpdir, s12params):
    file_in  = os.path.join(ICDATADIR    , "kr_rwf_0_0_7bar_NEXT_v1_00_05_v0.9.2_20171011_krmc_diomira_3evt.h5")
    file_out = os.path.join(output_tmpdir, "kr_rwf_0_0_7bar_NEXT_v1_00_05_v0.9.2_20171011_pmaps_3evt.h5")

    nrequired = 3
    conf = configure('dummy invisible_cities/config/irene.conf'.split())
    conf.update(dict(run_number  = 4714,
                     files_in    = file_in,
                     file_out    = file_out,
                     event_range = (0, nrequired),
                     **unpack_s12params(s12params))) # s12params are just dummy values in this test

    cnt = irene(**conf)

    assert cnt.events_in           == 3
    assert cnt.  over_thr.n_failed == 0

    with tb.open_file(file_in) as fin:
        with tb.open_file(file_out) as fout:
            got      = fout.root.Run.events.cols.evt_number[:]
            expected = fin .root.Run.events.cols.evt_number[::2] # skip event in the middle
            assert got == exactly(expected)


def test_irene_read_multiple_files(ICDATADIR, output_tmpdir, s12params):
    file_in     = os.path.join(ICDATADIR                                       ,
                               "Tl_v1_00_05_nexus_v5_02_08_7bar_RWF_5evts_*.h5")
    file_out    = os.path.join(output_tmpdir                                   ,
                               "Tl_v1_00_05_nexus_v5_02_08_7bar_pmaps_10evts.h5")

    nevents_per_file = 5

    nrequired = 10
    conf = configure('dummy invisible_cities/config/irene.conf'.split())
    conf.update(dict(run_number  = -4735,
                     files_in    = file_in,
                     file_out    = file_out,
                     event_range = (0, nrequired),
                     **unpack_s12params(s12params))) # s12params are just dummy values in this test

    irene(**conf)

    first_file  = os.path.join(ICDATADIR                                       ,
                               "Tl_v1_00_05_nexus_v5_02_08_7bar_RWF_5evts_0.h5")
    second_file = os.path.join(ICDATADIR                                       ,
                               "Tl_v1_00_05_nexus_v5_02_08_7bar_RWF_5evts_1.h5")

    particles_in1 = load_mcparticles_df( first_file)
    hits_in1      = load_mchits_df     ( first_file)
    particles_in2 = load_mcparticles_df(second_file)
    hits_in2      = load_mchits_df     (second_file)
    particles_out = load_mcparticles_df(   file_out)
    hits_out      = load_mchits_df     (   file_out)

    # back to front cos the events are that way for some reason
    evt_in  = np.concatenate([hits_in2.index.levels[0],
                              hits_in1.index.levels[0]])
    evt_out = hits_out.index.levels[0]
    assert all(evt_in == evt_out)

    all_hit_in      = pd.concat([hits_in1     ,      hits_in2])
    pd.testing.assert_frame_equal(all_hit_in, hits_out)
    all_particle_in = pd.concat([particles_in1, particles_in2])
    pd.testing.assert_frame_equal(all_particle_in, particles_out)


def test_irene_trigger_type(config_tmpdir, ICDATADIR, s12params):
    PATH_IN  = os.path.join(ICDATADIR    ,       '6229_000_trg_type.h5')
    PATH_OUT = os.path.join(config_tmpdir, 'pmaps_6229_000_trg_type.h5')

    nrequired  = 1

    run_number = 6229
    conf = configure('dummy invisible_cities/config/irene.conf'.split())
    conf.update(dict(run_number  = run_number,
                     files_in    = PATH_IN,
                     file_out    = PATH_OUT,
                     event_range = (0, nrequired),
                     **unpack_s12params(s12params)))

    cnt = irene(**conf)
    assert cnt.events_in == nrequired

    with tb.open_file(PATH_IN , mode='r') as h5in, \
         tb.open_file(PATH_OUT, mode='r') as h5out:
            nrow        = 0
            trigger_in  = h5in .root.Trigger.trigger[nrow]
            trigger_out = h5out.root.Trigger.trigger[nrow]
            np.testing.assert_array_equal(trigger_in, trigger_out)


def test_irene_trigger_channels(config_tmpdir, ICDATADIR, s12params):
    PATH_IN  = os.path.join(ICDATADIR    ,       '6229_000_trg_channels.h5')
    PATH_OUT = os.path.join(config_tmpdir, 'pmaps_6229_000_trg_channels.h5')

    nrequired  = 1
    run_number = 6229

    conf = configure('dummy invisible_cities/config/irene.conf'.split())
    conf.update(dict(run_number  = run_number,
                     files_in    = PATH_IN,
                     file_out    = PATH_OUT,
                     event_range = (0, nrequired),
                     **unpack_s12params(s12params)))

    cnt = irene(**conf)
    assert cnt.events_in == nrequired

    with tb.open_file(PATH_IN,  mode='r') as h5in, \
         tb.open_file(PATH_OUT, mode='r') as h5out:
            nrow        = 0
            trigger_in  = h5in .root.Trigger.events[nrow]
            trigger_out = h5out.root.Trigger.events[nrow]
            np.testing.assert_array_equal(trigger_in, trigger_out)


def test_irene_exact_result(ICDATADIR, output_tmpdir):
    file_in     = os.path.join(ICDATADIR    , "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts.RWF.h5")
    file_out    = os.path.join(output_tmpdir,                        "exact_result_irene.h5")
    true_output = os.path.join(ICDATADIR    , "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts.PMP.h5")

    conf = configure("irene invisible_cities/config/irene.conf".split())
    conf.update(dict(run_number   = -6340,
                     files_in     = file_in,
                     file_out     = file_out,
                     event_range  = all_events))

    irene(**conf)

    ## tables = (     "MC/extents"    ,      "MC/hits"      ,    "MC/particles", "MC/generators",
    ##             "PMAPS/S1"         ,   "PMAPS/S2"        , "PMAPS/S2Si"     ,
    ##             "PMAPS/S1Pmt"      ,   "PMAPS/S2Pmt"     ,
    ##               "Run/events"     ,     "Run/runInfo"   ,
    ##           "Trigger/events"     , "Trigger/trigger"   ,
    ##           "Filters/s12_indices", "Filters/empty_pmap")
    tables = (  "PMAPS/S1"         ,   "PMAPS/S2"        , "PMAPS/S2Si"     ,
                "PMAPS/S1Pmt"      ,   "PMAPS/S2Pmt"     ,
                  "Run/events"     ,     "Run/runInfo"   ,
              "Trigger/events"     , "Trigger/trigger"   ,
              "Filters/s12_indices", "Filters/empty_pmap")
    with tb.open_file(true_output)  as true_output_file:
        with tb.open_file(file_out) as      output_file:
            for table in tables:
                got      = getattr(     output_file.root, table)
                expected = getattr(true_output_file.root, table)
                assert_tables_equality(got, expected)


def test_irene_filters_empty_pmaps(ICDATADIR, output_tmpdir):
    file_in  = os.path.join(ICDATADIR                                     ,
                            "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts.RWF.h5")
    file_out = os.path.join(output_tmpdir, "test_irene_filters_empty_pmaps.h5")

    conf = configure("irene invisible_cities/config/irene.conf".split())
    conf.update(dict(run_number   = -6340,
                     files_in     = file_in,
                     file_out     = file_out,
                     event_range  = all_events,
                     # Search for peaks where there are
                     # not any so the produced is empty
                     s1_tmin      = 0 * units.mus,
                     s1_tmax      = 1 * units.mus,
                     s2_tmin      = 0 * units.mus,
                     s2_tmax      = 1 * units.mus))

    cnt = irene(**conf)

    assert cnt.full_pmap.n_failed == 3

    ## tables = (     "MC/extents",      "MC/hits"   ,    "MC/particles", "MC/generators",
    ##             "PMAPS/S1"     ,   "PMAPS/S2"     , "PMAPS/S2Si"     ,
    ##             "PMAPS/S1Pmt"  ,   "PMAPS/S2Pmt"  ,
    ##               "Run/events" ,     "Run/runInfo",
    ##           "Trigger/events" , "Trigger/trigger")
    tables = (  "PMAPS/S1"     ,   "PMAPS/S2"     , "PMAPS/S2Si"     ,
                "PMAPS/S1Pmt"  ,   "PMAPS/S2Pmt"  ,
                  "Run/events" ,     "Run/runInfo",
              "Trigger/events" , "Trigger/trigger")
    with tb.open_file(file_out) as      output_file:
        for table_name in tables:
            table = getattr(output_file.root, table_name)
            assert table.nrows == 0


def test_irene_empty_input_file(config_tmpdir, ICDATADIR):
    # Irene must run on an empty file without raising any exception
    # The input file has the complete structure of a PMAP but no events.

    PATH_IN  = os.path.join(ICDATADIR    , 'empty_rwf.h5')
    PATH_OUT = os.path.join(config_tmpdir, 'empty_pmaps.h5')

    conf = configure('dummy invisible_cities/config/irene.conf'.split())
    conf.update(dict(files_in      = PATH_IN,
                     file_out      = PATH_OUT))

    irene(**conf)



def test_irene_sequential_times(config_tmpdir, ICDATADIR):

    PATH_IN  = os.path.join(ICDATADIR    , 'single_evt_nonseqtime_rwf.h5')
    PATH_OUT = os.path.join(config_tmpdir, 'test_pmaps.h5')

    conf = configure('dummy invisible_cities/config/irene.conf'.split())
    conf.update(dict(files_in    = PATH_IN           ,
                     file_out    = PATH_OUT          ,
                     run_number  =   6351            ,
                     n_baseline  =  48000            ,
                     thr_sipm    =      1 * units.pes,
                     s1_tmin     =      0 * units.mus,
                     s1_tmax     =    640 * units.mus,
                     s1_lmin     =      5            ,
                     s1_lmax     =     30            ,
                     s2_tmin     =    645 * units.mus,
                     s2_tmax     =   1300 * units.mus,
                     s2_lmin     =     80            ,
                     s2_lmax     = 200000            ,
                     thr_sipm_s2 =      5 * units.pes))

    irene(**conf)

    pmaps_out = load_pmaps(PATH_OUT)

    assert np.all(np.diff(pmaps_out[3348].s2s[1].times) > 0)

def test_error_when_different_sample_widths(ICDATADIR, config_tmpdir):
    ## Tests that ValueError message is correct
    PATH_IN  = os.path.join(ICDATADIR,                   'run_2983.h5')
    PATH_OUT = os.path.join(config_tmpdir, 'r2983_pmaps_diffparams.h5')
    conf = configure('dummy invisible_cities/config/irene.conf'.split())

    pmt_samp_wid  = 25 * units.ns
    sipm_samp_wid =  2 * units.mus
    conf.update(dict(run_number    = 2983,
                     files_in      = PATH_IN,
                     file_out      = PATH_OUT,
                     pmt_samp_wid  = pmt_samp_wid,
                     sipm_samp_wid = sipm_samp_wid))

    msg  = "Shapes don't match!\n"
    msg += "times has length 6\n"
    msg += "pmts  has length 6 \n"
    msg += "sipms has length 3\n"

    with raises(ValueError) as error:
        irene(**conf)
    assert str(error.value) == msg


def test_irene_other_sample_widths(ICDATADIR, config_tmpdir):
    ## Tests irene works when running with other sample frequencies
    file_in     = os.path.join(ICDATADIR,               'run_2983.h5')
    file_out    = os.path.join(config_tmpdir, 'r2983_pmaps_output.h5')
    true_output = os.path.join(ICDATADIR, 'run_2983_pmaps_2evts_sfreq_5ns_200ns.h5')
    conf = configure('dummy invisible_cities/config/irene.conf'.split())

    pmt_samp_wid  = 5  * units.ns
    sipm_samp_wid = 0.2 * units.mus
    n_events = 2

    conf.update(dict(run_number    = 2983,
                     files_in      = file_in,
                     file_out      = file_out,
                     event_range   = (0, n_events),
                     pmt_samp_wid  = pmt_samp_wid,
                     sipm_samp_wid = sipm_samp_wid))
    irene(**conf)

    tables = (  "PMAPS/S1"         ,   "PMAPS/S2"        , "PMAPS/S2Si"     ,
                "PMAPS/S1Pmt"      ,   "PMAPS/S2Pmt"     ,
                  "Run/events"     ,     "Run/runInfo"   ,
              "Trigger/events"     , "Trigger/trigger"   ,
              "Filters/s12_indices", "Filters/empty_pmap")
    with tb.open_file(true_output)  as true_output_file:
        with tb.open_file(file_out) as      output_file:
            for table in tables:
                got      = getattr(     output_file.root, table)
                expected = getattr(true_output_file.root, table)
                assert_tables_equality(got, expected)
