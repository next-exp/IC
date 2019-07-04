import os
from collections import namedtuple

import tables as tb
import numpy  as np

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

from .. database       import load_db
from .. database.load_db       import DetDB

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


@fixture(scope='module')
def job_info_missing_pmts(config_tmpdir, ICDATADIR):
    # Specifies a name for a data configuration file. Also, default number
    # of events set to 1.
    job_info = namedtuple("job_info",
                          "run_number pmt_missing pmt_active input_filename output_filename")

    run_number  = 3366
    pmt_missing = [11]
    pmt_active  = list(filter(lambda x: x not in pmt_missing, range(12)))


    ifilename = os.path.join(ICDATADIR    , 'electrons_40keV_z250_RWF.h5')
    ofilename = os.path.join(config_tmpdir, 'electrons_40keV_z250_pmaps_missing_PMT.h5')

    return job_info(run_number, pmt_missing, pmt_active, ifilename, ofilename)


@mark.slow
@mark.parametrize("thr_sipm_type thr_sipm_value".split(),
                  (("common"    , 3.5 ),
                   ("individual", 0.99)))
def test_irene_electrons_40keV(config_tmpdir, ICDATADIR, s12params,
                               thr_sipm_type, thr_sipm_value):
    # NB: avoid taking defaults for PATH_IN and PATH_OUT
    # since they are in general test-specific
    # NB: avoid taking defaults for run number (test-specific)

    PATH_IN  = os.path.join(ICDATADIR    , 'electrons_40keV_z250_RWF.h5')
    PATH_OUT = os.path.join(config_tmpdir, 'electrons_40keV_z250_CWF.h5')

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

    with tb.open_file(PATH_IN , mode='r') as h5in, \
         tb.open_file(PATH_OUT, mode='r') as h5out:
            nrow = 0
            mctracks_in  = h5in .root.MC.particles[nrow]
            mctracks_out = h5out.root.MC.particles[nrow]
            np.testing.assert_array_equal(mctracks_in, mctracks_out)

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


@mark.skip
def test_irene_electrons_40keV_pmt_active_is_correctly_set(job_info_missing_pmts, s12params):
    "Check that PMT active correctly describes the PMT configuration of the detector"
    nrequired = 1
    conf = configure('dummy invisible_cities/config/irene.conf'.split())
    conf.update(dict(run_number  =  job_info_missing_pmts.run_number,
                     files_in    =  job_info_missing_pmts. input_filename,
                     file_out    =  job_info_missing_pmts.output_filename,
                     event_range = (0, nrequired),
                     **unpack_s12params(s12params))) # s12params are just dummy values in this test

    irene = Irene(**conf)

    assert irene.pmt_active == job_info_missing_pmts.pmt_active


@mark.skip
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
    file_in     = os.path.join(ICDATADIR    , "Tl_v1_00_05_nexus_v5_02_08_7bar_RWF_5evts_*.h5")
    file_out    = os.path.join(output_tmpdir, "Tl_v1_00_05_nexus_v5_02_08_7bar_pmaps_10evts.h5")
    second_file = os.path.join(ICDATADIR    , "Tl_v1_00_05_nexus_v5_02_08_7bar_RWF_5evts_1.h5")

    nevents_per_file = 5

    nrequired = 10
    conf = configure('dummy invisible_cities/config/irene.conf'.split())
    conf.update(dict(run_number  = -4735,
                     files_in    = file_in,
                     file_out    = file_out,
                     event_range = (0, nrequired),
                     **unpack_s12params(s12params))) # s12params are just dummy values in this test

    irene(**conf)

    with tb.open_file(file_out) as h5out:
        last_particle_list = h5out.root.MC.extents[:]['last_particle']
        last_hit_list      = h5out.root.MC.extents[:]['last_hit'     ]

        assert all(x<y for x, y in zip(last_particle_list, last_particle_list[1:]))
        assert all(x<y for x, y in zip(last_hit_list     , last_hit_list     [1:]))

        with tb.open_file(second_file) as h5second:
            nparticles_in_first_event = h5second.root.MC.extents[0]['last_particle'] + 1
            nhits_in_first_event      = h5second.root.MC.extents[0]['last_hit'     ] + 1

            assert last_particle_list[nevents_per_file] - last_particle_list[nevents_per_file - 1] == nparticles_in_first_event
            assert last_hit_list     [nevents_per_file] - last_hit_list     [nevents_per_file - 1] == nhits_in_first_event


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


@mark.skip(reason="Trigger split not implemented in liquid cities")
def test_irene_split_trigger(config_tmpdir, ICDATADIR, s12params):
    PATH_IN   = os.path.join(ICDATADIR    , '6229_000_split_trigger.h5')
    PATH_OUT1 = os.path.join(config_tmpdir, 'pmaps_6229_000_trg1.h5')
    PATH_OUT2 = os.path.join(config_tmpdir, 'pmaps_6229_000_trg2.h5')

    nrequired  = 3
    run_number = 6229

    conf = configure('dummy invisible_cities/config/irene.conf'.split())
    conf.update(dict(run_number     = run_number,
                     files_in       = PATH_IN,
                     file_out       = PATH_OUT1,
                     file_out2      = PATH_OUT2,
                     split_triggers = True,
                     trg1_code      = 1,
                     trg2_code      = 9,
                     event_range    = (0, nrequired),
                     **unpack_s12params(s12params)))

    with warns(UserWarning) as record:
        cnt = irene(**conf)
        assert cnt.events_in == nrequired

    #check there is a warning for unknown trigger types
    assert len(record) >= 1

    evt_warn = 2
    trg_warn = 5
    message = "Event {} has an unknown trigger type ({})".format(evt_warn, trg_warn)
    assert sum(1 for r in record if r.message.args[0] == message) == 1

    # Check events has been properly redirected to their files
    with tb.open_file(PATH_IN  , mode='r') as h5in  , \
         tb.open_file(PATH_OUT1, mode='r') as h5out1, \
         tb.open_file(PATH_OUT2, mode='r') as h5out2:
            # There is only one event per file
            assert h5out1.root.Trigger.events.shape[0] == 1
            assert h5out2.root.Trigger.events.shape[0] == 1

            # Event number and trigger type are correct
            evt1 = 0
            evt2 = 2
            trigger_in1  = h5in  .root.Trigger.events[evt1]
            trigger_in2  = h5in  .root.Trigger.events[evt2]
            trigger_out1 = h5out1.root.Trigger.events[0]
            trigger_out2 = h5out2.root.Trigger.events[0]
            np.testing.assert_array_equal(trigger_in1, trigger_out1)
            np.testing.assert_array_equal(trigger_in2, trigger_out2)

            evt_in1  = h5in  .root.Run.events[evt1][0]
            evt_in2  = h5in  .root.Run.events[evt2][0]
            evt_out1 = h5out1.root.Run.events[0][0]
            evt_out2 = h5out2.root.Run.events[0][0]
            np.testing.assert_array_equal(evt_in1, evt_out1)
            np.testing.assert_array_equal(evt_in2, evt_out2)


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

    tables = (     "MC/extents"    ,      "MC/hits"      ,    "MC/particles", "MC/generators",
                "PMAPS/S1"         ,   "PMAPS/S2"        , "PMAPS/S2Si"     ,
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
    file_in  = os.path.join(ICDATADIR    , "Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts.RWF.h5")
    file_out = os.path.join(output_tmpdir,            "test_irene_filters_empty_pmaps.h5")

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

    tables = (     "MC/extents",      "MC/hits"   ,    "MC/particles", "MC/generators",
                "PMAPS/S1"     ,   "PMAPS/S2"     , "PMAPS/S2Si"     ,
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
