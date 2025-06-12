import os
import shutil
import numpy  as np
import tables as tb

from pytest import mark

from .. core                 import system_of_units as units
from .. io                   import dst_io      as dio
from .  esmeralda            import esmeralda
from .. core.testing_utils   import assert_tables_equality
from .. core.testing_utils   import ignore_warning


@ignore_warning.no_config_group
def test_esmeralda_runs(esmeralda_config, config_tmpdir):
    path_out = os.path.join(config_tmpdir, 'esmeralda_runs.h5')
    nevt_req = 1
    esmeralda_config.update(dict( file_out    = path_out
                                , event_range = nevt_req))

    cnt = esmeralda(**esmeralda_config)
    assert cnt.events_in == nevt_req


@ignore_warning.no_config_group
def test_esmeralda_contains_all_tables(esmeralda_config, config_tmpdir):
    path_out = os.path.join(config_tmpdir, 'esmeralda_contains_all_tables.h5')
    nevt_req = 1
    esmeralda_config.update(dict( file_out    = path_out
                                , event_range = nevt_req))

    esmeralda(**esmeralda_config)

    nodes = ( "MC", "MC/hits", "MC/particles"
            , "Tracking", "Tracking/Tracks"
            , "Summary", "Summary/Events"
            , "CHITS", "CHITS/highTh"
            , "Run", "Run/events", "Run/runInfo"
            , "Filters", "Filters/high_th_select", "Filters/topology_select"
            , "DST", "DST/Events")

    with tb.open_file(path_out) as h5out:
        for node in nodes:
            assert node in h5out.root


@ignore_warning.no_config_group
def test_esmeralda_thresholds_hits(esmeralda_config, config_tmpdir):
    path_out  = os.path.join(config_tmpdir, "esmeralda_thresholds_hits.h5")
    threshold = 50 * units.pes
    esmeralda_config.update(dict( file_out    = path_out
                                , event_range = 1
                                , threshold   = threshold))

    esmeralda(**esmeralda_config)

    df = dio.load_dst(path_out, "CHITS", "highTh")
    assert np.all(df.Q >= threshold)


@ignore_warning.no_config_group
def test_esmeralda_drops_external_hits(esmeralda_config, config_tmpdir):
    path_out   = os.path.join(config_tmpdir, "esmeralda_drops_external_hits.h5")
    fiducial_r = 450 * units.mm;
    esmeralda_config.update(dict( file_out    = path_out
                                , event_range = 1
                                , fiducial_r  = fiducial_r))

    esmeralda(**esmeralda_config)

    df = dio.load_dst(path_out, "CHITS", "highTh")
    assert np.all(df.X**2 + df.Y**2 <= fiducial_r**2)


@ignore_warning.no_config_group
def test_esmeralda_filters_events_threshold(esmeralda_config, config_tmpdir):
    path_out = os.path.join(config_tmpdir, "esmeralda_filters_events_threshold.h5")
    esmeralda_config.update(dict( file_out    = path_out
                                , event_range = (6, 8)))

    cnt = esmeralda(**esmeralda_config)

    evt_all        = [400078, 400270]
    evt_all_nexus  = [200039, 200135]
    evt_pass       = [400078]
    #evt_pass_nexus = [200039]
    assert cnt.events_in   == 2
    assert cnt.events_out  == 2
    assert cnt.evtnum_list == evt_all

    df_hits    = dio.load_dst(path_out, 'CHITS'   , 'highTh')
    df_tracks  = dio.load_dst(path_out, 'Tracking', 'Tracks')
    df_summary = dio.load_dst(path_out, 'Summary' , 'Events')
    df_mc      = dio.load_dst(path_out, 'MC'      , 'hits'  )
    df_events  = dio.load_dst(path_out, 'Run'     , 'events')

    assert  df_hits   .event     .drop_duplicates().tolist() == evt_pass
    assert  df_tracks .event     .drop_duplicates().tolist() == evt_pass
    assert  df_summary.event     .drop_duplicates().tolist() == evt_pass
    assert  df_mc     .event_id  .drop_duplicates().tolist() == evt_all_nexus
    assert  df_events .evt_number.drop_duplicates().tolist() == evt_all


@ignore_warning.no_config_group
def test_esmeralda_exact_result(esmeralda_config, Th228_tracks, config_tmpdir):
    path_out  = os.path.join(config_tmpdir, "esmeralda_exact_result.h5")
    esmeralda_config["file_out"] = path_out

    esmeralda(**esmeralda_config)

    tables = ( "CHITS/highTh"
             , "Tracking/Tracks"
             , "Run/events", "Run/runInfo"
             , "DST/Events"
             , "Summary/Events"
             , "Filters/high_th_select", "Filters/topology_select"
             , "MC/event_mapping", "MC/hits", "MC/particles", "MC/configuration")

    with tb.open_file(Th228_tracks) as true_output_file:
        with tb.open_file(path_out) as      output_file:
            for table in tables:
                assert hasattr(output_file.root, table), table
                got      = getattr(     output_file.root, table)
                expected = getattr(true_output_file.root, table)
                assert_tables_equality(got, expected)


#if the first analyzed events has no overlap in blob buggy esmeralda will cast all overlap energy to integers
@ignore_warning.no_config_group
def test_esmeralda_blob_overlap_float_dtype(esmeralda_config, Th228_tracks, config_tmpdir):
    path_out = os.path.join(config_tmpdir, "esmeralda_blob_overlap_float_dtype.h5")

    # first event has no overlap, second event does
    esmeralda_config.update(dict( file_out    = path_out
                                , event_range = 2))
    esmeralda(**esmeralda_config)

    tracks = dio.load_dst(path_out, 'Tracking', 'Tracks')
    assert tracks.ovlp_blob_energy.dtype == float


@ignore_warning.no_config_group
def test_esmeralda_tracks_have_correct_number_of_hits(esmeralda_config, Th228_tracks, config_tmpdir):
    path_out = os.path.join(config_tmpdir, "esmeralda_summary_gives_correct_number_of_hits.h5")
    nevt_req = 3
    esmeralda_config.update(dict( file_out    = path_out
                                , event_range = nevt_req))
    esmeralda_config["paolina_params"]["energy_threshold"] = 0

    esmeralda(**esmeralda_config)

    df_tracks = dio.load_dst(path_out, 'Tracking', 'Tracks')
    df_phits  = dio.load_dst(path_out, 'CHITS'   , 'highTh')

    for (event_num, ev_phits) in df_phits.groupby('event'):
        track = df_tracks.loc[df_tracks.event==event_num]
        assert sum(track.numb_of_hits) == len(ev_phits)


@ignore_warning.no_config_group
def test_esmeralda_all_hits_after_drop_voxels(esmeralda_config, Th228_hits, config_tmpdir):
    path_out   = os.path.join(config_tmpdir, "esmeralda_all_hits_after_drop_voxels.h5")
    nevt_req   = 2
    threshold  = esmeralda_config["threshold"]
    fiducial_r = esmeralda_config["fiducial_r"]
    events     = 400062, 400064
    esmeralda_config.update(dict( file_out    = path_out
                                , event_range = nevt_req))

    esmeralda(**esmeralda_config)

    # First event drops 5 voxels, second event doesn't drop any
    after  = dio.load_dst(path_out  , 'CHITS', 'highTh')
    before = dio.load_dst(Th228_hits,  'RECO', 'Events')

    for event in events:
        evt_after      =      after.loc[lambda x: x.event       == event]
        evt_before     =     before.loc[lambda x: x.event       == event]
        evt_before     = evt_before.loc[lambda x: x.X**2+x.Y**2 <  fiducial_r**2]
        evt_before_thr = evt_before.loc[lambda x: x.Q           >= threshold]

        assert len(evt_before_thr)  == len(evt_after)
        assert np.isclose(np.sum(evt_after.Ec),  np.nansum(evt_before.Ec))
        assert np.isclose(np.sum(evt_after.Ec),  np.   sum(evt_after .Ep))


#TODO: refactor paolina to include this as a filter
@ignore_warning.no_config_group
def test_esmeralda_filters_events_with_too_many_hits(esmeralda_config, Th228_tracks, config_tmpdir):
    path_out  = os.path.join(config_tmpdir, "esmeralda_filters_events_with_too_many_hits.h5")
    nevt_req  = 2
    evt_pass  = [False, True]
    nhits_max = 100
    esmeralda_config.update(dict( file_out    = path_out
                                , event_range = nevt_req))

    esmeralda_config["paolina_params"].update(dict(max_num_hits=nhits_max))

    esmeralda(**esmeralda_config)

    summary       = dio.load_dst(path_out, 'Summary' , 'Events')
    tracks        = dio.load_dst(path_out, 'Tracking', 'Tracks')
    hits          = dio.load_dst(path_out,   'CHITS' , 'highTh')
    filter_output = dio.load_dst(path_out, 'Filters', 'topology_select')

    print(summary.evt_nhits)
    assert len(summary.event.drop_duplicates()) == nevt_req
    assert len( tracks.event.drop_duplicates()) == sum(evt_pass)
    assert len(   hits.event.drop_duplicates()) == nevt_req
    assert len(filter_output)                   == nevt_req

    assert (summary.evt_ntrks > 0        ).tolist() == evt_pass
    assert (summary.evt_nhits < nhits_max).tolist() == evt_pass
    assert            filter_output.passed.tolist() == evt_pass


@ignore_warning.no_hits
@ignore_warning.no_config_group
def test_esmeralda_does_not_crash_with_no_hits(esmeralda_config, Th228_hits_missing, config_tmpdir):
    path_out  = os.path.join(config_tmpdir, "esmeralda_does_not_crash_with_no_hits.h5")
    esmeralda_config.update(dict( files_in    = Th228_hits_missing
                                , file_out    = path_out
                                , event_range = 1))

    # just test that it doesn't crash
    esmeralda(**esmeralda_config)
