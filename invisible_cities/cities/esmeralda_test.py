import os
import numpy  as np
import tables as tb
import pandas as pd

from  . components           import get_event_info
from  . components           import length_of
from .. core                 import system_of_units as units
from .. core.configure       import configure
from .. core.configure       import all         as all_events
from .. io                   import dst_io      as dio
from .. io.mcinfo_io         import load_mchits_df
from .  esmeralda            import esmeralda
from .. core.testing_utils   import assert_dataframes_close
from .. core.testing_utils   import assert_tables_equality


def test_esmeralda_contains_all_tables(KrMC_hdst_filename, correction_map_MC_filename, config_tmpdir):
    PATH_IN   = KrMC_hdst_filename
    PATH_OUT  = os.path.join(config_tmpdir, "Kr_tracks_with_MC.h5")
    conf      = configure('dummy invisible_cities/config/esmeralda.conf'.split())
    nevt_req  = all_events
    conf.update(dict(files_in                  = PATH_IN                   ,
                     file_out                  = PATH_OUT                  ,
                     event_range               = nevt_req                  ,
                     cor_hits_params           = dict(
                         map_fname             = correction_map_MC_filename,
                         threshold_charge_low  = 6  * units.pes            ,
                         threshold_charge_high = 30 * units.pes            ,
                         same_peak             = True                      ,
                         apply_temp            = False                    )))
    esmeralda(**conf)

    with tb.open_file(PATH_OUT) as h5out:
        assert "MC"                      in h5out.root
        assert "MC/hits"                 in h5out.root
        assert "MC/particles"            in h5out.root
        assert "Tracking/Tracks"         in h5out.root
        assert "Summary/Events"          in h5out.root
        assert "CHITS"                   in h5out.root
        assert "CHITS/highTh"            in h5out.root
        assert "CHITS/lowTh"             in h5out.root
        assert "Run"                     in h5out.root
        assert "Run/events"              in h5out.root
        assert "Run/runInfo"             in h5out.root
        assert "Filters/low_th_select"   in h5out.root
        assert "Filters/high_th_select"  in h5out.root
        assert "Filters/topology_select" in h5out.root
        assert "DST/Events"              in h5out.root



def test_esmeralda_filters_events(KrMC_hdst_filename_toy, correction_map_MC_filename, config_tmpdir):
    PATH_IN   = KrMC_hdst_filename_toy
    PATH_OUT  = os.path.join(config_tmpdir, "Kr_tracks_with_MC_filtered.h5")
    conf      = configure('dummy invisible_cities/config/esmeralda.conf'.split())
    nevt_req  = 8

    conf.update(dict(files_in                  = PATH_IN                   ,
                     file_out                  = PATH_OUT                  ,
                     event_range               = nevt_req                  ,
                     cor_hits_params           = dict(
                         map_fname             = correction_map_MC_filename,
                         threshold_charge_low  = 150  * units.pes          ,
                         threshold_charge_high = 200  * units.pes          ,
                         same_peak             = True                      ,
                         apply_temp            = False                    )))

    cnt = esmeralda(**conf)

    events_pass_low_th  =  [0, 1, 2, 3, 4, 5, 6]
    events_pass_paolina =  [3, 4, 5, 6]
    nevt_in             =  cnt.events_in
    nevt_out            =  cnt.events_out
    assert nevt_req     == nevt_in
    assert nevt_out     == len(set(events_pass_paolina))

    df_hits_low_th      =  dio.load_dst(PATH_OUT, 'CHITS'   , 'lowTh' )
    df_hits_paolina     =  dio.load_dst(PATH_OUT, 'CHITS'   , 'highTh')
    df_tracks_paolina   =  dio.load_dst(PATH_OUT, 'Tracking', 'Tracks')
    df_summary_paolina  =  dio.load_dst(PATH_OUT, 'Summary' , 'Events')

    assert set(df_hits_low_th .event.unique()) ==  set(events_pass_low_th )
    assert set(df_hits_paolina.event.unique()) ==  set(events_pass_paolina)

    assert set(df_hits_paolina.event.unique()) ==  set(df_tracks_paolina  .event.unique())
    assert set(df_hits_paolina.event.unique()) ==  set(df_summary_paolina.event.unique())

    #assert event number in EventInfo and MC/Extents iqual to nevt_req
    with tb.open_file(PATH_OUT)  as h5out:
        event_info = get_event_info(h5out)
        assert length_of(event_info) == nevt_req
        MC_num_evs = load_mchits_df(PATH_OUT).index.levels[0]
        assert len(MC_num_evs) == nevt_req



def test_esmeralda_with_out_of_map_hits(KrMC_hdst_filename_toy, correction_map_MC_filename, config_tmpdir):
    PATH_IN   = KrMC_hdst_filename_toy
    PATH_OUT  = os.path.join(config_tmpdir, "Kr_tracks_with_MC_out_of_map.h5")
    conf      = configure('dummy invisible_cities/config/esmeralda.conf'.split())
    nevt_req  = 8
    conf.update(dict(files_in                  = PATH_IN                   ,
                     file_out                  = PATH_OUT                  ,
                     event_range               = nevt_req                  ,
                     cor_hits_params           = dict(
                         map_fname             = correction_map_MC_filename,
                         threshold_charge_low  = 20   * units.pes          ,
                         threshold_charge_high = 20   * units.pes          ,
                         same_peak             = True                      ,
                         apply_temp            = False                    )))

    cnt = esmeralda(**conf)

    events_pass_low_th  =  [0, 1, 2, 3, 4, 5, 6, 7]
    events_pass_paolina =  [0, 1, 2, 3, 4, 5, 6, 7]
    nevt_in             =  cnt.events_in
    nevt_out            =  cnt.events_out
    assert nevt_req     == nevt_in
    assert nevt_out     == len(set(events_pass_paolina))

    df_hits_low_th      =  dio.load_dst(PATH_OUT, 'CHITS', 'lowTh' )
    df_hits_paolina     =  dio.load_dst(PATH_OUT, 'CHITS', 'highTh')

    assert set(df_hits_low_th .event.unique()) ==  set(events_pass_low_th )
    assert set(df_hits_paolina.event.unique()) ==  set(events_pass_paolina)

    summary_table       =  dio.load_dst(PATH_OUT, 'Summary', 'Events')
    #assert event with nan energy labeled in summary_table
    events_energy       =  df_hits_paolina.groupby('event').Ec.apply(pd.Series.sum, skipna=False)
    np.testing.assert_array_equal(summary_table.evt_out_of_map, np.isnan(events_energy.values))


def test_esmeralda_tracks_exact(data_hdst, esmeralda_tracks, correction_map_filename, config_tmpdir):
    PATH_IN   = data_hdst
    PATH_OUT  = os.path.join(config_tmpdir, "exact_tracks_esmeralda.h5")
    conf      = configure('dummy invisible_cities/config/esmeralda.conf'.split())
    nevt_req  = all_events

    conf.update(dict(files_in                  = PATH_IN                ,
                     file_out                  = PATH_OUT               ,
                     event_range               = nevt_req               ,
                     run_number                = 6822                   ,
                     cor_hits_params           = dict(
                         map_fname             = correction_map_filename,
                         threshold_charge_low  = 10   * units.pes       ,
                         threshold_charge_high = 30   * units.pes       ,
                         same_peak             = True                   ,
                         apply_temp            = False                 ),
                     paolina_params            = dict(
                         vox_size              = [15 * units.mm] * 3    ,
                         strict_vox_size       = False                  ,
                         energy_threshold      = 0 * units.keV          ,
                         min_voxels            = 2                      ,
                         blob_radius           = 21 * units.mm          ,
                         max_num_hits          = 10000                 )))

    esmeralda(**conf)

    df_tracks           =  dio.load_dst(PATH_OUT, 'Tracking', 'Tracks' )
    df_tracks_exact     =  pd.read_hdf(esmeralda_tracks, key = 'Tracks')
    columns2 = df_tracks_exact.columns
    #some events are not in df_tracks_exact
    events = df_tracks_exact.event.unique()
    df_tracks_cut  = df_tracks[df_tracks.event.isin(events)]

    assert_dataframes_close (df_tracks_cut[columns2]  .reset_index(drop=True),
                             df_tracks_exact[columns2].reset_index(drop=True))
    #make sure out_of_map is true for events not in df_tracks_exact
    diff_events = list(set(df_tracks.event.unique()).difference(events))
    df_summary  = dio.load_dst(PATH_OUT, 'Summary', 'Events')
    assert np.all(df_summary.loc[df_summary.event.isin(diff_events),'evt_out_of_map'])


def test_esmeralda_empty_input_file(config_tmpdir, ICDATADIR):
    # Esmeralda must run on an empty file without raising any exception
    # The input file has the complete structure of a PMAP but no events.

    PATH_IN  = os.path.join(ICDATADIR    , 'empty_hdst.h5')
    PATH_OUT = os.path.join(config_tmpdir, 'empty_voxels.h5')

    conf = configure('dummy invisible_cities/config/esmeralda.conf'.split())
    conf.update(dict(files_in = PATH_IN,
                     file_out = PATH_OUT))

    esmeralda(**conf)

#if the first analyzed events has no overlap in blob buggy esmeralda will cast all overlap energy to integers
def test_esmeralda_blob_overlap_bug(data_hdst, correction_map_filename, config_tmpdir):
    PATH_IN   = data_hdst
    PATH_OUT  = os.path.join(config_tmpdir, "exact_tracks_esmeralda_bug.h5")
    conf      = configure('dummy invisible_cities/config/esmeralda.conf'.split())
    nevt_req  = 4, 8

    conf.update(dict(files_in                  = PATH_IN                ,
                     file_out                  = PATH_OUT               ,
                     event_range               = nevt_req               ,
                     run_number                = 6822                   ,
                     cor_hits_params           = dict(
                         map_fname             = correction_map_filename,
                         threshold_charge_low  = 10   * units.pes       ,
                         threshold_charge_high = 30   * units.pes       ,
                         same_peak             = True                   ,
                         apply_temp            = False                 ),
                     paolina_params            = dict(
                         vox_size              = [15 * units.mm] * 3    ,
                         strict_vox_size       = False                  ,
                         energy_threshold      = 0 * units.keV          ,
                         min_voxels            = 2                      ,
                         blob_radius           = 21 * units.mm          ,
                         max_num_hits          = 10000                 )))

    esmeralda(**conf)

    df_tracks = dio.load_dst(PATH_OUT, 'Tracking', 'Tracks')
    assert df_tracks['ovlp_blob_energy'].dtype == float

def test_esmeralda_exact_result_all_events(ICDATADIR, KrMC_hdst_filename, correction_map_MC_filename, config_tmpdir):
    file_in   = KrMC_hdst_filename
    file_out  = os.path.join(config_tmpdir, "exact_Kr_tracks_with_MC.h5")
    conf      = configure('dummy invisible_cities/config/esmeralda.conf'.split())
    true_out  =  os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC_KDST_no_filter.h5")
    nevt_req  = all_events
    conf.update(dict(files_in                  = file_in                   ,
                     file_out                  = file_out                  ,
                     event_range               = nevt_req                  ,
                     cor_hits_params           = dict(
                         map_fname             = correction_map_MC_filename,
                         threshold_charge_low  = 10   * units.pes          ,
                         threshold_charge_high = 20   * units.pes          ,
                         same_peak             = True                      ,
                         apply_temp            = False                    ),
                     paolina_params            = dict(
                         vox_size              = [15 * units.mm] * 3       ,
                         strict_vox_size       = False                     ,
                         energy_threshold      = 0 * units.keV             ,
                         min_voxels            = 2                         ,
                         blob_radius           = 21 * units.mm             ,
                         max_num_hits          = 10000                    )))

    esmeralda(**conf)


    ## tables = ["MC/extents", "MC/hits", "MC/particles",
    ##           "Tracking/Tracks", "CHITS/lowTh", "CHITS/highTh",
    ##           "Run/events", "Run/runInfo", "DST/Events", "Summary/Events",
    ##           "Filters/low_th_select", "Filters/high_th_select", "Filters/topology_select"]
    tables = ["Tracking/Tracks", "CHITS/lowTh", "CHITS/highTh",
              "Run/events", "Run/runInfo", "DST/Events", "Summary/Events",
              "Filters/low_th_select", "Filters/high_th_select", "Filters/topology_select"]

    with tb.open_file(true_out)  as true_output_file:
        with tb.open_file(file_out) as   output_file:
            for table in tables:
                got      = getattr(     output_file.root, table)
                expected = getattr(true_output_file.root, table)
                assert_tables_equality(got, expected)



#test showing that all events that pass charge threshold are contained in hits output
def test_esmeralda_bug_duplicate_hits(data_hdst, correction_map_filename, config_tmpdir):
    PATH_IN   = data_hdst
    PATH_OUT  = os.path.join(config_tmpdir, "exact_tracks_esmeralda_drop_voxels_bug.h5")
    conf      = configure('dummy invisible_cities/config/esmeralda.conf'.split())
    nevt_req  = 1
    conf.update(dict(files_in                  = PATH_IN                ,
                     file_out                  = PATH_OUT               ,
                     event_range               = nevt_req               ,
                     run_number                = 6822                   ,
                     cor_hits_params           = dict(
                         map_fname             = correction_map_filename,
                         threshold_charge_low  = 10   * units.pes       ,
                         threshold_charge_high = 30   * units.pes       ,
                         same_peak             = True                   ,
                         apply_temp            = False                 ),
                     paolina_params            = dict(
                         vox_size              = [15 * units.mm] * 3    ,
                         strict_vox_size       = False                  ,
                         energy_threshold      = 0 * units.keV          ,
                         min_voxels            = 2                      ,
                         blob_radius           = 21 * units.mm          ,
                         max_num_hits          = 10000                 )))

    esmeralda(**conf)

    df_tracks = dio.load_dst(PATH_OUT, 'Tracking', 'Tracks')
    df_phits  = dio.load_dst(PATH_OUT, 'CHITS'   , 'highTh')

    for (event_num, ev_phits) in df_phits.groupby('event'):
        assert  sum(df_tracks[df_tracks.event==event_num].numb_of_hits) == len(ev_phits)


#test showing that all events that pass charge threshold are contained in hits output
def test_esmeralda_all_hits_after_drop_voxels(data_hdst, esmeralda_tracks, correction_map_filename, config_tmpdir):
    PATH_IN   = data_hdst
    PATH_OUT  = os.path.join(config_tmpdir, "exact_tracks_esmeralda_drop_voxels.h5")
    conf      = configure('dummy invisible_cities/config/esmeralda.conf'.split())
    nevt_req  = all_events
    th_p      = 30 * units.pes
    conf.update(dict(files_in                  = PATH_IN                ,
                     file_out                  = PATH_OUT               ,
                     event_range               = nevt_req               ,
                     run_number                = 6822                   ,
                     cor_hits_params           = dict(
                         map_fname             = correction_map_filename,
                         threshold_charge_low  = 10   * units.pes       ,
                         threshold_charge_high = th_p                   ,
                         same_peak             = True                   ,
                         apply_temp            = False                 ),
                     paolina_params            = dict(
                         vox_size              = [15 * units.mm] * 3    ,
                         strict_vox_size       = False                  ,
                         energy_threshold      = 20 * units.keV         ,
                         min_voxels            = 2                      ,
                         blob_radius           = 21 * units.mm          ,
                         max_num_hits          = 10000                 )))
    esmeralda(**conf)

    df_phits            =  dio.load_dst(PATH_OUT,   'CHITS' , 'highTh')
    df_in_hits          =  dio.load_dst(PATH_IN ,    'RECO' , 'Events')

    num_pass_th_in_hits = sum(df_in_hits.Q >= th_p)
    num_pass_th_p_hits  = len(df_phits)
    assert num_pass_th_in_hits == num_pass_th_p_hits

    #the number of finite Ep should be equal to Ec if no voxels were dropped.
    num_paolina_hits   = sum(np.isfinite(df_phits.Ep))
    assert num_paolina_hits <= num_pass_th_p_hits

    #check that the sum of Ep and Ec energies is the same
    assert np.nansum(df_phits.Ec) == np.nansum(df_phits.Ep)

def test_esmeralda_filters_events_with_too_many_hits(data_hdst, correction_map_filename, config_tmpdir):
    PATH_IN   = data_hdst
    PATH_OUT  = os.path.join(config_tmpdir, "esmeralda_filters_long_events.h5")
    conf      = configure('dummy invisible_cities/config/esmeralda.conf'.split())
    nevt_req  = 9
    nhits_max = 50
    paolina_events = {3021898, 3021914, 3021930, 3020951, 3020961}
    conf.update(dict(files_in                  = PATH_IN                ,
                     file_out                  = PATH_OUT               ,
                     event_range               = nevt_req               ,
                     run_number                = 6822                   ,
                     cor_hits_params           = dict(
                         map_fname             = correction_map_filename,
                         threshold_charge_low  = 10   * units.pes       ,
                         threshold_charge_high = 30   * units.pes       ,
                         same_peak             = True                   ,
                         apply_temp            = False                 ),
                     paolina_params            = dict(
                         vox_size              = [15 * units.mm] * 3    ,
                         strict_vox_size       = False                  ,
                         energy_threshold      = 20 * units.keV         ,
                         min_voxels            = 2                      ,
                         blob_radius           = 21 * units.mm          ,
                         max_num_hits          = nhits_max            )))

    esmeralda(**conf)

    summary = dio.load_dst(PATH_OUT, 'Summary' , 'Events')
    tracks  = dio.load_dst(PATH_OUT, 'Tracking', 'Tracks')
    hits    = dio.load_dst(PATH_OUT,   'CHITS' , 'highTh')

    #assert only paolina_events inside tracks
    assert set(tracks .event.unique()) == paolina_events

    #assert all events in summary table
    assert summary.event.nunique() == nevt_req
    #assert ntrk is 0 for non_paolina events
    assert np.all(summary[ summary.event.isin(paolina_events)].evt_ntrks >  0        )
    assert np.all(summary[~summary.event.isin(paolina_events)].evt_ntrks == 0        )
    assert np.all(summary[~summary.event.isin(paolina_events)].evt_nhits >  nhits_max)

    #assert all hits and events are in hits table
    assert len(hits) == 601
    assert hits.event.nunique() == nevt_req

    #assert all events in topology filter with corresponding bool
    topology_filter = dio.load_dst(PATH_OUT, 'Filters', 'topology_select')
    assert len(topology_filter) == nevt_req
    assert np.all(topology_filter[ topology_filter.event.isin(paolina_events)].passed == 1)
    assert np.all(topology_filter[~topology_filter.event.isin(paolina_events)].passed == 0)
