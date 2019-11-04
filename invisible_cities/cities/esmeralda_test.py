import os
import numpy  as np
import tables as tb
import pandas as pd

from pytest                    import mark
from  . components             import get_event_info
from  . components             import length_of
from .. core.system_of_units_c import units
from .. core.configure         import configure
from .. core.configure         import all         as all_events
from .. io                     import dst_io      as dio
from .  esmeralda              import esmeralda
from .. core.testing_utils     import assert_dataframes_close
from .. core.testing_utils     import assert_tables_equality


def test_esmeralda_contains_all_tables(KrMC_hdst_filename, correction_map_MC_filename, config_tmpdir):
    PATH_IN   = KrMC_hdst_filename
    PATH_OUT  = os.path.join(config_tmpdir, "Kr_tracks_with_MC.h5")
    conf      = configure('dummy invisible_cities/config/esmeralda.conf'.split())
    nevt_req  = all_events
    conf.update(dict(files_in                     = PATH_IN                   ,
                     file_out                     = PATH_OUT                  ,
                     event_range                  = nevt_req                  ,
                     cor_hits_params              = dict(
                         map_fname                = correction_map_MC_filename,
                         threshold_charge_NN      = 6  * units.pes            ,
                         threshold_charge_paolina = 30 * units.pes            ,
                         same_peak                = True                      ,
                         apply_temp               = False                   )))
    esmeralda(**conf)

    with tb.open_file(PATH_OUT) as h5out:
        assert "MC"                      in h5out.root
        assert "MC/extents"              in h5out.root
        assert "MC/hits"                 in h5out.root
        assert "MC/particles"            in h5out.root
        assert "PAOLINA"                 in h5out.root
        assert "PAOLINA/Events"          in h5out.root
        assert "PAOLINA/Summary"         in h5out.root
        assert "PAOLINA/Tracks"          in h5out.root
        assert "RECO/Events"             in h5out.root
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

    conf.update(dict(files_in                     = PATH_IN                   ,
                     file_out                     = PATH_OUT                  ,
                     event_range                  = nevt_req                  ,
                     cor_hits_params              = dict(
                         map_fname                = correction_map_MC_filename,
                         threshold_charge_NN      = 150  * units.pes          ,
                         threshold_charge_paolina = 200  * units.pes          ,
                         same_peak                = True                      ,
                         apply_temp               = False                    )))

    cnt = esmeralda(**conf)

    events_pass_low_th  =  [0, 1, 2, 3, 4, 5, 6]
    events_pass_paolina =  [3, 4, 5, 6]
    nevt_in             =  cnt.events_in
    nevt_out            =  cnt.events_out
    assert nevt_req     == nevt_in
    assert nevt_out     == len(set(events_pass_paolina))

    df_hits_low_th      =  dio.load_dst(PATH_OUT, 'RECO'   , 'Events' )
    df_hits_paolina     =  dio.load_dst(PATH_OUT, 'PAOLINA', 'Events' )
    df_tracks_paolina   =  dio.load_dst(PATH_OUT, 'PAOLINA', 'Tracks' )
    df_summary_paolina  =  dio.load_dst(PATH_OUT, 'PAOLINA', 'Summary')

    assert set(df_hits_low_th .event.unique()) ==  set(events_pass_low_th )
    assert set(df_hits_paolina.event.unique()) ==  set(events_pass_paolina)

    assert set(df_hits_paolina.event.unique()) ==  set(df_tracks_paolina  .event.unique())
    assert set(df_hits_paolina.event.unique()) ==  set(df_summary_paolina.event.unique())

    #assert event number in EventInfo and MC/Extents iqual to nevt_req
    with tb.open_file(PATH_OUT)  as h5out:
        event_info = get_event_info(h5out)
        assert length_of(event_info) == nevt_req

    with tb.open_file(PATH_OUT)  as h5out:
        MC_num_evs = h5out.root.MC.extents[:]['evt_number']
        assert len(MC_num_evs) == nevt_req



def test_esmeralda_with_out_of_map_hits(KrMC_hdst_filename_toy, correction_map_MC_filename, config_tmpdir):
    PATH_IN   = KrMC_hdst_filename_toy
    PATH_OUT  = os.path.join(config_tmpdir, "Kr_tracks_with_MC_out_of_map.h5")
    conf      = configure('dummy invisible_cities/config/esmeralda.conf'.split())
    nevt_req  = 8
    conf.update(dict(files_in                     = PATH_IN                   ,
                     file_out                     = PATH_OUT                  ,
                     event_range                  = nevt_req                  ,
                     cor_hits_params              = dict(
                         map_fname                = correction_map_MC_filename,
                         threshold_charge_NN      = 20   * units.pes          ,
                         threshold_charge_paolina = 20   * units.pes          ,
                         same_peak                = True                      ,
                         apply_temp               = False                    )))

    cnt = esmeralda(**conf)

    events_pass_low_th  =  [0, 1, 2, 3, 4, 5, 6, 7]
    events_pass_paolina =  [0, 1, 2, 3, 4, 5, 6, 7]
    nevt_in             =  cnt.events_in
    nevt_out            =  cnt.events_out
    assert nevt_req     == nevt_in
    assert nevt_out     == len(set(events_pass_paolina))

    df_hits_low_th      =  dio.load_dst(PATH_OUT, 'RECO'   , 'Events')
    df_hits_paolina     =  dio.load_dst(PATH_OUT, 'PAOLINA', 'Events')

    assert set(df_hits_low_th     .event.unique()) ==  set(events_pass_low_th     )
    assert set(df_hits_paolina.event.unique()) ==  set(events_pass_paolina)

    summary_table       =  dio.load_dst(PATH_OUT, 'PAOLINA', 'Summary')
    #assert event with nan energy labeled in summary_table
    events_energy = df_hits_paolina.groupby('event')['Ec'].apply(pd.Series.sum, skipna=False)
    np.testing.assert_array_equal( summary_table['out_of_map'], np.isnan(events_energy.values))
    #assert event number in EventInfo and MC/Extents iqual to nevt_req
    with tb.open_file(PATH_OUT)  as h5out:
        event_info = get_event_info(h5out)
        assert length_of(event_info) == nevt_req

    with tb.open_file(PATH_OUT)  as h5out:
        MC_num_evs = h5out.root.MC.extents[:]['evt_number']
        assert len(MC_num_evs) == nevt_req



def test_esmeralda_tracks_exact(data_hdst, esmeralda_tracks, correction_map_filename, config_tmpdir):
    PATH_IN   = data_hdst
    PATH_OUT  = os.path.join(config_tmpdir, "exact_tracks_esmeralda.h5")
    conf      = configure('dummy invisible_cities/config/esmeralda.conf'.split())
    nevt_req  = all_events

    conf.update(dict(files_in                     = PATH_IN                ,
                     file_out                     = PATH_OUT               ,
                     event_range                  = nevt_req               ,
                     run_number                   = 6822                   ,
                     cor_hits_params              = dict(
                         map_fname                = correction_map_filename,
                         threshold_charge_NN      = 10   * units.pes       ,
                         threshold_charge_paolina = 30   * units.pes       ,
                         same_peak                = True                   ,
                         apply_temp               = False                 ),
                     paolina_params               = dict(
                         vox_size                 = [15 * units.mm] * 3    ,
                         energy_type              = 'corrected'            ,
                         strict_vox_size          = False                  ,
                         energy_threshold         = 0 * units.keV          ,
                         min_voxels               = 2                      ,
                         blob_radius              = 21 * units.mm)        ))
    cnt = esmeralda(**conf)

    df_tracks           =  dio.load_dst(PATH_OUT, 'PAOLINA', 'Tracks')
    df_tracks_exact     =  pd.read_hdf(esmeralda_tracks, key = 'Tracks')
    columns1 = df_tracks      .columns
    columns2 = df_tracks_exact.columns
    #some events are not in df_tracks_exact
    events = df_tracks_exact.event.unique()
    assert(sorted(columns1) == sorted(columns2))
    df_tracks_cut  = df_tracks[df_tracks.event.isin(events)]
    assert_dataframes_close (df_tracks_cut[sorted(columns1)], df_tracks_exact[sorted(columns2)])
    #make sure out_of_map is true for events not in df_tracks_exact
    diff_events = list(set(df_tracks.event.unique()).difference(events))
    df_summary = dio.load_dst(PATH_OUT, 'PAOLINA', 'Summary')
    assert all(df_summary[df_summary.event.isin(diff_events)]['out_of_map'])

#The old test file should contain all tables the same except PAOLINA/Summary
def test_esmeralda_exact_result_old(ICDATADIR, KrMC_hdst_filename, correction_map_MC_filename, config_tmpdir):
    file_in   = KrMC_hdst_filename
    file_out  = os.path.join(config_tmpdir, "exact_Kr_tracks_with_MC.h5")
    conf      = configure('dummy invisible_cities/config/esmeralda.conf'.split())
    true_out  =  os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC.h5")
    nevt_req  = all_events
    conf.update(dict(files_in                     = file_in                   ,
                     file_out                     = file_out                  ,
                     event_range                  = nevt_req                  ,
                     cor_hits_params              = dict(
                         map_fname                = correction_map_MC_filename,
                         threshold_charge_NN      = 10   * units.pes          ,
                         threshold_charge_paolina = 20   * units.pes          ,
                         same_peak                = True                      ,
                         apply_temp               = False                    ),
                     paolina_params               = dict(
                         vox_size                 = [15 * units.mm] * 3       ,
                         energy_type              = 'corrected'               ,
                         strict_vox_size          = False                     ,
                         energy_threshold         = 0 * units.keV             ,
                         min_voxels               = 2                         ,
                         blob_radius              = 21 * units.mm           )))

    cnt = esmeralda(**conf)
    tables = ( "RECO/Events"      , "PAOLINA/Events"        , "PAOLINA/Tracks")


    with tb.open_file(true_out)  as true_output_file:
        with tb.open_file(file_out) as      output_file:
            for table in tables:
                got      = getattr(     output_file.root, table)
                expected = getattr(true_output_file.root, table)
                assert_tables_equality(got, expected)

    #The PAOLINA/Summary should contain this columns, and they should stay the same
    columns = ['event', 'S2ec' ,  'S2qc', 'ntrks', 'nhits',
               'x_avg', 'y_avg', 'z_avg', 'r_avg',
               'x_min', 'y_min', 'z_min', 'r_min',
               'x_max', 'y_max', 'z_max', 'r_max']

    summary_out  = dio.load_dst(file_out, 'PAOLINA', 'Summary')
    summary_true = dio.load_dst(true_out, 'PAOLINA', 'Summary')
    assert_dataframes_close(summary_out[columns], summary_true[columns])

    # PAOLINA/Summary should contain only columns plus out_of_map flag
    assert sorted(summary_out.columns) == sorted(columns + ['out_of_map'] )

    #Finally lets confirm Esmeralda contains KDST, RUN and MC table from Penthesilea file
    tables_in = ( "MC/extents"  , "MC/hits"       , "MC/particles"  , "MC/generators",
                  "DST/Events"  , "Run/events"    , "Run/runInfo"                    )
    with tb.open_file(file_in)  as true_output_file:
        with tb.open_file(file_out) as      output_file:
            for table in tables_in:
                got      = getattr(     output_file.root, table)
                expected = getattr(true_output_file.root, table)
                assert_tables_equality(got, expected)


def test_esmeralda_empty_input_file(config_tmpdir, ICDATADIR):
    # Esmeralda must run on an empty file without raising any exception
    # The input file has the complete structure of a PMAP but no events.

    PATH_IN  = os.path.join(ICDATADIR    , 'empty_hdst.h5')
    PATH_OUT = os.path.join(config_tmpdir, 'empty_voxels.h5')

    conf = configure('dummy invisible_cities/config/esmeralda.conf'.split())
    conf.update(dict(files_in      = PATH_IN,
                     file_out      = PATH_OUT))

    esmeralda(**conf)

#if the first analyzed events has no overlap in blob buggy esmeralda will cast all overlap energy to integers    
@mark.serial
def test_esmeralda_blob_overlap_bug(data_hdst, esmeralda_tracks, correction_map_filename, config_tmpdir):
    PATH_IN   = data_hdst
    PATH_OUT  = os.path.join(config_tmpdir, "exact_tracks_esmeralda.h5")
    conf      = configure('dummy invisible_cities/config/esmeralda.conf'.split())
    nevt_req  = 4, 8

    conf.update(dict(files_in                     = PATH_IN                ,
                     file_out                     = PATH_OUT               ,
                     event_range                  = nevt_req               ,
                     run_number                   = 6822                   ,
                     cor_hits_params              = dict(
                         map_fname                = correction_map_filename,
                         threshold_charge_NN      = 10   * units.pes       ,
                         threshold_charge_paolina = 30   * units.pes       ,
                         same_peak                = True                   ,
                         apply_temp               = False                 ),
                     paolina_params               = dict(
                         vox_size                 = [15 * units.mm] * 3    ,
                         energy_type              = 'corrected'            ,
                         strict_vox_size          = False                  ,
                         energy_threshold         = 0 * units.keV          ,
                         min_voxels               = 2                      ,
                         blob_radius              = 21 * units.mm)        ))
    cnt = esmeralda(**conf)

    df_tracks           =  dio.load_dst(PATH_OUT, 'PAOLINA', 'Tracks')
    assert df_tracks['ovlp_blob_energy'].dtype == float

def test_esmeralda_exact_result_all_events(ICDATADIR, KrMC_hdst_filename, correction_map_MC_filename, config_tmpdir):
    file_in   = KrMC_hdst_filename
    file_out  = os.path.join(config_tmpdir, "exact_Kr_tracks_with_MC.h5")
    conf      = configure('dummy invisible_cities/config/esmeralda.conf'.split())
    true_out  =  os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC_KDST_no_filter.h5")
    nevt_req  = all_events
    conf.update(dict(files_in                     = file_in                   ,
                     file_out                     = file_out                  ,
                     event_range                  = nevt_req                  ,
                     cor_hits_params              = dict(
                         map_fname                = correction_map_MC_filename,
                         threshold_charge_NN      = 10   * units.pes          ,
                         threshold_charge_paolina = 20   * units.pes          ,
                         same_peak                = True                      ,
                         apply_temp               = False                    ),
                     paolina_params               = dict(
                         vox_size                 = [15 * units.mm] * 3       ,
                         energy_type              = 'corrected'               ,
                         strict_vox_size          = False                     ,
                         energy_threshold         = 0 * units.keV             ,
                         min_voxels               = 2                         ,
                         blob_radius              = 21 * units.mm           )))

    cnt = esmeralda(**conf)

    tables = ["MC/extents", "MC/hits", "MC/particles",
              "PAOLINA/Events", "PAOLINA/Summary", "PAOLINA/Tracks", "RECO/Events",
              "Run/events", "Run/runInfo", "DST/Events",
              "Filters/low_th_select", "Filters/high_th_select", "Filters/topology_select"]

    with tb.open_file(true_out)  as true_output_file:
        with tb.open_file(file_out) as      output_file:
            for table in tables:
                got      = getattr(     output_file.root, table)
                expected = getattr(true_output_file.root, table)
                assert_tables_equality(got, expected)
