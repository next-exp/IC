import os
import numpy  as np
import tables as tb

from pytest                    import mark

from .. core.system_of_units_c import units
from .. core.configure         import configure
from .. core.configure         import all         as all_events
from .. io                     import dst_io      as dio
from .  esmeralda              import esmeralda



@mark.serial
def test_esmeralda_contains_all_tables(KrMC_hdst_filename, correction_map_MC_filename, config_tmpdir):
    PATH_IN   = KrMC_hdst_filename
    PATH_OUT  = os.path.join(config_tmpdir, "Kr_tracks_with_MC.h5")
    conf      = configure('dummy invisible_cities/config/esmeralda.conf'.split())
    nevt_req  = all_events
    conf.update(dict(files_in              = PATH_IN                   ,
                     file_out              = PATH_OUT                  ,
                     event_range           = nevt_req)                 ,
                     cor_hits_params_NN    = dict(
                         map_fname         = correction_map_MC_filename,
                         threshold_charge  = 6  * units.pes            ,
                         same_peak         = True                      ,
                         norm_strat        = 'kr'                      ,
                         apply_temp        = False)                    ,
                     cor_hits_params_PL    = dict(
                         map_fname         = correction_map_MC_filename,
                         threshold_charge  = 30 * units.pes            ,
                         same_peak         = True                      ,
                         norm_strat        = 'kr'                      ,
                         apply_temp        = False)                   ))
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
        assert "Filters/NN_select"       in h5out.root
        assert "Filters/paolina_select"  in h5out.root


@mark.serial
def test_esmeralda_filters_events(KrMC_hdst_filename_toy, correction_map_MC_filename, config_tmpdir):
    PATH_IN   = KrMC_hdst_filename_toy
    PATH_OUT  = os.path.join(config_tmpdir, "Kr_tracks_with_MC_filtered.h5")
    conf      = configure('dummy invisible_cities/config/esmeralda.conf'.split())
    nevt_req  = 8

    conf.update(dict(files_in              = PATH_IN                   ,
                     file_out              = PATH_OUT                  ,
                     event_range           = nevt_req)                 ,
                     cor_hits_params_NN    = dict(
                         map_fname         = correction_map_MC_filename,
                         threshold_charge  = 150  * units.pes          ,
                         same_peak         = True                      ,
                         norm_strat        = 'kr'                      ,
                         apply_temp        = False)                    ,
                     cor_hits_params_PL    = dict(
                         map_fname         = correction_map_MC_filename,
                         threshold_charge  = 200 * units.pes           ,
                         same_peak         = True                      ,
                         norm_strat        = 'kr'                      ,
                         apply_temp        = False)                   ))
    cnt = esmeralda(**conf)

    events_pass_NN      =  [0, 1, 2, 3, 4, 5, 6]
    events_pass_paolina =  [3, 4, 5, 6]
    nevt_in             =  cnt.events_in
    nevt_out            =  cnt.events_out
    assert nevt_req     == nevt_in
    assert nevt_out     == len(set(events_pass_paolina))

    df_hits_NN          =  dio.load_dst(PATH_OUT, 'RECO'   , 'Events' )
    df_hits_paolina     =  dio.load_dst(PATH_OUT, 'PAOLINA', 'Events' )
    df_tracks_paolina   =  dio.load_dst(PATH_OUT, 'PAOLINA', 'Tracks' )
    df_summary_paolina  =  dio.load_dst(PATH_OUT, 'PAOLINA', 'Summary')

    assert set(df_hits_NN     .event.unique()) ==  set(events_pass_NN     )
    assert set(df_hits_paolina.event.unique()) ==  set(events_pass_paolina)

    assert set(df_hits_paolina.event.unique()) ==  set(df_tracks_paolina  .event.unique())
    assert set(df_hits_paolina.event.unique()) ==  set(df_summary_paolina.event.unique())


@mark.serial
def test_esmeralda_with_out_of_map_hits(KrMC_hdst_filename_toy, correction_map_MC_filename, config_tmpdir):
    PATH_IN   = KrMC_hdst_filename_toy
    PATH_OUT  = os.path.join(config_tmpdir, "Kr_tracks_with_MC_out_of_map.h5")
    conf      = configure('dummy invisible_cities/config/esmeralda.conf'.split())
    nevt_req  = 8

    conf.update(dict(files_in              = PATH_IN                   ,
                     file_out              = PATH_OUT                  ,
                     event_range           = nevt_req)                 ,
                     cor_hits_params_NN    = dict(
                         map_fname         = correction_map_MC_filename,
                         threshold_charge  = 15   * units.pes          ,
                         same_peak         = True                      ,
                         norm_strat        = 'kr'                      ,
                         apply_temp        = False)                    ,
                     cor_hits_params_PL    = dict(
                         map_fname         = correction_map_MC_filename,
                         threshold_charge  = 10  * units.pes           ,
                         same_peak         = True                      ,
                         norm_strat        = 'kr'                      ,
                         apply_temp        = False)                   ))
    cnt = esmeralda(**conf)

    events_pass_NN      =  [0, 1, 2, 3, 4, 5, 6, 7]
    events_pass_paolina =  [0, 1, 2, 4, 5, 6, 7]
    nevt_in             =  cnt.events_in
    nevt_out            =  cnt.events_out
    assert nevt_req     == nevt_in
    assert nevt_out     == len(set(events_pass_paolina))

    df_hits_NN          =  dio.load_dst(PATH_OUT, 'RECO'   , 'Events')
    df_hits_paolina     =  dio.load_dst(PATH_OUT, 'PAOLINA', 'Events')

    assert set(df_hits_NN     .event.unique()) ==  set(events_pass_NN     )
    assert set(df_hits_paolina.event.unique()) ==  set(events_pass_paolina)
