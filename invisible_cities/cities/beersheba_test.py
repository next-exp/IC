import os
import numpy  as np
import tables as tb
import pandas as pd

from pytest                import mark
from pytest                import raises

from .. core               import system_of_units as units
from .. io                 import dst_io      as dio
from .  beersheba          import beersheba
from .  beersheba          import create_deconvolution_df
from .  beersheba          import distribute_energy
from .  beersheba          import deconvolve_signal
from .. core.testing_utils import assert_dataframes_close
from .. core.testing_utils import assert_tables_equality
from .. database.load_db   import DataSiPM
from .. types.symbols      import HitEnergy
from .. types.symbols      import DeconvolutionMode
from .. types.symbols      import CutType


def test_create_deconvolution_df(ICDATADIR):
    true_in  = os.path.join(ICDATADIR, "exact_Kr_deconvolution_with_MC.h5")
    true_dst = dio.load_dst(true_in, 'DECO', 'Events')
    ecut     = 1e-2
    new_dst  = pd.concat([create_deconvolution_df(t, t.E.values, (t.X.values, t.Y.values, t.Z.values),
                                                  CutType.abs, ecut, 3) for _, t in true_dst.groupby('event')])
    true_dst = true_dst.loc[true_dst.E > ecut, :].reset_index(drop=True)

    assert_dataframes_close(new_dst .reset_index(drop=True), true_dst.reset_index(drop=True))


@mark.parametrize("cut_type", CutType.__members__)
def test_create_deconvolution_df_cuttype(ICDATADIR, cut_type):
    true_in  = os.path.join(ICDATADIR, "exact_Kr_deconvolution_with_MC.h5")
    true_dst = dio.load_dst(true_in, 'DECO', 'Events')
    ecut     = 1e-2

    with raises(ValueError):
        create_deconvolution_df(true_dst, true_dst.E.values,
                                (true_dst.X.values, true_dst.Y.values, true_dst.Z.values),
                                cut_type, ecut, 3)


def test_distribute_energy(ICDATADIR):
    true_in   = os.path.join(ICDATADIR, "exact_Kr_deconvolution_with_MC.h5")
    true_dst1 = dio.load_dst(true_in, 'DECO', 'Events')
    true_dst2 = true_dst1[:len(true_dst1)//2].copy()
    true_dst3 = true_dst2.copy()

    distribute_energy(true_dst2, true_dst1, HitEnergy.E)

    assert np.allclose(true_dst2.E.values/true_dst2.E.sum(), true_dst3.E.values/true_dst3.E.sum())
    assert np.isclose (true_dst1.E.sum(), true_dst2.E.sum())


@mark.filterwarnings("ignore:.*not of kdst type.*:UserWarning")
def test_beersheba_contains_all_tables(beersheba_config, config_tmpdir):
    path_out = os.path.join(config_tmpdir, "beersheba_contains_all_tables.h5")
    beersheba_config.update(dict( file_out    = path_out
                                , event_range = 1))

    beersheba(**beersheba_config)

    nodes = ( "MC", "MC/hits", "MC/particles"
            , "CHITS", "CHITS/lowTh"
            , "DECO", "DECO/Events"
            , "Run", "Run/events", "Run/runInfo")
    with tb.open_file(path_out) as h5out:
        for node in nodes:
            assert node in h5out.root


@mark.filterwarnings("ignore:.*not of kdst type.*:UserWarning")
@mark.parametrize("deco", DeconvolutionMode.__members__)
@mark.slow
def test_beersheba_exact_result( deco
                               , beersheba_config
                               , beersheba_config_separate
                               , Th228_deco
                               , Th228_deco_separate
                               , config_tmpdir):
    config   = beersheba_config if deco is DeconvolutionMode.joint else beersheba_config_separate
    true_out = Th228_deco       if deco is DeconvolutionMode.joint else Th228_deco_separate

    path_out = os.path.join(config_tmpdir, f"beersheba_exact_result_{deco}.h5")
    config.update(dict(file_out = path_out))

    beersheba(**config)

    tables = ( "DECO/Events"
             , "CHITS/lowTh"
             , "Run/events", "Run/runInfo"
             , "MC/event_mapping", "MC/configuration", "MC/hits", "MC/particles")

    with     tb.open_file(true_out) as true_output_file:
        with tb.open_file(path_out) as      output_file:
            for table in tables:
                assert hasattr(output_file.root, table), table
                got      = getattr(     output_file.root, table)
                expected = getattr(true_output_file.root, table)
                assert_tables_equality(got, expected)


@mark.parametrize("ndim", (1, 3))
def test_beersheba_only_ndim_2_is_valid(beersheba_config, ndim, config_tmpdir):
    path_out = os.path.join(config_tmpdir, "beersheba_only_ndim_2_is_valid.h5")
    beersheba_config.update(dict(file_out = path_out))
    beersheba_config['deconv_params'].update(dict(n_dim = ndim))

    with raises(ValueError):
        beersheba(**beersheba_config)


@mark.skip(reason = "This is handled by other means now")
@mark.parametrize("param_name", ('cut_type', 'deconv_mode', 'energy_type', 'inter_method'))
def test_deconvolve_signal_enums(deconvolution_config, param_name):
    conf, _   = deconvolution_config
    conf_dict = conf['deconv_params']

    conf_dict.pop("q_cut")
    conf_dict.pop("drop_dist")

    conf_dict[param_name] = conf_dict[param_name].name

    with raises(ValueError):
        deconvolve_signal(DataSiPM('new'), **conf_dict)


@mark.skip(reason = "This should be handled by other means")
@mark.filterwarnings("ignore:.*not of kdst type.*:UserWarning")
def test_beersheba_expandvar(deconvolution_config):
    conf, _ = deconvolution_config

    conf['deconv_params']['psf_fname'] = '$ICDIR/database/test_data/PSF_dst_sum_collapsed.h5'

    beersheba(**conf)


def test_beersheba_copies_kdst(beersheba_config, Th228_hits, config_tmpdir):
    path_out = os.path.join(config_tmpdir, "beersheba_copies_kdst.h5")
    beersheba_config.update(dict( file_out    = path_out
                                , event_range =        2))
    expected_events = [400062, 400064]

    beersheba(**beersheba_config)

    got_events = dio.load_dst(path_out, "DST", "Events").event.drop_duplicates().tolist()
    assert expected_events == got_events


def test_beersheba_thresholds_hits(beersheba_config, config_tmpdir):
    path_out  = os.path.join(config_tmpdir, "beersheba_thresholds_hits.h5")
    threshold = 15 * units.pes
    beersheba_config.update(dict( file_out    = path_out
                                , event_range = 1
                                , threshold   = threshold))

    beersheba(**beersheba_config)

    df = dio.load_dst(path_out, "CHITS", "lowTh")
    assert np.all(df.Q >= threshold)


def test_beersheba_filters_empty_dfs(beersheba_config, config_tmpdir):
    path_out = os.path.join(config_tmpdir, "beersheba_filters_empty_dfs.h5")
    q_cut    = 1e8 * units.pes
    beersheba_config.update(dict( file_out    = path_out
                                , event_range = 1))
    beersheba_config["deconv_params"].update(dict(q_cut = q_cut))

    cnt = beersheba(**beersheba_config)

    assert cnt.events_in            == 1
    assert cnt.events_out           == 0
    assert cnt.events_pass.n_passed == 0
    assert cnt.events_pass.n_failed == 1

    df = dio.load_dst(path_out, "Filters", "nohits")
    assert df.passed.tolist() == [False]
