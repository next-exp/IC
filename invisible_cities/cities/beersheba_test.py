import os
import numpy  as np
import tables as tb
import pandas as pd

from pytest                    import mark
from pytest                    import raises

from .. io                     import dst_io      as dio
from .  beersheba              import beersheba
from .  beersheba              import create_deconvolution_df
from .  beersheba              import distribute_energy
from .  beersheba              import deconvolve_signal
from .  beersheba              import CutType
from .  beersheba              import DeconvolutionMode
from .. reco.deconv_functions  import InterpolationMethod
from .. evm .event_model       import HitEnergy
from .. core.testing_utils     import assert_dataframes_close
from .. core.testing_utils     import assert_tables_equality


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


def test_beersheba_contains_all_tables(deconvolution_config):
    conf, PATH_OUT = deconvolution_config
    beersheba(**conf)
    with tb.open_file(PATH_OUT) as h5out:
        assert "MC"             in h5out.root
        assert "MC/hits"        in h5out.root
        assert "MC/particles"   in h5out.root
        assert "DECO/Events"    in h5out.root
        assert "Summary/Events" in h5out.root
        assert "Run"            in h5out.root
        assert "Run/events"     in h5out.root
        assert "Run/runInfo"    in h5out.root


def test_beersheba_exact_result_joint(ICDATADIR, deconvolution_config):
    true_out         = os.path.join(ICDATADIR, "test_Xe2nu_NEW_exact_deconvolution_joint.h5")
    conf, PATH_OUT   = deconvolution_config
    beersheba(**conf)

    ## tables = ( "MC/extents"    , "MC/hits"     , "MC/particles" , "MC/generators",
    ##            "DECO/Events"   ,
    ##            "Summary/Events",
    ##            "Run/events"    , "Run/runInfo" )
    tables = ( "DECO/Events"   ,
               "Summary/Events",
               "Run/events"    , "Run/runInfo" )

    with tb.open_file(true_out)  as true_output_file:
        with tb.open_file(PATH_OUT) as      output_file:
            for table in tables:
                got      = getattr(     output_file.root, table)
                expected = getattr(true_output_file.root, table)
                assert_tables_equality(got, expected)


def test_beersheba_exact_result_separate(ICDATADIR, deconvolution_config):
    true_out         = os.path.join(ICDATADIR, "test_Xe2nu_NEW_exact_deconvolution_separate.h5")
    conf, PATH_OUT   = deconvolution_config
    conf['deconv_params']['deconv_mode'   ] = 'separate'
    conf['deconv_params']['n_iterations'  ] = 50
    conf['deconv_params']['n_iterations_g'] = 50
    beersheba(**conf)

    ## tables = ( "MC/extents"    , "MC/hits"     , "MC/particles" , "MC/generators",
    ##            "DECO/Events"   ,
    ##            "Summary/Events",
    ##            "Run/events"    , "Run/runInfo" )
    tables = ( "DECO/Events"   ,
               "Summary/Events",
               "Run/events"    , "Run/runInfo" )

    with tb.open_file(true_out)  as true_output_file:
        with tb.open_file(PATH_OUT) as      output_file:
            for table in tables:
                got      = getattr(     output_file.root, table)
                expected = getattr(true_output_file.root, table)
                print(got)
                print(expected)
                assert_tables_equality(got, expected)


@mark.parametrize("ndim", (1, 3))
def test_beersheba_param_dim(deconvolution_config, ndim):
    conf, _ = deconvolution_config

    conf['deconv_params']['n_dim'  ] = ndim

    with raises(ValueError):
        beersheba(**conf)


@mark.parametrize("param_name", ('cut_type', 'deconv_mode', 'energy_type', 'inter_method'))
def test_deconvolve_signal_enums(deconvolution_config, param_name):
    conf, _   = deconvolution_config
    conf_dict = conf['deconv_params']

    conf_dict.pop("q_cut")
    conf_dict.pop("drop_dist")

    conf_dict['cut_type'    ] = CutType            (conf_dict['cut_type'    ])
    conf_dict['deconv_mode' ] = DeconvolutionMode  (conf_dict['deconv_mode' ])
    conf_dict['energy_type' ] = HitEnergy          (conf_dict['energy_type' ])
    conf_dict['inter_method'] = InterpolationMethod(conf_dict['inter_method'])

    conf_dict[param_name]     = param_name

    with raises(ValueError):
        deconvolve_signal(**conf_dict)


def test_beersheba_expandvar(deconvolution_config):
    conf, _ = deconvolution_config

    conf['deconv_params']['psf_fname'] = '$ICDIR/database/test_data/PSF_dst_sum_collapsed.h5'

    beersheba(**conf)
