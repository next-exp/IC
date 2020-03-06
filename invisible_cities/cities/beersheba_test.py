import os
import numpy  as np
import tables as tb
import pandas as pd

from .. io                     import dst_io      as dio
from .  beersheba              import beersheba
from .  beersheba              import create_deconvolution_df
from .  beersheba              import distribute_energy
from .  beersheba              import CutType
from .. evm.event_model        import HitEnergy
from .. core.testing_utils     import assert_dataframes_close
from .. core.testing_utils     import assert_tables_equality


def test_create_deconvolution_df(ICDATADIR):
    true_in  = os.path.join(ICDATADIR, "exact_Kr_deconvolution_with_MC.h5")
    true_dst = dio.load_dst(true_in, 'DECO', 'Events')
    ecut     = 1e-2
    new_dst  = pd.concat([create_deconvolution_df(t, t.E.values, (t.X.values, t.Y.values, t.Z.values),
                                                  CutType.abs, ecut, 3) for _, t in true_dst.groupby('event')])
    true_dst = true_dst.loc[true_dst.E > ecut, :].reset_index(drop=True)

    assert_dataframes_close(new_dst, true_dst)


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
        assert "MC"           in h5out.root
        assert "MC/extents"   in h5out.root
        assert "MC/hits"      in h5out.root
        assert "MC/particles" in h5out.root
        assert "DECO/Events"  in h5out.root
        assert "Run"          in h5out.root
        assert "Run/events"   in h5out.root
        assert "Run/runInfo"  in h5out.root


def test_beersheba_exact_result_joint(ICDATADIR, deconvolution_config):
    true_out         = os.path.join(ICDATADIR, "test_Xe2nu_NEW_exact_deconvolution_joint.h5")
    conf, PATH_OUT   = deconvolution_config
    beersheba(**conf)

    tables = ( "MC/extents"  , "MC/hits"     , "MC/particles" , "MC/generators",
               "DECO/Events" ,
               "Run/events"  , "Run/runInfo" )

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

    tables = ( "MC/extents"  , "MC/hits"     , "MC/particles" , "MC/generators",
               "DECO/Events" ,
               "Run/events"  , "Run/runInfo" )

    with tb.open_file(true_out)  as true_output_file:
        with tb.open_file(PATH_OUT) as      output_file:
            for table in tables:
                got      = getattr(     output_file.root, table)
                expected = getattr(true_output_file.root, table)
                print(got)
                print(expected)
                assert_tables_equality(got, expected)


def test_beersheba_empty_input_file(deconvolution_config):
    conf, PATH_OUT = deconvolution_config
    beersheba(**conf)
