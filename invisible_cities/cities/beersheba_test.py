import os
import numpy  as np
import tables as tb
import pandas as pd

from pytest                    import mark

from .. core.system_of_units_c import units
from .. core.configure         import configure
from .. core.configure         import all         as all_events
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
    assert true_dst1.E.sum() == true_dst2.E.sum()


def test_beersheba_contains_all_tables(ICDIR, ICDATADIR, PSFDIR, config_tmpdir):
    PATH_IN     = os.path.join(ICDATADIR    , "exact_Kr_tracks_with_MC.h5")
    PATH_OUT    = os.path.join(config_tmpdir,            "beersheba_MC.h5")
    config_path = os.path.join(ICDIR        ,      "config/beersheba.conf")
    conf        = configure(f'dummy {config_path}'.split())
    nevt_req    = all_events
    conf.update(dict(files_in      = PATH_IN ,
                     file_out      = PATH_OUT,
                     event_range   = nevt_req,
                     deconv_params = dict(q_cut         =         10,
                                          drop_dist     = [10., 10.],
                                          psf_fname     =     PSFDIR,
                                          e_cut         =       1e-3,
                                          n_iterations  =         10,
                                          iteration_tol =       0.01,
                                          sample_width  = [10., 10.],
                                          bin_size      = [ 1.,  1.],
                                          energy_type   =        'E',
                                          diffusion     = (1.0, 1.0),
                                          deconv_mode   =    'joint',
                                          n_dim         =          2,
                                          cut_type      =      'abs',
                                          inter_method  =    'cubic')))
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


def test_beersheba_exact_result(ICDIR, ICDATADIR, PSFDIR, config_tmpdir):
    PATH_IN     = os.path.join(ICDATADIR    ,        "exact_Kr_tracks_with_MC.h5")
    true_out    = os.path.join(ICDATADIR    , "exact_Kr_deconvolution_with_MC.h5")
    PATH_OUT    = os.path.join(config_tmpdir,                   "beersheba_MC.h5")
    config_path = os.path.join(ICDIR        ,             "config/beersheba.conf")
    conf        = configure(f'dummy {config_path}'.split())
    nevt_req    = all_events
    conf.update(dict(files_in      = PATH_IN ,
                     file_out      = PATH_OUT,
                     event_range   = nevt_req,
                     deconv_params = dict(q_cut         =         10,
                                          drop_dist     = [10., 10.],
                                          psf_fname     =     PSFDIR,
                                          e_cut         =       1e-3,
                                          n_iterations  =         10,
                                          iteration_tol =       0.01,
                                          sample_width  = [10., 10.],
                                          bin_size      = [ 1.,  1.],
                                          energy_type   =        'E',
                                          diffusion     = (1.0, 1.0),
                                          deconv_mode   =    'joint',
                                          n_dim         =          2,
                                          cut_type      =      'rel',
                                          inter_method  =    'cubic')))
    beersheba(**conf)

    tables = ( "MC/extents"  , "MC/hits"     , "MC/particles", "MC/generators",
               "DECO/Events" ,
               "Run/events"  , "Run/runInfo" )

    with tb.open_file(true_out)  as true_output_file:
        with tb.open_file(PATH_OUT) as      output_file:
            for table in tables:
                got      = getattr(     output_file.root, table)
                expected = getattr(true_output_file.root, table)
                assert_tables_equality(got, expected)


def test_beersheba_empty_input_file(ICDIR, ICDATADIR, PSFDIR, config_tmpdir):
    PATH_IN     = os.path.join(ICDATADIR    ,           'empty_hdst.h5')
    PATH_OUT    = os.path.join(config_tmpdir,         'empty_voxels.h5')
    config_path = os.path.join(ICDIR        ,   "config/beersheba.conf")
    conf        = configure(f'dummy {config_path}'.split())
    nevt_req    = all_events
    conf.update(dict(files_in      = PATH_IN ,
                     file_out      = PATH_OUT,
                     event_range   = nevt_req,
                     deconv_params = dict(q_cut         =         10,
                                          drop_dist     = [10., 10.],
                                          psf_fname     =     PSFDIR,
                                          e_cut         =       1e-3,
                                          n_iterations  =         10,
                                          iteration_tol =       0.01,
                                          sample_width  = [10., 10.],
                                          bin_size      = [ 1.,  1.],
                                          energy_type   =        'E',
                                          diffusion     = (1.0, 1.0),
                                          deconv_mode   =    'joint',
                                          n_dim         =          2,
                                          cut_type      =      'abs',
                                          inter_method  =    'cubic')))

    beersheba(**conf)
