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
from .. core.testing_utils     import assert_dataframes_close
from .. core.testing_utils     import assert_tables_equality


@mark.serial
def test_beersheba_contains_all_tables(ICDIR, ICDATADIR, config_tmpdir):
    PATH_IN     = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC.h5")
    PATH_OUT    = os.path.join(config_tmpdir, "beersheba_MC.h5")
    config_path = os.path.join(ICDIR, "config/beersheba.conf")
    conf        = configure(f'dummy {config_path}'.split())
    nevt_req    = all_events
    conf.update(dict(files_in                     = PATH_IN                   ,
                     file_out                     = PATH_OUT                  ,
                     event_range                  = nevt_req                  ,
                     deconv_params     = dict(
                                         q_cut           =         10,
                                         drop_dist       = [10., 10.],
                                         psf_fname       = f'{ICDIR}database/test_data/PSF_dst_sum_collapsed.h5',
                                         ecut            =       1e-3,
                                         iterationNumber =         10,
                                         iterationThr    =       0.01,
                                         sampleWidth     = [10., 10.],
                                         bin_size        = [ 1.,  1.],
                                         energy_type     =        'E',
                                         diffusion       = (1.0, 1.0),
                                         deconv_mode     =    'joint',
                                         n_dim           =          2,
                                         interMethod     = 'cubic')))
    beersheba(**conf)

    with tb.open_file(PATH_OUT) as h5out:
        assert "MC"                      in h5out.root
        assert "MC/extents"              in h5out.root
        assert "MC/hits"                 in h5out.root
        assert "MC/particles"            in h5out.root
        assert "DECO/Events"             in h5out.root
        assert "Run"                     in h5out.root
        assert "Run/events"              in h5out.root
        assert "Run/runInfo"             in h5out.root


@mark.serial
def test_beersheba_exact_result(ICDIR, ICDATADIR, config_tmpdir):
    PATH_IN     = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC.h5")
    true_out    = os.path.join(ICDATADIR, "exact_Kr_deconvolution_with_MC.h5")
    PATH_OUT    = os.path.join(config_tmpdir, "beersheba_MC.h5")
    config_path = os.path.join(ICDIR, "config/beersheba.conf")
    conf        = configure(f'dummy {config_path}'.split())
    nevt_req    = all_events
    conf.update(dict(files_in                     = PATH_IN                   ,
                     file_out                     = PATH_OUT                  ,
                     event_range                  = nevt_req                  ,
                     deconv_params     = dict(
                                         q_cut           =         10,
                                         drop_dist       = [10., 10.],
                                         psf_fname       = f'{ICDIR}database/test_data/PSF_dst_sum_collapsed.h5',
                                         ecut            =       1e-3,
                                         iterationNumber =         10,
                                         iterationThr    =       0.01,
                                         sampleWidth     = [10., 10.],
                                         bin_size        = [ 1.,  1.],
                                         energy_type     =        'E',
                                         diffusion       = (1.0, 1.0),
                                         deconv_mode     =    'joint',
                                         n_dim           =          2,
                                         interMethod     = 'cubic')))
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


@mark.serial
def test_beersheba_exact_result3d(ICDIR, ICDATADIR, config_tmpdir):
    PATH_IN     = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC.h5")
    true_out    = os.path.join(ICDATADIR, "exact_Kr_deconvolution3d_with_MC.h5")
    PATH_OUT    = os.path.join(config_tmpdir, "beersheba_MC.h5")
    config_path = os.path.join(ICDIR, "config/beersheba.conf")
    conf        = configure(f'dummy {config_path}'.split())
    nevt_req    = all_events
    conf.update(dict(files_in                     = PATH_IN                   ,
                     file_out                     = PATH_OUT                  ,
                     event_range                  = nevt_req                  ,
                     deconv_params     = dict(
                                         q_cut           =         10,
                                         drop_dist       = [10., 10.],
                                         psf_fname       = f'{ICDIR}database/test_data/PSF_dst_sum_collapsed.h5',
                                         ecut            =       1e-3,
                                         iterationNumber =         10,
                                         iterationThr    =       0.01,
                                         sampleWidth     = [10., 10., 2.],
                                         bin_size        = [ 1.,  1., 1.],
                                         energy_type     =        'E',
                                         diffusion       = (1.0, 1.0, 0.3),
                                         deconv_mode     =    'joint',
                                         n_dim           =          3,
                                         interMethod     = 'linear')))
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


def test_beersheba_empty_input_file(ICDIR, config_tmpdir, ICDATADIR):
    PATH_IN  = os.path.join(ICDATADIR    , 'empty_hdst.h5')
    PATH_OUT = os.path.join(config_tmpdir, 'empty_voxels.h5')

    config_path = os.path.join(ICDIR, "config/beersheba.conf")
    conf        = configure(f'dummy {config_path}'.split())
    nevt_req    = all_events
    conf.update(dict(files_in                     = PATH_IN                   ,
                     file_out                     = PATH_OUT                  ,
                     event_range                  = nevt_req                  ,
                     deconv_params     = dict(
                                         q_cut           =         10,
                                         drop_dist       = [10., 10.],
                                         psf_fname       = f'{ICDIR}database/test_data/PSF_dst_sum_collapsed.h5',
                                         ecut            =       1e-3,
                                         iterationNumber =         10,
                                         iterationThr    =       0.01,
                                         sampleWidth     = [10., 10.],
                                         bin_size        = [ 1.,  1.],
                                         energy_type     =        'E',
                                         diffusion       = (1.0, 1.0),
                                         deconv_mode     =    'joint',
                                         n_dim           =          2,
                                         interMethod     = 'cubic')))

    beersheba(**conf)
