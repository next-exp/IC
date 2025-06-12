import os

import numpy  as np
import tables as tb
import pandas as pd

from pytest import fixture

from .. cities.eutropia        import eutropia
from .. core  .configure       import configure
from .. core  .system_of_units import mm
from .. core  .testing_utils   import assert_tables_equality
from .. core  .testing_utils   import ignore_warning


@fixture(scope="module")
def kr_hits(ICDATADIR):
    return os.path.join(ICDATADIR, "Kr_hits_for_psf_*.h5")


@fixture(scope="module")
def fast_psf_config(ICDATADIR):
    return dict( files_in    = os.path.join(ICDATADIR, "Kr_hits_for_psf_0.h5")
               , run_number  = -7000
               , zbins       = (   0, 550)
               , xsectors    = (-200, 200)
               , ysectors    = (-200, 200)
               , bin_size_xy = 200 * mm
               )


@ignore_warning.no_config_group
def test_eutropia_contains_tables(fast_psf_config, output_tmpdir):
    output_filename = os.path.join(output_tmpdir, "psf_tables.h5")
    conf = configure('eutropia $ICTDIR/invisible_cities/config/eutropia.conf'.split())
    conf.update(fast_psf_config)
    conf.update(dict(file_out=output_filename))

    eutropia(**conf)

    with tb.open_file(output_filename) as file:
        assert "PSF"  in file.root
        assert "PSFs" in file.root.PSF

        assert "Run"     in file.root
        assert "runInfo" in file.root.Run
        assert "events"  in file.root.Run


@ignore_warning.no_config_group
def test_eutropia_output_types(fast_psf_config, output_tmpdir):
    output_filename = os.path.join(output_tmpdir, "psf_tables.h5")
    conf = configure('eutropia $ICTDIR/invisible_cities/config/eutropia.conf'.split())
    conf.update(fast_psf_config)
    conf.update(dict(file_out=output_filename))

    eutropia(**conf)

    df = pd.read_hdf(output_filename, "/PSF/PSFs")
    dtypes = [float]*7 + [np.uint]
    assert np.all(df.dtypes.values == dtypes)


@ignore_warning.no_config_group
def test_eutropia_run_info(fast_psf_config, output_tmpdir):
    output_filename = os.path.join(output_tmpdir, "psf_events.h5")
    conf = configure('eutropia $ICTDIR/invisible_cities/config/eutropia.conf'.split())
    conf.update(fast_psf_config)
    conf.update(dict(file_out=output_filename))

    eutropia(**conf)

    input_filename = fast_psf_config["files_in"]
    with tb.open_file(input_filename) as file_in:
        with tb.open_file(output_filename) as file_out:
            for node in "events runInfo".split():
                got      = getattr(file_out.root.Run, node).read()
                expected = getattr(file_in .root.Run, node).read()
                assert all(got == expected)


@ignore_warning.no_config_group
def test_eutropia_centers(output_tmpdir):
    file_out    = os.path.join(output_tmpdir, "psf_centers.h5")

    conf = configure("eutropia invisible_cities/config/eutropia.conf".split())
    conf.update(dict(file_out = file_out))

    eutropia(**conf)

    x_edges = np.asarray(conf["xsectors"])
    y_edges = np.asarray(conf["ysectors"])
    z_edges = np.asarray(conf["zbins"   ])
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2

    with tb.open_file(file_out) as output_file:
        table = output_file.root.PSF.PSFs
        assert np.in1d(table.col("x"), x_centers).all()
        assert np.in1d(table.col("y"), y_centers).all()
        assert np.in1d(table.col("z"), z_centers).all()


@ignore_warning.no_config_group
def test_eutropia_exact_result(ICDATADIR, output_tmpdir):
    file_out    = os.path.join(output_tmpdir, "exact_result_eutropia.h5")
    true_output = os.path.join(ICDATADIR    , "exact_result_eutropia.h5")

    conf = configure("eutropia invisible_cities/config/eutropia.conf".split())
    conf.update(dict(file_out = file_out))

    eutropia(**conf)

    tables = "PSF/PSFs",
    with tb.open_file(true_output)  as true_output_file:
        with tb.open_file(file_out) as      output_file:
            for table in tables:
                got      = getattr(     output_file.root, table)
                expected = getattr(true_output_file.root, table)
                assert_tables_equality(got, expected)
