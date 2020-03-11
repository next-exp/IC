import os

import numpy  as np
import tables as tb

from pytest        import mark
from numpy.testing import assert_allclose

from ..core.testing_utils import assert_dataframes_equal
from ..io.dst_io          import load_dst
from . kdst_io            import kr_writer
from . kdst_io            import xy_correction_writer
from . kdst_io            import xy_lifetime_writer
from . kdst_io            import psf_writer
from ..evm.event_model    import KrEvent


def test_Kr_writer(config_tmpdir, KrMC_kdst):
    filename = os.path.join(config_tmpdir, 'test_dst.h5')
    tbl      = KrMC_kdst[0].file_info
    df       = KrMC_kdst[0].true

    def dump_df(write, df):
        for evt_no in sorted(set(df.event)):
            evt = KrEvent(-1, -1)
            for row in df[df.event == evt_no].iterrows():
                for col in df.columns:
                    value = row[1][col]
                    try:
                        getattr(evt, col).append(value)
                    except AttributeError:
                        setattr(evt, col, value)

            evt.nS1 = int(evt.nS1)
            evt.nS2 = int(evt.nS2)
            for col in ("Z", "DT"):
                column = list(getattr(evt, col))
                setattr(evt, col, [])
                for i in range(evt.nS2):
                    s1_data = column[:evt.nS1 ]
                    column  = column[ evt.nS1:]
                    getattr(evt, col).append(s1_data)
            write(evt)

    with tb.open_file(filename, 'w') as h5out:
        write = kr_writer(h5out)
        dump_df(write, df)

    dst = load_dst(filename, group = tbl.group, node = tbl.node)
    assert_dataframes_equal(dst, df, False)


@mark.parametrize("writer".split(),
                  ((xy_correction_writer,),
                   (xy_lifetime_writer,)))
def test_xy_writer(config_tmpdir, corr_toy_data, writer):
    output_file = os.path.join(config_tmpdir, "test_corr.h5")

    _, (x, y, F, U, N) = corr_toy_data

    group = "Corrections"
    name  = "XYcorrections"

    with tb.open_file(output_file, 'w') as h5out:
        write = writer(h5out,
                       group      = group,
                       table_name = name)
        write(x, y, F, U, N)

    x, y    = np.repeat(x, y.size), np.tile(y, x.size)
    F, U, N = F.flatten(), U.flatten(), N.flatten()

    dst = load_dst(output_file,
                   group = group,
                   node  = name)
    assert_allclose(x, dst.x          .values)
    assert_allclose(y, dst.y          .values)
    assert_allclose(F, dst.factor     .values)
    assert_allclose(U, dst.uncertainty.values)
    assert_allclose(N, dst.nevt       .values)


def test_psf_writer(config_tmpdir):
    output_file = os.path.join(config_tmpdir, "test_psf.h5")

    xdim, ydim, zdim = 101, 101, 11
    xr, yr, zr = np.linspace(-50, 50, xdim), np.linspace(-20, 20, ydim), np.linspace(20, 30, zdim)
    x, y, z    = 3, 4, 5
    factors    = np.linspace(0, 1, xdim*ydim*zdim   ).reshape(xdim, ydim, zdim)
    nevt       = np.arange  (0,    xdim*ydim*zdim, 1).reshape(xdim, ydim, zdim)

    with tb.open_file(output_file, 'w') as h5out:
        write = psf_writer(h5out)
        write(xr, yr, zr, x, y, z, factors, nevt)

    psf = load_dst(output_file,
                   group = 'PSF',
                   node  = 'PSFs')

    xx, yy, zz = np.meshgrid(xr, yr, zr, indexing='ij')

    assert_allclose(xx            .flatten(), psf.xr    .values)
    assert_allclose(yy            .flatten(), psf.yr    .values)
    assert_allclose(zz            .flatten(), psf.zr    .values)
    assert_allclose(factors       .flatten(), psf.factor.values)
    assert_allclose(nevt          .flatten(), psf.nevt  .values)
    assert_allclose(np.full(factors.size, x), psf.x     .values)
    assert_allclose(np.full(factors.size, y), psf.y     .values)
    assert_allclose(np.full(factors.size, z), psf.z     .values)
