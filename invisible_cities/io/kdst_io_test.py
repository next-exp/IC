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
from ..evm.event_model    import KrEvent


def test_Kr_writer(config_tmpdir, Kr_dst_data):
    filename = os.path.join(config_tmpdir, 'test_dst.h5')
    _, df    = Kr_dst_data

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
            write(evt)

    with tb.open_file(filename, 'w') as h5out:
        write = kr_writer(h5out)
        dump_df(write, df)

    dst = load_dst(filename, group = "DST", node = "Events")
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
