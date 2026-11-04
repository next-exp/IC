import os

import tables as tb

from ..core.testing_utils import assert_dataframes_equal
from ..io.dst_io          import load_dst
from . kdst_io            import kdst_writer


def test_kdst_writer(config_tmpdir, KrMC_kdst):
    filename = os.path.join(config_tmpdir, 'test_dst.h5')
    tbl      = KrMC_kdst[0].file_info
    df       = KrMC_kdst[0].true

    with tb.open_file(filename, 'w') as h5out:
        kdst_writer(h5out)(df)

    dst = load_dst(filename, group = tbl.group, node = tbl.node)
    assert_dataframes_equal(dst, df, check_dtype=False)
