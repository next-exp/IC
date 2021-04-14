import os

import numpy  as np
import tables as tb

from numpy.testing import assert_allclose

from ..core.testing_utils import assert_dataframes_equal
from ..io.dst_io          import load_dst
from . kdst_io            import kr_writer

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
