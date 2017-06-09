import os

import tables as tb

from pytest import mark

from .. core.test_utils    import assert_dataframes_equal
from .. reco.dst_functions import load_dst
from .  dst_io             import kr_writer
from .  dst_io             import PersistentKrEvent


def test_Kr_writer(config_tmpdir, Kr_dst_data):
    filename = os.path.join(str(config_tmpdir), 'test_dst.h5')
    _, df    = Kr_dst_data

    def dump_df(write, df):
        for evt_no in sorted(set(df.event)):
            evt = PersistentKrEvent()
            for row in df[df.event == evt_no].iterrows():
                for col in df.columns:
                    value = row[1][col]
                    try:
                        getattr(evt, col).append(value)
                    except AttributeError:
                        setattr(evt, col, value)
            write(evt)

    with tb.open_file(str(filename), 'w') as h5out:
        write = kr_writer(h5out)
        dump_df(write, df)

    dst = load_dst(filename, group = "DST", node = "Events")
    assert_dataframes_equal(dst, df, False)
