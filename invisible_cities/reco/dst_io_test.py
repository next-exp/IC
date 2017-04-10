import os

from invisible_cities.core.test_utils    import assert_dataframes_equal
from invisible_cities.reco.dst_io        import Kr_writer, PointLikeEvent
from invisible_cities.reco.dst_functions import load_dst

from pytest import mark


@mark.parametrize(  'filename,          with_',
                  (('test_dst_auto.h5', True ),
                   ('test_dst_manu.h5', False)))
def test_Kr_writer(config_tmpdir, filename, with_, Kr_dst_data):
    filename = os.path.join(str(config_tmpdir), filename)
    _, df    = Kr_dst_data

    group = "DST"
    table = "EVT"
    def dump_df(write, df):
        for evt_no in sorted(set(df.event)):
            evt = PointLikeEvent()
            for row in df[df.event == evt_no].iterrows():
                for col in df.columns:
                    value = row[1][col]
                    try:
                        getattr(evt, col).append(value)
                    except AttributeError:
                        setattr(evt, col, value)
            write(evt)

    if with_: # Close implicitly with context manager
        with Kr_writer(filename, group, table_name=table) as write:
            dump_df(write, df)
    else: # Close manually
        write = Kr_writer(filename, group, table_name=table)
        dump_df(write, df)
        write.close()

    dst = load_dst(filename, group, table)
    assert_dataframes_equal(dst, df, False)
