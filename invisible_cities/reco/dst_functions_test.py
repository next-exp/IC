import pandas as pd

from .. core.test_utils    import assert_dataframes_equal

from .  corrections        import Correction
from .  dst_functions      import load_dst
from .  dst_functions      import load_dsts
from .  dst_functions      import load_xy_corrections


def test_load_dst(Kr_dst_data):
    (filename, group, node), df = Kr_dst_data
    dst = load_dst(filename, group, node)

    assert_dataframes_equal(dst, df, False)


def test_load_dsts_single_file(Kr_dst_data):
    (filename, group, node), df = Kr_dst_data
    dst = load_dsts([filename], group, node)

    assert_dataframes_equal(dst, df, False)


def test_load_dsts_double_file(Kr_dst_data):
    (filename, group, node), df = Kr_dst_data
    dst = load_dsts([filename]*2, group, node)
    df  = pd.concat([df, df])

    assert_dataframes_equal(dst, df, False)


def test_load_xy_corrections(corr_toy_data):
    filename, true_data = corr_toy_data
    x, y, E, U, _ = true_data
    corr          = load_xy_corrections(filename)
    assert corr == Correction((x,y), E, U)
