import os
import numpy  as np
import tables as tb
import pandas as pd
from . dst_io              import load_dst
from numpy.testing import assert_allclose

from .. core.testing_utils    import assert_dataframes_equal
# from ..types.ic_types      import xy
# from ..evm.event_model     import Cluster
# from ..evm.event_model     import Hit
# from ..evm.event_model     import HitCollection
# from . dst_io              import hits_writer
from . dst_io                 import load_dst
from . dst_io                 import load_dsts


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
