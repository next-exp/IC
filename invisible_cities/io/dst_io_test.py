import os
import numpy  as np
import tables as tb
import pandas as pd
from . dst_io              import load_dst
from numpy.testing import assert_allclose

from .. core.testing_utils    import assert_dataframes_close
# from ..types.ic_types      import xy
# from ..evm.event_model     import Cluster
# from ..evm.event_model     import Hit
# from ..evm.event_model     import HitCollection
# from . dst_io              import hits_writer
from . dst_io                 import load_dst
from . dst_io                 import load_dsts


def test_load_dst(KrMC_kdst):
    # The function load_dst is actually used in the fixture
    df_read = KrMC_kdst.read
    assert_dataframes_close(df_read, KrMC_kdst.true, False)


def test_load_dsts_single_file(KrMC_kdst):
    tbl     = KrMC_kdst.file_info
    df_read = load_dsts([tbl.filename], tbl.group, tbl.node)

    assert_dataframes_close(df_read, KrMC_kdst.true, False)


def test_load_dsts_double_file(KrMC_kdst):
    tbl     = KrMC_kdst.file_info
    df_true = KrMC_kdst.true
    df_read = load_dsts([tbl.filename]*2, tbl.group, tbl.node)
    df_true = pd.concat([df_true, df_true])

    assert_dataframes_close(df_read, df_true, False)
