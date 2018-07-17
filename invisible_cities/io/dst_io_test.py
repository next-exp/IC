import os
import pandas as pd

from ..core.testing_utils import assert_dataframes_close
from .      dst_io        import load_dst
from .      dst_io        import load_dsts

import warnings
import pytest

def test_load_dst(KrMC_kdst):
    df_read = load_dst(*KrMC_kdst[0].file_info)
    assert_dataframes_close(df_read, KrMC_kdst[0].true, False)


def test_load_dsts_single_file(KrMC_kdst):
    tbl     = KrMC_kdst[0].file_info
    df_read = load_dsts([tbl.filename], tbl.group, tbl.node)

    assert_dataframes_close(df_read, KrMC_kdst[0].true, False)


def test_load_dsts_double_file(KrMC_kdst):
    tbl     = KrMC_kdst[0].file_info
    df_true = KrMC_kdst[0].true
    df_read = load_dsts([tbl.filename]*2, tbl.group, tbl.node)
    df_true = pd.concat([df_true, df_true])

    assert_dataframes_close(df_read, df_true, False)

def test_load_dsts_reads_good_kdst(ICDATADIR):
    good_file = "Kr83_nexus_v5_03_00_ACTIVE_7bar_10evts_KDST.h5"
    good_file = os.path.join(ICDATADIR, good_file)
    group = "DST"
    node  = "Events"
    load_dsts([good_file], group, node)

def test_load_dsts_warns_not_of_kdst_type(ICDATADIR):
    good_file = "Kr83_nexus_v5_03_00_ACTIVE_7bar_10evts_KDST.h5"
    good_file = os.path.join(ICDATADIR, good_file)

    wrong_file = "kdst_5881_map_lt.h5"
    wrong_file = os.path.join(ICDATADIR, wrong_file)
    group = "DST"
    node  = "Events"
    with pytest.warns(UserWarning, match='not of kdst type'):
        load_dsts([good_file, wrong_file], group, node)

def test_load_dsts_warns_if_not_existing_file(ICDATADIR):
    good_file = "Kr83_nexus_v5_03_00_ACTIVE_7bar_10evts_KDST.h5"
    good_file = os.path.join(ICDATADIR, good_file)

    wrong_file = "non_existing_file.h5"

    group = "DST"
    node  = "Events"
    with pytest.warns(UserWarning, match='does not exist'):
        load_dsts([good_file, wrong_file], group, node)
