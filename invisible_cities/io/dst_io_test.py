import pandas as pd

from ..core.testing_utils import assert_dataframes_close
from .      dst_io        import load_dst
from .      dst_io        import load_dsts


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


def test_load_dsts_throw_except_if_wrong_file(KrMC_kdst):
    tbl     = KrMC_kdst[0].file_info
    df_true = KrMC_kdst[0].true
    wrong_file =KrMC_kdst[1].filename
    cdst = load_dsts([tbl.filename, tbl.filename, wrong_file], tbl.group, tbl.node)
