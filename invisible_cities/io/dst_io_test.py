import os
import string
import pytest
import re

import pandas as pd
import tables as tb

from ..core.testing_utils import assert_dataframes_close
from ..core.testing_utils import assert_dataframes_equal
from ..core.exceptions    import TableMismatch
from . dst_io             import load_dst
from . dst_io             import load_dsts
from . dst_io             import store_pandas_as_tables
from . dst_io             import _make_tabledef


from pytest                  import raises
from pytest                  import fixture
from hypothesis              import given
from hypothesis.extra.pandas import data_frames
from hypothesis.extra.pandas import column
from hypothesis.extra.pandas import range_indexes
from hypothesis.strategies   import text


@fixture()
def empty_dataframe(columns=['int_value', 'float_value', 'bool_value', 'str_value'], dtypes=['int32', 'float32', 'bool', 'object'], index=None):
    assert len(columns) == len(dtypes)
    df = pd.DataFrame(index=index)
    for c, d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)
    return df

@fixture()
def fixed_dataframe():
    return pd.DataFrame({'int':[0, 1], 'float':[10., 20.], 'string':['aa', 'bb']})

dataframe          = data_frames(index=range_indexes(min_size=1, max_size=5), columns=[column('int_value' , dtype = int    ),
                                                                                       column('float_val' , dtype = float  ),
                                                                                       column('bool_value', dtype = bool   )])

dataframe_diff     = data_frames(index=range_indexes(min_size=1, max_size=5), columns=[column('int_value' , dtype = int    ),
                                                                                      column('float_val' , dtype = float)])

strings_dataframe  = data_frames(index=range_indexes(min_size=1, max_size=5), columns=[column('str_val', elements=text(alphabet=string.ascii_letters, min_size=10, max_size=32))])


def test_load_dst(KrMC_kdst):
    df_read = load_dst(*KrMC_kdst[0].file_info)
    assert_dataframes_close(df_read, KrMC_kdst[0].true,
                            False  , rtol=1e-5)


def test_load_dsts_single_file(KrMC_kdst):
    tbl     = KrMC_kdst[0].file_info
    df_read = load_dsts([tbl.filename], tbl.group, tbl.node)

    assert_dataframes_close(df_read, KrMC_kdst[0].true,
                            False  , rtol=1e-5)


def test_load_dsts_double_file(KrMC_kdst):
    tbl     = KrMC_kdst[0].file_info
    df_true = KrMC_kdst[0].true
    df_read = load_dsts([tbl.filename]*2, tbl.group, tbl.node)
    df_true = pd.concat([df_true, df_true], ignore_index=True)

    assert_dataframes_close(df_read, df_true  ,
                            False  , rtol=1e-5)
    #assert index number unique (important for saving it to pytable)
    assert all(~df_read.index.duplicated())

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

@given(df=dataframe)
def test_store_pandas_as_tables_exact(config_tmpdir, df):
    filename   = config_tmpdir + 'dataframe_to_table_exact.h5'
    group_name = 'test_group'
    table_name = 'table_name_1'
    with tb.open_file(filename, 'w') as h5out:
        store_pandas_as_tables(h5out, df, group_name, table_name)
    df_read = load_dst(filename, group_name, table_name)
    assert_dataframes_close(df_read, df)

@given(df1=dataframe, df2=dataframe)
def test_store_pandas_as_tables_2df(config_tmpdir, df1, df2):
    filename   = config_tmpdir + 'dataframe_to_table_exact.h5'
    group_name = 'test_group'
    table_name = 'table_name_2'
    with tb.open_file(filename, 'w') as h5out:
        store_pandas_as_tables(h5out, df1, group_name, table_name)
        store_pandas_as_tables(h5out, df2, group_name, table_name)
    df_read = load_dst(filename, group_name, table_name)
    assert_dataframes_close(df_read, pd.concat([df1, df2]).reset_index(drop=True))

@given(df1=dataframe, df2=dataframe_diff)
def test_store_pandas_as_tables_raises_exception(config_tmpdir, df1, df2):
    filename   = config_tmpdir + 'dataframe_to_table_exact.h5'
    group_name = 'test_group'
    table_name = 'table_name_2'
    with tb.open_file(filename, 'w') as h5out:
        store_pandas_as_tables(h5out, df1, group_name, table_name)
        with raises(TableMismatch):
            store_pandas_as_tables(h5out, df2, group_name, table_name)

@given(df=strings_dataframe)
def test_strings_store_pandas_as_tables(config_tmpdir, df):
    filename   = config_tmpdir + 'dataframe_to_table_exact.h5'
    group_name = 'test_group'
    table_name = 'table_name_str'
    with tb.open_file(filename, 'w') as h5out:
        store_pandas_as_tables(h5out, df, group_name, table_name)
    df_read    = load_dst(filename, group_name, table_name)
    #we have to cast from byte strings to compare with original dataframe
    df_read.str_val = df_read.str_val.str.decode('utf-8')
    assert_dataframes_equal(df_read, df)

def test_make_tabledef(empty_dataframe):
    tabledef = _make_tabledef(empty_dataframe.to_records(index=False).dtype, 32)
    expected_tabledef = {'int_value'   : tb.Int32Col  (             shape=(), dflt=0    , pos=0),
                         'float_value' : tb.Float32Col(             shape=(), dflt=0    , pos=1),
                         'bool_value'  : tb.BoolCol   (             shape=(), dflt=False, pos=2),
                         'str_value'   : tb.StringCol (itemsize=32, shape=(), dflt=b''  , pos=3)}
    assert tabledef == expected_tabledef

def test_store_pandas_as_tables_raises_warning_empty_dataframe(config_tmpdir, empty_dataframe):
    filename   = config_tmpdir + 'dataframe_to_table_exact.h5'
    group_name = 'test_group'
    table_name = 'table_name_3'
    with tb.open_file(filename, 'w') as h5out:
        with pytest.warns(UserWarning, match='dataframe is empty'):
            store_pandas_as_tables(h5out, empty_dataframe, group_name, table_name)

@given(df=strings_dataframe)
def test_store_pandas_as_tables_raises_warning_long_string(config_tmpdir, df):
    filename   = config_tmpdir + 'dataframe_to_table_long_string.h5'
    group_name = 'test_group'
    table_name = 'table_name_lstr'
    with tb.open_file(filename, 'w') as h5out:
        with pytest.warns(UserWarning, match='dataframe contains strings longer than allowed'):
            store_pandas_as_tables(h5out, df, group_name, table_name, str_col_length=1)


@given(df=dataframe)
def test_store_pandas_as_tables_raises_TableMismatch_inconsistent_types(config_tmpdir, df):
    filename   = config_tmpdir + 'dataframe_to_table_exact.h5'
    group_name = 'test_group'
    table_name = 'table_name_inttype'
    with tb.open_file(filename, 'w') as h5out:
        store_pandas_as_tables(h5out, df, group_name, table_name)
        with raises(TableMismatch, match='dataframe numeric types not consistent with the table existing ones'):
            store_pandas_as_tables(h5out, df.astype(float), group_name, table_name)

@given(df1=dataframe, df2=dataframe)
def test_store_pandas_as_tables_unordered_df(config_tmpdir, df1, df2):
    filename   = config_tmpdir + 'dataframe_to_table_exact.h5'
    group_name = 'test_group'
    table_name = 'table_name_unordered'
    with tb.open_file(filename, 'w') as h5out:
        store_pandas_as_tables(h5out, df1, group_name, table_name)
        df2 = df2[['bool_value', 'int_value', 'float_val']]
        store_pandas_as_tables(h5out, df2, group_name, table_name)
    df_read = load_dst(filename, group_name, table_name)
    assert_dataframes_equal(df_read, pd.concat([df1, df2], sort=True).reset_index(drop=True))


def test_store_pandas_as_tables_index(config_tmpdir, fixed_dataframe):
    df = fixed_dataframe
    filename   = config_tmpdir + 'dataframe_to_table_index.h5'
    group_name = 'test_group'
    table_name = 'table_name'
    columns_to_index = ['int', 'float']
    with tb.open_file(filename, 'w') as h5out:
        store_pandas_as_tables(h5out, df, group_name, table_name, columns_to_index=columns_to_index)
        table = h5out.root[group_name][table_name]
        assert set(columns_to_index) == set(table.attrs.columns_to_index)

def test_store_pandas_as_tables_index_error(config_tmpdir, fixed_dataframe):
    df = fixed_dataframe
    filename   = config_tmpdir + 'dataframe_to_table_index.h5'
    group_name = 'test_group'
    table_name = 'table_name'
    columns_to_index = ['float_']
    with tb.open_file(filename, 'w') as h5out:
        with raises(KeyError, match=(re.escape(f"columns {columns_to_index} not present in the dataframe"))):
            store_pandas_as_tables(h5out, df, group_name, table_name, columns_to_index=columns_to_index)

def test_store_pandas_as_tables_no_index(config_tmpdir, fixed_dataframe):
    df = fixed_dataframe
    filename   = config_tmpdir + 'dataframe_to_table_index.h5'
    group_name = 'test_group'
    table_name = 'table_name'
    with tb.open_file(filename, 'w') as h5out:
        store_pandas_as_tables(h5out, df, group_name, table_name)
        table = h5out.root[group_name][table_name]
        assert 'columns_to_index' not in table.attrs
