import tables as tb
import pandas as pd
import numpy  as np

import warnings

from    tables             import NoSuchNodeError
from    tables             import HDF5ExtError
from .. core.exceptions    import TableMismatch
from .  table_io           import make_table
from    typing             import Optional

def load_dst(filename, group, node):
    """load a kdst if filename, group and node correctly found"""
    try:
        with tb.open_file(filename) as h5in:
            try:
                table = getattr(getattr(h5in.root, group), node).read()
                return pd.DataFrame.from_records(table)
            except NoSuchNodeError:
                warnings.warn(f' not of kdst type: file= {filename} ', UserWarning)
    except HDF5ExtError:
        warnings.warn(f' corrupted: file = {filename} ', UserWarning)
    except IOError:
        warnings.warn(f' does not exist: file = {filename} ', UserWarning)


def load_dsts(dst_list, group, node):
    dsts = [load_dst(filename, group, node) for filename in dst_list]
    return pd.concat(dsts)

def _make_tabledef(column_types : pd.Series, str_col_length : int=32) -> dict:
    tabledef = {}
    for indx, colname in enumerate(column_types.index):
        coltype = column_types[colname].name
        if coltype == 'object':
            tabledef[colname] = tb.StringCol(str_col_length, pos=indx)
        else:
            tabledef[colname] = tb.Col.from_type(coltype, pos=indx)
    return tabledef

def _store_pandas_as_tables(h5out : tb.file.File, df : pd.DataFrame, group_name : str, table_name : str, compression : str='ZLIB4', descriptive_string : [str]="", str_col_length : int=32) -> None:
    if len(df) == 0:
        warnings.warn(f'dataframe is empty', UserWarning)
    if group_name not in h5out.root:
        group = h5out.create_group(h5out.root, group_name)

    group = getattr(h5out.root, group_name)
    if table_name not in group:
        tabledef = _make_tabledef(df.dtypes)
        table    =  make_table(h5out,
                               group       = group_name,
                               name        = table_name,
                               fformat     = tabledef,
                               description = descriptive_string,
                               compression = compression)

    table = getattr(group, table_name)
    if not np.array_equal(df.columns, table.colnames):
        raise TableMismatch(f'dataframe differs from already existing table structure')
    for indx in df.index:
        tablerow = table.row
        for colname in table.colnames:
            tablerow[colname] = df.at[indx, colname]
        tablerow.append()
    table.flush()
