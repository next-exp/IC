import tables as tb
import pandas as pd
import numpy  as np

import warnings

from    tables          import NoSuchNodeError
from    tables          import HDF5ExtError
from .. core.exceptions import TableMismatch
from .  table_io        import make_table

from    typing          import Optional
from    typing          import Sequence


def _decode_str_columns(df):
    to_string = lambda x: x.str.decode('utf-8') if x.dtype == np.object else x
    df        = df.apply(to_string)
    return df

def load_dst(filename, group, node):
    """load a kdst if filename, group and node correctly found"""
    try:
        with tb.open_file(filename) as h5in:
            try:
                table = getattr(getattr(h5in.root, group), node).read()
                return _decode_str_columns(pd.DataFrame.from_records(table))
            except NoSuchNodeError:
                warnings.warn(f' not of kdst type: file= {filename} ', UserWarning)
    except HDF5ExtError:
        warnings.warn(f' corrupted: file = {filename} ', UserWarning)
    except IOError:
        warnings.warn(f' does not exist: file = {filename} ', UserWarning)


def load_dsts(dst_list, group, node):
    dsts = [load_dst(filename, group, node) for filename in dst_list]
    return pd.concat(dsts, ignore_index=True)

def _make_tabledef(column_types : np.dtype, str_col_length : int) -> dict:
    tabledef = {}
    for indx, colname in enumerate(column_types.names):
        coltype = column_types[colname].name
        if coltype == 'object':
            tabledef[colname] = tb.StringCol(str_col_length, pos=indx)
        else:
            tabledef[colname] = tb.Col.from_type(coltype, pos=indx)
    return tabledef

def _check_castability(arr : np.ndarray, table_types : np.dtype):
    arr_types = arr.dtype

    if set(arr_types.names) != set(table_types.names):
        raise TableMismatch(f'dataframe differs from already existing table structure')

    for name in arr_types.names:
        if arr_types[name].name == 'object':
            max_str_length = max(map(len, arr[name]))
            if max_str_length > table_types[name].itemsize:
                warnings.warn(f'dataframe contains strings longer than allowed', UserWarning)

        elif not np.can_cast(arr_types[name], table_types[name], casting='same_kind'):
            raise TableMismatch(f'dataframe numeric types not consistent with the table existing ones')


def store_pandas_as_tables(h5out              : tb.file.File ,
                           df                 : pd.DataFrame ,
                           group_name         : str          ,
                           table_name         : str          ,
                           compression        : str = 'ZLIB4',
                           descriptive_string : str = ""     ,
                           str_col_length     : int = 32     ,
                           columns_to_index   : Optional[Sequence[str]] = None
                           ) -> None:
    """ The function writes a dataframe to open pytables file.
    Parameters:
    h5out              : open pytable file for writing
    df                 : DataFrame to be written
    group_name         : group name where table is to be saved)
                         (group is created if doesnt exist)
    table_name         : table name
                         (table is created if doesnt exist)
    compression        : compression type
    descriptive_string : table description
    str_col_length     : maximum length in characters of strings
    columns_to_index   : list of columns to be flagged for indexing
    """
    if group_name not in h5out.root:
        group = h5out.create_group(h5out.root, group_name)
    group = getattr(h5out.root, group_name)

    arr = df.to_records(index=False)

    if table_name not in group:
        tabledef = _make_tabledef(arr.dtype, str_col_length=str_col_length)
        table    =  make_table(h5out,
                               group       = group_name,
                               name        = table_name,
                               fformat     = tabledef,
                               description = descriptive_string,
                               compression = compression)
    else:
        table = getattr(group, table_name)

    data_types = table.dtype
    if len(arr) == 0:
        warnings.warn(f'dataframe is empty', UserWarning)
    else:
        _check_castability(arr, data_types)
        columns = list(data_types.names)
        arr = arr[columns].astype(data_types)
        table.append(arr)
        table.flush()

    if columns_to_index is not None:
        if set(columns_to_index).issubset(set(df.columns)):
            table.set_attr('columns_to_index', columns_to_index)
        else:
            not_found = list(set(columns_to_index).difference(set(df.columns)))
            raise KeyError(f'columns {not_found} not present in the dataframe')
