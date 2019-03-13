import tables as tb
import pandas as pd
import numpy  as np
from tables import NoSuchNodeError
from tables import HDF5ExtError
import warnings

from .. core.exceptions    import TableMismatch
from .  table_io           import make_table


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


def _store_pandas_as_tables(h5out, df, group_name, table_name, compression='ZLIB4', descriptive_string=None, str_col_length=32):
    if len(df) == 0:
        warnings.warn(f' dataframe is empty', UserWarning)
    if '/'+group_name not in h5out:
        group = h5out.create_group(h5out.root,group_name)
    else:
        group = getattr(h5out.root,group_name)
    if table_name in group:
        table=getattr(group,table_name)
    else:
        tabledef={}
        for indx, colname in enumerate(df.columns):
            coltype = df[colname].dtype.name
            try:
                tabledef[colname] = tb.Col.from_type(coltype, pos = indx)
            except ValueError:
                tabledef[colname] = tb.StringCol(32, pos = indx)

        if descriptive_string is None:
            descriptive_string = ''
        table = make_table(h5out,
                           group       = group_name,
                           name        = table_name,
                           fformat     = tabledef,
                           description = descriptive_string,
                           compression = compression)
    if not np.array_equal(df.columns,table.colnames):
        raise TableMismatch
    for indx in df.index:
        tablerow = table.row
        for colname in table.colnames:
            tablerow[colname] = df.at[indx,colname]
        tablerow.append()
    table.flush()
