import tables as tb
import pandas as pd
from tables import NoSuchNodeError
from tables import HDF5ExtError
import warnings



def load_dst(filename, group, node):
    try:
        with tb.open_file(filename) as h5in:
            try:
                table = getattr(getattr(h5in.root, group), node).read()
                return pd.DataFrame.from_records(table)
            except NoSuchNodeError:
                warnings.warn(f' warning:  {filename} not of kdst type')
    except HDF5ExtError:
        warnings.warn(f' warning:  {filename} corrupted')
    except IOError:
        warnings.warn(f' warning:  {filename} does not exist')
        

def load_dsts(dst_list, group, node):
    dsts = [load_dst(filename, group, node) for filename in dst_list]
    return pd.concat(dsts)
