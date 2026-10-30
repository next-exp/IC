from typing import Callable

from functools import partial

import numpy  as np
import pandas as pd

from . dst_io import df_writer


def hits_writer(hdf5_file, group_name, table_name, *, compression=None):
    """
    Produce a single-argument function that writes the input DataFrame.
    """
    return partial( df_writer, hdf5_file
                  , group_name         = group_name
                  , table_name         = table_name
                  , descriptive_string = "Hits"
                  , columns_to_index   = ["event"]
                  , compression        = compression)
