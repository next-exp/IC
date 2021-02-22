import numpy  as np
import pandas as pd
import warnings
from typing import Callable
from typing import Optional

from functools import partial
from .. core                import system_of_units as units
from .. core.core_functions import binedges_from_bincenters

from .. io.dst_io  import load_dst


def read_lighttable(fname      : str,
                    group_name : str,
                    el_gap     : Optional[float]=None,
                    active_r   : Optional[float]=None):
    """ A helper function to extract dataframes and configuration infro from files.
    Parameters:
        :fname:      str
              lighttable filename (full path)
        :group_name: str
            group name under which table is saved
        :el_gap:   float (optional)
            optional set of el gap width. Warning is issued if it differs
            from the config one
        :active_r: float (optional)
            optional set of active radius. Warning is issued in case it differs
            from the config one
    Returns: tuple
        :lt_df:     pd.DataFrame
            light table values in pd.DataFrame format
        :config_df: pd.DataFrame
            light table config in pd.DataFrame format
        :el_gap:   float
            input el_gap or el_gap read from file
        :active_r: float
            input active_r or active_r read from file
    """

    lt_df      = load_dst(fname, group_name, "LightTable")
    config_df  = load_dst(fname, group_name, "Config").set_index('parameter')
    el_gap_f   = float(config_df.loc["EL_GAP"    ].value) * units.mm
    active_r_f = float(config_df.loc["ACTIVE_rad"].value) * units.mm
    if el_gap and (el_gap != el_gap_f):
        warnings.warn('el_gap parameter mismatch, setting to user defined one',
                      UserWarning)
    else:
        el_gap = el_gap_f

    if active_r and (active_r != active_r_f):
            warnings.warn('active_r parameter mismatch, setting to user defined one',
                          UserWarning)
    else:
        active_r = active_r_f

    return lt_df, config_df, el_gap, active_r


def create_lighttable_function(filename : str,
                               active_r : Optional[float]=None)->Callable:
    """From a lighttable file, it returns a function of (x, y) for S2 signal
    or (x, y, z) for S1 signal type. Signal type is read from the table.
    Parameters:
        :filename: str
            name of the lighttable file
    Returns:
        :get_lt_values: Callable
            this is a function which access the desired value inside
            the lighttable. The lighttable values would be the nearest
            points to the input positions. If the input positions are
            outside the lighttable boundaries, zero is returned.
            Input values must be vectors of same lenght, I. The output
            shape will be (I, number_of_pmts).
    """
    lt, config, el_gap, act_r = read_lighttable(filename, 'LT', active_r=active_r)
    sensor = config.loc["sensor"].value
    lt     = lt.drop(sensor + "_total", axis=1) # drop total column

    def get_lt_values(xs, ys, zs):
        if len(zs) == 1:
            zs = np.full(len(xs), zs)
        if not (len(xs) == len(ys) == len(zs)):
            raise Exception("input arrays must be of same shape")
        sel = (np.sqrt(xs**2 + ys**2) <= act_r) & (zbins[0]<=zs) & (zs<=zbins[-1]) #inside bins
        xindices = pd.cut(xs[sel], xbins, include_lowest=True, labels=xcenters)
        yindices = pd.cut(ys[sel], ybins, include_lowest=True, labels=ycenters)
        zindices = pd.cut(zs[sel], zbins, include_lowest=True, labels=zcenters)
        indices  = pd.Index(zip(xindices, yindices, zindices), name=("x", "y", "z"))
        values   = np.zeros((len(xs), nsensors))
        values[sel] = lt.reindex(indices, fill_value=0)
        return values

    if lt.get("z") is None:
        lt.loc[:, "z"] = 1 # add fake z
        had_z = False
    else:
        had_z = True

    lt = lt.set_index(["x", "y", "z"])
    nsensors = lt.shape[-1]

    xcenters = np.unique(lt.index.get_level_values('x'))
    ycenters = np.unique(lt.index.get_level_values('y'))
    zcenters = np.unique(lt.index.get_level_values('z'))

    # A trick to make the function work with arbitrary act_r, extending the range if necessary.
    # Note that binedges_from_bincenters does not accept range smaller than centers range
    range_x = (min(xcenters[0], -act_r), max(xcenters[-1], act_r)) # centers are already ordered
    range_y = (min(ycenters[0], -act_r), max(ycenters[-1], act_r))
    xbins = binedges_from_bincenters(xcenters, range=range_x)
    ybins = binedges_from_bincenters(ycenters, range=range_y)
    zbins = binedges_from_bincenters(zcenters)

    if had_z: return get_lt_values
    else    : return partial(get_lt_values, zs=np.array([1]))
