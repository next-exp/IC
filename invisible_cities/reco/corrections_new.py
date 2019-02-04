import numpy  as np
import pandas as pd
from   pandas    import DataFrame
from   pandas    import Series
from dataclasses import dataclass
from typing      import List
from typing      import Tuple
from typing      import Optional
from typing      import TypeVar
from .. evm.event_model         import Hit

@dataclass
class ASectorMap:  # Map in chamber sector containing average of pars
    chi2    : DataFrame
    e0      : DataFrame
    lt      : DataFrame
    e0u     : DataFrame
    ltu     : DataFrame
    mapinfo : Optional[Series]
    
def xy_correction_matrix_(X  : np.array,
                         Y  : np.array,
                         C  : DataFrame,
                         xr : Tuple[int, int],
                         yr : Tuple[int, int],
                         nx : int,
                         ny : int)->np.array:
    """
    Returns a correction matrix in XY computed from the
    map represented by the DataFrame C:
    """
    vx = sizeof_xy_voxel_(xr, nx)
    vy = sizeof_xy_voxel_(yr, ny)
    I = get_xy_indexes_(X, Y, abs(xr[0]), abs(yr[0]), vx, vy)
    return np.array([C[i[0]][i[1]] for i in I])

def sizeof_xy_voxel_(rxy : Tuple[int,int], nxy : int)->float:
    """
    rxy = (x0, x1) defines de interval in x (y), e.g, (x0, x1) = (-220, 220)
    nxy is the number of bins in x (y).
    then, the interval is shifted to positive values and divided by number of bins:
    x0' --> abs(x0) ; x1' = x0' + x1
    fr = x1' / n
    """
    x0 = abs(rxy[0])
    x1 = rxy[1] + x0
    fr = x1 / nxy
    return fr

def get_xy_indexes_(X  : np.array,
                    Y  : np.array,
                    x0 : float,
                    y0 : float,
                    fx : float,
                    fy : float)->List[Tuple[int,int]]:
    """Returns a list of pairs of ints, (x_i, y_i)"""
    x_i = ((X + x0) / fx).astype(int)
    y_i = ((Y + y0) / fy).astype(int)
    return list(zip(x_i, y_i))


def e0_xy_correction(E   : np.array,
                     X   : np.array,
                     Y   : np.array,
                     E0M : DataFrame,
                     xr  : Tuple[int, int],
                     yr  : Tuple[int, int],
                     nx  : int,
                     ny  : int)->np.array:
    """
    Computes the energy vector corrected by geometry in bins of XY.
    Parameters
    ----------
    E
        The uncorrected energy vector.
    X
        Array of X bins.
    Y
        Array of Y bins.
    E0M
        Map of geometrical corrections (E0 map).
    xr
        Range of X (e.g, (-220,220)).
    yr
        Range of Y (e.g, (-220,220)).
    nx
        Number of bins in X.
    ny
        Number of bins in Y.
    Returns
    -------
    np.array
        The corrected energy vector (by energy).
    """
    CE = xy_correction_matrix_(X, Y, E0M, xr, yr, nx, ny)
    return E / CE


def lt_xy_correction(E    : np.array,
                     X    : np.array,
                     Y    : np.array,
                     Z    : np.array,
                     LTM  : DataFrame,
                     xr   : Tuple[int, int],
                     yr   : Tuple[int, int],
                     nx   : int,
                     ny   : int)->np.array:
    """
    Computes the energy vector corrected by lifetime in bins of XY.
    Parameters
    ----------
    E
        The uncorrected energy vector.
    X
        Array of X bins.
    Y
        Array of Y bins.
    LTM
        Map of lifetime corrections (LT map).
    xr
        Range of X (e.g, (-220,220)).
    yr
        Range of Y (e.g, (-220,220)).
    nx
        Number of bins in X.
    ny
        Number of bins in Y.
    Returns
    -------
    np.array
        The corrected energy vector (by lifetime).
    """
    LT = xy_correction_matrix_(X, Y, LTM, xr, yr, nx, ny)
    return E * np.exp(Z / LT)

def temporal_correction(E   : np.array,
                        X   : np.array,
                        Y   : np.array,
                        tmp_pars : DataFrame,
                        xr  : Tuple[int, int],
                        yr  : Tuple[int, int],
                        nx  : int,
                        ny  : int)->np.array:
    raise NotImplementedError 

def read_maps(filename : str)->ASectorMap:
    chi2     = pd.read_hdf(filename, 'chi2')
    e0       = pd.read_hdf(filename, 'e0')
    e0u      = pd.read_hdf(filename, 'e0u')
    lt       = pd.read_hdf(filename, 'lt')
    ltu      = pd.read_hdf(filename, 'ltu')
    mapinfo  = pd.read_hdf(filename, 'mapinfo')
    return  ASectorMap(chi2, e0, lt, e0u, ltu, mapinfo)

def apply_all_correction(cmap:ASectorMap, hits : List[Hit]) -> List[Hit]:
    """ This function calls all three - geometric, lifetime and temporal corrections """ 
    raise NotImplementedError 
