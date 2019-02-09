import numpy  as np
import pandas as pd
from   pandas      import DataFrame
from   pandas      import Series
from   dataclasses import dataclass
from   typing      import Callable
from   typing      import List
from   enum        import auto
from .. types.ic_types  import AutoNameEnumBase
from .. evm.event_model import Hit

@dataclass
class ASectorMap:  # Map in chamber sector containing average of pars
    chi2    : DataFrame
    e0      : DataFrame
    lt      : DataFrame
    e0u     : DataFrame
    ltu     : DataFrame
    mapinfo : Series



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
class CorrectionsDF(AutoNameEnumBase):
    e0      = auto()
    lt      = auto()
    e0u     = auto()
    ltu     = auto()

def read_maps(filename : str)->ASectorMap:
    chi2     = pd.read_hdf(filename, 'chi2')
    e0       = pd.read_hdf(filename, 'e0')
    e0u      = pd.read_hdf(filename, 'e0u')
    lt       = pd.read_hdf(filename, 'lt')
    ltu      = pd.read_hdf(filename, 'ltu')
    mapinfo  = pd.read_hdf(filename, 'mapinfo')
    return  ASectorMap(chi2, e0, lt, e0u, ltu, mapinfo)

def maps_coefficient_getter(maps : ASectorMap, corrections_df : CorrectionsDF) -> Callable:
    mapinfo = maps.mapinfo
    binsx   = np.linspace(mapinfo.xmin,mapinfo.xmax,mapinfo.nx+1)
    binsy   = np.linspace(mapinfo.ymin,mapinfo.ymax,mapinfo.ny+1)
    df      = getattr(maps,corrections_df.value)
    def get_maps_coefficient(x : np.array, y : np.array) -> np.array:
        ix = np.digitize(x, binsx)-1
        iy = np.digitize(y, binsy)-1
        return np.array([df.get(j, {}).get(i, np.nan) for i, j in zip(iy,ix)])
    return get_maps_coefficient
def apply_all_correction(cmap:ASectorMap, hits : List[Hit]) -> List[Hit]:
    """ This function calls all three - geometric, lifetime and temporal corrections """ 
    raise NotImplementedError 
