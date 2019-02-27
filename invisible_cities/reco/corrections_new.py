import numpy  as np
import pandas as pd
from   pandas      import DataFrame
from   pandas      import Series
from   dataclasses import dataclass
from   typing      import Callable
from   typing      import List
from   typing      import Optional
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
    mapinfo : Optional[Series]
    t_evol  : Optional[DataFrame]

@dataclass
class FitMapValue:  # A ser of values of a FitMap
    chi2  : float
    e0    : float
    lt    : float
    e0u   : float
    ltu   : float

def amap_max(amap : ASectorMap)->FitMapValue:
    return FitMapValue(chi2 = amap.chi2.max().max(),
                       e0   = amap.e0  .max().max(),
                       lt   = amap.lt  .max().max(),
                       e0u  = amap.e0u .max().max(),
                       ltu  = amap.ltu .max().max())

def read_maps(filename : str)->ASectorMap:

    """
    Read 'filename' variable and creates ASectorMap class.
    If the map corresponds to a data run (run_number>0),
    ASectorMap will also contain a DataFrame with time evolution information.

    Parameters
    ----------
    filename : string
        Name of the file that contains the correction maps.

    Returns
    -------
    ASectorMap:

@dataclass
class ASectorMap:
    chi2    : DataFrame            # chi2 value for each bin
    e0      : DataFrame            # geometric map
    lt      : DataFrame            # lifetime map
    e0u     : DataFrame            # uncertainties of geometric map
    ltu     : DataFrame            # uncertainties of lifetime map
    mapinfo : Optional[Series]     # series with some info about the
    t_evol  : Optional[DataFrame]  # time evolution of some parameters
                                     (only for data)
    """

    chi2     = pd.read_hdf(filename, 'chi2')
    e0       = pd.read_hdf(filename, 'e0')
    e0u      = pd.read_hdf(filename, 'e0u')
    lt       = pd.read_hdf(filename, 'lt')
    ltu      = pd.read_hdf(filename, 'ltu')
    mapinfo  = pd.read_hdf(filename, 'mapinfo')

    if mapinfo.run_number>0:
        t_evol = pd.read_hdf(filename, 'time_evolution')
        maps   = ASectorMap(chi2, e0, lt, e0u, ltu, mapinfo, t_evol)

    else: maps = ASectorMap(chi2, e0, lt, e0u, ltu, mapinfo, None)

    return  maps


def maps_coefficient_getter(mapinfo : Series,
                            map_df  : DataFrame) -> Callable:
    """
    For a given correction map,
    it returns a function that yields the values of map
    for a given (X,Y) position.

    Parameters
    ----------
    mapinfo : Series
        Stores some information about the map
        (run number, number of X-Y bins, X-Y range)
    map_df : DataFrame
        DataFrame of a correction map (lt or e0)

    Returns
    -------
        A function that returns the value of the 'map_df' map
        for a given (X,Y) position
    """

    binsx   = np.linspace(mapinfo.xmin,mapinfo.xmax,mapinfo.nx+1)
    binsy   = np.linspace(mapinfo.ymin,mapinfo.ymax,mapinfo.ny+1)
    def get_maps_coefficient(x : np.array, y : np.array) -> np.array:
        ix = np.digitize(x, binsx)-1
        iy = np.digitize(y, binsy)-1
        return np.array([map_df.get(j, {}).get(i, np.nan) for i, j in zip(iy,ix)])
    return get_maps_coefficient


def correct_geometry_(CE : np.array) -> np.array:
    """
    Computes the geometric correction factor
    for a given correction coefficient

    Parameters
    ----------
    CE : np.array
        Array with geometric correction coefficients

    Returns
    -------
        An array with geometric correction factors
    """

    return 1/CE


def correct_lifetime_(Z : np.array, LT : np.array) -> np.array:
    """
    Computes the lifetime correction factor
    for a given correction coefficient

    Parameters
    ----------
    LT : np.array
        Array with lifetime correction coefficients

    Returns
    -------
        An array with lifetime correction factors
    """

    return np.exp(Z / LT)

def correct_temporal(E : np.array, X : np.array, Y : np.array, **kwargs):
    raise NotImplementedError

def e0_xy_corrections(E : np.array, X : np.array, Y : np.array, maps : ASectorMap)-> np.array:
    mapinfo = maps.mapinfo
    map_df  = maps.e0
    get_maps_coefficient= maps_coefficient_getter(mapinfo, map_df)
    CE  = get_maps_coefficient(X,Y)
    return correct_geometry_(E,CE)

def lt_xy_corrections(E : np.array, X : np.array, Y : np.array, Z : np.array, maps : ASectorMap)-> np.array:
    mapinfo = maps.mapinfo
    map_df  = maps.lt
    get_maps_coefficient= maps_coefficient_getter(mapinfo, map_df)
    LT  = get_maps_coefficient(X,Y)
    return correct_lifetime_(E,Z,LT)

def temporal_corrections(E : np.array, X : np.array, Y : np.array, Z : np.array, maps : ASectorMap)-> np.array:
    raise NotImplementedError

def apply_all_correction(cmap:ASectorMap, hits : List[Hit]) -> List[Hit]:
    """ This function calls all three - geometric, lifetime and temporal corrections """ 
    raise NotImplementedError 
