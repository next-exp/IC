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

def correct_geometry_(E : np.array, CE : np.array) -> np.array:
    return E/CE

def correct_lifetime_(E : np.array, Z : np.array, LT : np.array) -> np.array:
    return E * np.exp(Z / LT)

def correct_temporal(E : np.array, X : np.array, Y : np.array, **kwargs):
    raise NotImplementedError

def e0_xy_corrections(E : np.array, X : np.array, Y : np.array, maps : ASectorMap)-> np.array:
    get_maps_coefficient= maps_coefficient_getter(maps, CorrectionsDF.e0)
    CE  = get_maps_coefficient(X,Y)
    return correct_geometry_(E,CE)

def lt_xy_corrections(E : np.array, X : np.array, Y : np.array, Z : np.array, maps : ASectorMap)-> np.array:
    get_maps_coefficient= maps_coefficient_getter(maps, CorrectionsDF.lt)
    LT  = get_maps_coefficient(X,Y)
    return correct_lifetime_(E,Z,LT)

def temporal_corrections(E : np.array, X : np.array, Y : np.array, Z : np.array, maps : ASectorMap)-> np.array:
    raise NotImplementedError

def apply_all_correction(cmap:ASectorMap, hits : List[Hit]) -> List[Hit]:
    """ This function calls all three - geometric, lifetime and temporal corrections """ 
    raise NotImplementedError 
