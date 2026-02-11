import pandas as pd
import numpy  as np
from scipy.interpolate import griddata
from .. core    .core_functions import in_range
from .. types   .symbols import NormMethod


def normalization(krmap    : pd.DataFrame,
                  method   : NormMethod,
                  xy_params : dict = None) -> float:
    """
    Given a input krypton map, normalizes the whole map to a certain value
    as a function of the chosen method.
    Parameters
    ----------
    krmap : pd.DataFrame
      Input krypton map whose bins are going to be normalized.
    method : NormMethod
      Method for normalization, defined in class function NormMethod.
    xy_params : dict
      Limits in x and y that define the region inside of which the normalization
      will be performed.
    Returns
    -------
    Normalization value using the chosen method.
    """

    krmap = krmap.dropna(subset=['mu'])

    anode = krmap[krmap.k == 0]

    if method is NormMethod.maximum:
        E_reference_max = krmap.mu.max()
        return E_reference_max

    if method is NormMethod.mean_chamber:
        E_reference_chamber = krmap.mu.mean()
        return E_reference_chamber

    if method is NormMethod.median_chamber:
        E_median_chamber = krmap.mu.median()
        return E_median_chamber

    if method is NormMethod.mean_anode:
        E_reference_anode = anode.mu.mean()
        return E_reference_anode

    if method is NormMethod.median_anode:
        E_median_anode = anode.mu.median()
        return E_median_anode

    mask_region = ( in_range(krmap.x, xy_params['x_low'], xy_params['x_high']) &
                    in_range(krmap.y, xy_params['y_low'], xy_params['y_high'])
                   ).values

    krmap = krmap[mask_region]

    if method is NormMethod.mean_region_chamber:
        E_reference_region = krmap.mu.mean()
        return E_reference_region

    if method is NormMethod.median_region_chamber:
        E_median_region = krmap.mu.median()
        return E_median_region

    anode = krmap[krmap.k == 0]

    if method is NormMethod.mean_region_anode:
        E_reference_slice_anode = anode.mu.mean()
        return E_reference_slice_anode

    if method is NormMethod.median_region_anode:
        E_median_region_anode = anode.mu.median()
        return E_median_region_anode



def apply_3Dmap(krmap        : pd.DataFrame,
                norm_method  : NormMethod,
                dt           : pd.core.series.Series,
                x            : pd.core.series.Series,
                y            : pd.core.series.Series,
                E            : pd.core.series.Series,
                xy_params    : dict = None,
                keV          : bool = False) -> pd.core.series.Series:
    """
    Applies a given krypton map normalized using the chosen normalization method
    to dt, x, y, E data from dataframes to get the corrected energy.

    -The corrected energy is computed using the expression

         E = S_2/S_0(x,y,z) E_0

         where S2 is the uncorrected S2 signal in pe,
         $E_0 = 41.55 keV$ is the known energy deposited by a 83mKr decay,
         and $S_0(x,y,z)$ is the average energy of 83mKr events
         from the corresponding voxel of the reference energy map.
    Parameters
    ----------
    krmap : pd.DataFrame
      Input krypton map whose bins are going to be normalized.
    method : NormMethod
      Method for normalization, defined in class function NormMethod.
    dt : pd.core.series.Series
      Drift time column from dataframe
    x : pd.core.series.Series
      x coordinate column from dataframe
    y : pd.core.series.Series
      y coordinate column from dataframe
    E : pd.core.series.Series
      S2e column from dataframe
    xy_params : dict
      Limits in x and y that define the region inside of which the normalization
      will be performed.
    keV : bool
      Boolean to decide whether the correction factor is applied in pes or keV
    Returns
    -------
    Ec : pd.core.series.Series
      Corrected energy

    """

    map_points = krmap['dt x y'.split()].values
    norm = normalization(krmap, norm_method, xy_params)

    data_points = np.stack([dt, x, y], axis = 1)
    E_interpolated_data = griddata(map_points, krmap.mu.values, data_points, method = 'nearest')

    correction_factor = norm/E_interpolated_data
    Ec = E * correction_factor

    if keV:
        Ec = Ec * (41.55 / norm)

    return Ec


def apply_correctionmap_inplace(kdst        : pd.DataFrame,
                                map3D       : pd.DataFrame,
                                norm_method : NormMethod,
                                xy_params   : dict,
                                col_name    : str,
                                keV         : bool = True ) -> pd.DataFrame:

    """
    Applies a given krypton map using apply_3Dmap to get as an output the same input
    kdst with a column for the corrected energy.
    Parameters
    ----------
    kdst : pd.DataFrame
      Dataframe to which the map is being applied.
    map3D : pd.DataFrame
      Krypton map to apply to the dataframe.
    norm_method : NormMethod
      Method chosen to normalize the krypton map.
    xy_params : dict
      Limits in x and y that define the region inside of which the normalization
      will be performed.
      Must be a dictionary: {'x_high': , 'x_low': , 'y_high': , 'y_low': }
    col_name : str
      Name of the column where the corrected energy will be in the new dataframe.
    keV : bool
      Boolean to decide whether the correction factor is applied in keV or pe
    Returns
    -------
    kdst : pd.DataFrame
      Dataframe containing the same data as the input dataframe with one more column named
      col_name filled with the corrected energy.

    """

    corrected_energy = apply_3Dmap(map3D, norm_method, kdst.DT, kdst.X, kdst.Y, kdst.S2e, xy_params = xy_params, keV = keV)

    kdst[col_name] = corrected_energy.values


    return kdst
