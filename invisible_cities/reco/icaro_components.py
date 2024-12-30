import numpy  as np
import pandas as pd

from   typing               import Tuple, Optional
from   sklearn.linear_model import RANSACRegressor

from .  corrections         import ASectorMap
from .  corrections         import apply_geo_correction

from .. types.symbols       import type_of_signal
from .. types.symbols       import Strictness
from .. types.symbols       import NormStrategy
from .. core.core_functions import all_in_range
from .. core.core_functions import in_range
from .. core.core_functions import shift_to_bin_centers
from .. core.fit_functions  import fit
from .. core.fit_functions  import gauss

def selection_nS_mask_and_checking(dst        : pd.DataFrame                         ,
                                   column     : type_of_signal                       ,
                                   input_mask : Optional[np.array]  = None           ,
                                   interval   : Tuple[float, float] = [0,1]          ,
                                   strictness : Strictness = Strictness.stop_proccess
                                   )->np.array:
    """
    Selects nS1(or nS2) == 1 for a given kr dst and
    returns the mask. It also computes selection efficiency,
    checking if the value is within a given interval, and
    saves histogram parameters.
    Parameters
    ----------
    dst: pd.Dataframe
        Krypton dst dataframe.
    column: type_of_signal
        The function can be appplied over nS1 or nS2.
    input_mask: np.array (Optional)
        Selection mask of the previous cut. If this is the first selection
        /no previous maks is input, input_mask is set to be an all True array.
    interval: length-2 tuple
        If the selection efficiency is out of this interval
        the map production will abort/just warn, depending on "strictness".
    sstrictness: Strictness
        If 'warning', function returns a False if the criteria
        is not matched. If 'stop_proccess' it raises an exception.
    Returns
    ----------
        A mask corresponding to the selected events.
    """
    input_mask = input_mask if input_mask is not None else [True] * len(dst)
    mask             = np.zeros_like(input_mask)
    mask[input_mask] = dst.loc[input_mask, column.value] == 1

    nevts_after      = dst[mask]      .event.nunique()
    nevts_before     = dst[input_mask].event.nunique()
    eff              = nevts_after / nevts_before
    all_in_range(data         = np.array(eff),
                 minval       = interval[0]  ,
                 maxval       = interval[1]  ,
                 display_name = column.value ,
                 strictness   = strictness   ,
                 right_closed = True)

    return mask


def band_selector_and_check(dst         : pd.DataFrame                                 ,
                            boot_map    : ASectorMap                                   ,
                            norm_strat  : NormStrategy               = NormStrategy.max,
                            input_mask  : np.ndarray                 = None            ,
                            range_DT    : Tuple[np.array, np.array]  = (10, 1300)      ,
                            range_E     : Tuple[np.array, np.array]  = (10.0e+3,14e+3) ,
                            nsigma_sel  : float                      = 3.5             ,
                            eff_interval: Tuple[float, float]        = [0,1]           ,
                            strictness  : Strictness = Strictness.stop_proccess
                            )->np.array:
    """
    This function returns a selection of the events that
    are inside the Kr E vz Z band, and checks
    if the selection efficiency is correct.

    Parameters
    ----------
    dst : pd.DataFrame
        Krypton dataframe.
    boot_map: str
        Name of bootstrap map file.
    norm_strt: norm_strategy
        Provides the desired normalization to be used.
    mask_input: np.array
        Mask of the previous selection cut.
    range_DT: Tuple[np.array, np.array]
        Range in Z-axis
    range_E: Tuple[np.array, np.array]
        Range in Energy-axis
    nsigma_sel: float
        Number of sigmas to set the band width
    eff_interval
        Limits of the range where selection efficiency
        is considered correct.
    Returns
    ----------
        A  mask corresponding to the selection made.
    """
    if input_mask is None:
        input_mask = [True] * len(dst)

    dst_sel = dst[input_mask]

    emaps = apply_geo_correction(boot_map, norm_strat  = norm_strat)
    E0    = dst_sel.S2e.values * emaps(dst_sel.X.values,
                                       dst_sel.Y.values)

    sel_krband = np.zeros_like(input_mask)
    sel_krband[input_mask] = selection_in_band(dst_sel.DT, E0     ,
                                               range_dt = range_DT,
                                               range_e  = range_E ,
                                               nsigma   = nsigma_sel)

    effsel   = dst[sel_krband].event.nunique()/dst[input_mask].event.nunique()

    all_in_range(data         = np.array(effsel)   ,
                 minval       = eff_interval[0]    ,
                 maxval       = eff_interval[1]    ,
                 display_name = "DT-band selection",
                 strictness   = strictness         ,
                 right_closed = True)

    return sel_krband


def selection_in_band(dt        : np.ndarray         ,
                      e         : np.ndarray         ,
                      range_dt  : Tuple[float, float],
                      range_e   : Tuple[float, float],
                      nsigma    : float   = 3.5) ->np.array:
    """
    This function returns a mask for the selection of the events that are inside the Kr E vz Z

    Parameters
    ----------
    dt: np.array
        axial (dt/z) values
    e: np.array
        energy values
    range_dt: Tuple[np.array, np.array]
        Range in DT-axis
    range_e: Tuple[np.array, np.array]
        Range in Energy-axis
    nsigma: float
        Number of sigmas to set the band width
    Returns
    ----------
        A  mask corresponding to the selection made.
    """
    # Reshapes and flattens are needed for RANSAC function

    dt_sel = dt[in_range(dt, *range_dt)]
    e_sel  = e [in_range( e, *range_e )]

    res_fit      = RANSACRegressor().fit(dt_sel.reshape(-1,1),
                                         np.log(e_sel).reshape(-1, 1))
    sigma        = sigma_estimation(dt_sel, np.log(e_sel), res_fit)

    prefict_fun  = lambda dt: res_fit.predict(dt.reshape(-1, 1)).flatten()
    upper_band   = lambda dt: prefict_fun(dt) + nsigma * sigma
    lower_band   = lambda dt: prefict_fun(dt) - nsigma * sigma
    sel_inband   = in_range(np.log(e), lower_band(dt), upper_band(dt))

    return  sel_inband

def sigma_estimation(dt     : np.ndarray     ,
                     e      : np.ndarray     ,
                     res_fit: RANSACRegressor
                    ) -> float:
    """
    This function estimates the sigma from the residuals to a line fit

    Parameters
    ----------
    dt: np.array
        axial (dt/z) values
    e: np.array
        energy values
    res_fit: RANSACRegressor
        RANSAC object fitted to the data

    Returns
    ----------
        The sigma of the residuals as a float.
    """
    # Reshapes and flattens are needed for RANSAC function

    in_mask      = res_fit.inlier_mask_
    e_predict    = res_fit.predict(dt[in_mask].reshape(-1, 1)).flatten()
    residuals_ln = e[in_mask] - e_predict
    resy, resx   = np.histogram(residuals_ln, 100)
    resx         = shift_to_bin_centers(resx)
    fitres       = fit(gauss, resx, resy, seed=[4e3,0,10])
    fitsigma     = fitres.values[2]

    return fitsigma
