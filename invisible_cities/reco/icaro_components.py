import copy
import itertools

import numpy        as np
import pandas       as pd
import scipy.stats  as stats

from   typing              import Tuple, Optional
from ..types.symbols       import KrFitFunction
from ..core.core_functions import shift_to_bin_centers
from .. evm.ic_containers  import FitFunction

from .  corrections         import ASectorMap
from .  corrections         import apply_geo_correction

from .. types.symbols       import type_of_signal
from .. types.symbols       import Strictness
from .. types.symbols       import NormStrategy
from .. core.core_functions import check_if_values_in_interval


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
    mask[input_mask] = getattr(dst[input_mask], column.value) == 1

    nevts_after      = dst[mask]      .event.nunique()
    nevts_before     = dst[input_mask].event.nunique()
    eff              = nevts_after / nevts_before
    check_if_values_in_interval(data         = np.array(eff),
                                minval       = interval[0]  ,
                                maxval       = interval[1]  ,
                                display_name = column.value ,
                                strictness   = strictness   ,
                                right_closed = True)

    return mask


def band_selector_and_check(dst         : pd.DataFrame,
                            boot_map    : ASectorMap,
                            norm_strat  : NormStrategy              = NormStrategy.max,
                            input_mask  : np.array                  = None            ,
                            range_Z     : Tuple[np.array, np.array] = (10, 1300)      ,
                            range_E     : Tuple[np.array, np.array] = (10.0e+3,14e+3) ,
                            nbins_z     : int                       = 50              ,
                            nbins_e     : int                       = 50              ,
                            nsigma_sel  : float                     = 3.5             ,
                            eff_interval: Tuple[float, float]       = [0,1]           ,
                            strictness : Strictness = Strictness.stop_proccess
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
    range_Z: Tuple[np.array, np.array]
        Range in Z-axis
    range_E: Tuple[np.array, np.array]
        Range in Energy-axis
    nbins_z: int
        Number of bins in Z-axis
    nbins_e: int
        Number of bins in energy-axis
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
    else: pass;

    emaps = apply_geo_correction(boot_map, norm_strat  = norm_strat)
    E0    = dst[input_mask].S2e.values * emaps(dst[input_mask].X.values,
                                               dst[input_mask].Y.values)

    sel_krband = np.zeros_like(input_mask)
    sel_krband[input_mask] = selection_in_band(dst[input_mask].Z,
                                               E0,
                                               range_z = range_Z,
                                               range_e = range_E,
                                               nbins_z = nbins_z,
                                               nbins_e = nbins_e,
                                               nsigma  = nsigma_sel)

    effsel   = dst[sel_krband].event.nunique()/dst[input_mask].event.nunique()

    check_if_values_in_interval(data         = np.array(effsel)  ,
                                minval       = eff_interval[0]   ,
                                maxval       = eff_interval[1]   ,
                                display_name = "Z-band selection",
                                strictness   = strictness        ,
                                right_closed = True)

    return sel_krband


def selection_in_band(z         : np.array,
                      e         : np.array,
                      range_z   : Tuple[float, float],
                      range_e   : Tuple[float, float],
                      nbins_z   : int     = 50,
                      nbins_e   : int     = 100,
                      nsigma    : float   = 3.5) ->np.array:
    """
    This function returns a mask for the selection of the events that are inside the Kr E vz Z

    Parameters
    ----------
    z: np.array
        axial (z) values
    e: np.array
        energy values
    range_z: Tuple[np.array, np.array]
        Range in Z-axis
    range_e: Tuple[np.array, np.array]
        Range in Energy-axis
    nbins_z: int
        Number of bins in Z-axis
    nbins_e: int
        Number of bins in energy-axis
    nsigma: float
        Number of sigmas to set the band width
    Returns
    ----------
        A  mask corresponding to the selection made.
    """

    # To be implemented

    return  [True] * len(z)


def get_par_name_from_fittype(fittype):

    par_name  = 'dedt'  if fittype == KrFitFunction.linear else 'lt'
    return par_name


def  create_df_kr_map(fittype : KrFitFunction,
                      bins    : Tuple[np.array, np.array],
                      counts  : np.array,
                      n_min   : int,
                      r_max   : float)->pd.DataFrame:
    '''
    This function creates the dataframe in which the map parameters are stored.

    Parameters
    ----------

    fittype : KrFitFunction
        Chosen fit function for map computation
    bins : Tuple[np.array, np.array]
        Tuple containing bins in both axis
    counts : np.array
        Number of events falling into each bin
    n_min : int
        Min number of events per bin required to perform fits
    r_max : float
        Radius defining the active area of the detector

    Returns
    -------

    kr_map : pd.DataFrame
        Kr map dataframe with all the info prior to the fits: bin label, events
        per bin, bin in/outside the active volume, bin position (X, Y, R), etc.
    '''


    par_name   = get_par_name_from_fittype(fittype)
    u_par_name = 'u' + par_name

    columns    = ['bin', 'counts', 'e0', 'ue0', par_name, u_par_name, 'covariance', 'res_std',
                  'pval', 'in_active', 'has_min_counts', 'fit_success', 'valid', 'R', 'X', 'Y']

    kr_map  = pd.DataFrame(columns = columns)

    n_xbins   = len(bins[0])-1
    n_ybins   = len(bins[1])-1
    b_center  = [shift_to_bin_centers(axis) for axis in bins]

    bin_index = range(n_xbins*n_ybins)
    geom_comb = itertools.product(b_center[1], b_center[0])
    r_values  = np.array([np.sqrt(x**2+y**2)for x, y in itertools.product(b_center[1], b_center[0])])

    kr_map['bin']            = bin_index
    kr_map['counts']         = counts
    kr_map['R']              = r_values
    kr_map[['Y', 'X']]       = pd.DataFrame(geom_comb)
    kr_map['in_active']      = kr_map['R']      <= r_max
    kr_map['has_min_counts'] = kr_map['counts'] >= n_min
    kr_map['fit_success']    = False
    kr_map['valid']          = False

    return kr_map




def get_number_of_bins(nevents : Optional[int] = None,
                       thr     : Optional[int] = 1e6,
                       n_bins  : Optional[int] = None)->int:
    """
    Computes the number of XY bins to be used in the creation
    of correction map regarding the number of selected events.
    Parameters
    ---------
    nevents: int (optional)
        Total number of provided events for the map computation.
    thr: int (optional)
        Event threshold to use 50x50 or 100x100 binning.
    n_bins: int (optional)
        The number of bins to use can be chosen a priori. If given,
        the returned number of bins is the one provided by the user.
        However, if no number of bins is given in advance, this will
        automatically select a value depending on the amount of events
        contained in the dst and the threshold.
    Returns
    ---------
    n_bins: int
        Number of bins in each direction (X,Y) (square map).
    """

    if    n_bins != None: return n_bins;
    elif  nevents < thr: return 50;
    else: return  100;


def get_XY_bins(n_bins   : int,
                XYrange  : Tuple[float, float]):


    """
    Returns the bins that will be used to make the map. It asumes both directions
    X, Y will have the same range and number of bins.

    Parameters
    ---------
    dst: int
        Number of bins to use per axis
    XYrange: Tuple[float, float]
        Limits (mm) of X and Y for the map computation

    Returns
    ---------
    bins: Tuple[np.array, np.array]
        Bins in each direction (X,Y) (square map).
    """

    bins = np.linspace(*XYrange, n_bins+1)

    return bins, bins


def get_binned_data(dst  : pd.DataFrame,
                    bins : Tuple[np.array, np.array]):

    '''
    This function distributes all the events in the DST into the selected
    bins. Given a certain binning, it computes which events fall into each
    square of the grid formed by the bins arrays. binned_statistic_2d returns
    counts (matrix here each matrix element is the number of events falling
    into that specific bin) and bin_indexes (a 2-D array labeling each event
    with the bin it belongs). Then counts is flattened into 1-D, bin_indexes
    is transformed into 1-D using the number of bins on each axis.

      Parameters
    --------------
    dst  : pd.DataFrame
         Krypton dataframe.
    bins : Tuple[np.array, np.array]
         Bins used to compute the map.

      Returns
    -------------
    counts     : np.array
         Total number of events falling into each bin
    bin_labels : np.array
         1D bin label for each event

      Further info:
    -----------------
    Why set expand_binnumbers to True (2D binning) if then we transform it to 1D?
    Because even though expand_binnumbers = False returns 1-D labels, it also adds
    two additional bins (per axis), taking into account out-of-range events which
    dont fall into the the binning passed to the binned_statistic_2d function. But
    since the dst is going to be previously selected and filtered with the desired
    binning, it's not convenient to use that. Maybe a visual example is more useful:

    2x2 binning (4 bins), natural index values shown both as 1D and 2D:

    || 0 | 1 ||          || (0, 0) | (0, 1) ||
    || 2 | 3 || (1D)  =  || (1, 0) | (1, 1) || (2D)

    Using expand_binnumbers = False, the 1D index values instead of (0, ..., 3)
    would be (0, ..., 15):

    || 0 | 1 | 2 | 3 ||
    || 4 | 5 | 6 | 7 ||  The bins that we "care about" (inner part) have indexes
    || 8 | 9 |10 |11 ||  5, 6, 9, 10 which I believe is not convenient at all.
    ||12 |13 |14 |15 ||  This creates (nx+2)*(ny+2) bins.
    '''

    n_xbins    = len(bins[0])-1
    n_ybins    = len(bins[1])-1

    counts, _, _, bin_indexes = stats.binned_statistic_2d(x=dst.X, y=dst.Y, values=None,
                                                          bins=bins, statistic='count',
                                                          expand_binnumbers=True)

    counts       = counts.flatten()
    bin_indexes -= 1
    bin_labels   = np.ravel_multi_index(bin_indexes, dims=(n_xbins, n_ybins),
                                        mode='clip', order = 'F')

    return counts, bin_labels


def lin_function(x, a, b):

    '''
    Linear function for fitting data.

    Parameters
    ----------
    x : np.array
        Independent variable.
    a : float
        Intercept parameter.
    b : float
        Slope parameter.

    Returns
    -------
    y : np.array
        Dependent variable values.
    '''

    y = a + b*x
    return y


def lin_seed(x, y):

    '''
    Estimate the seed for a linear fit.

    Parameters
    ----------
    x : np.array
        Independent variable.
    y : np.array
        Dependent variable.

    Returns
    -------
    seed : tuple
        Seed parameters (intercept, slope) for the linear fit.
    '''


    x0, x1 = x.min(), x.max()
    y0, y1 = y.min(), y.max()

    b = (y1 - y0) / (x1 - x0)
    a = y0 - b * x0

    seed = a, b

    return seed


def expo_function(x, const, mean):

    '''
    Exponential function for fitting decay data.

    Parameters
    ----------
    x : np.array
        Independent variable.
    const : float
        Constant scaling parameter.
    mean : float
        Mean parameter controlling the rate of decay.

    Returns
    -------
    y : np.array
        Dependent variable values (e.g., decayed energy).
    '''

    y = const * np.exp(-x / mean)
    return y


def expo_seed(x, y, eps=1e-12):

    '''
    Estimate the seed for an exponential fit.

    Parameters
    ----------
    x : np.array
        Independent variable.
    y : np.array
        Dependent variable.
    eps : float, optional
        Small value added to prevent division by zero, default is 1e-12.

    Returns
    -------
    seed : tuple
        Seed parameters (constant, mean) for the exponential fit.
    '''

    x, y = zip(*sorted(zip(x, y)))

    const = y[0]
    slope = (x[-1] - x[0]) / np.log(y[-1] / (y[0] + eps))

    seed = const, slope

    return seed


def get_fit_function_lt(fittype):

    '''
    Retrieve the fitting function and seed function based on the
    specified fittype.

    Parameters
    ----------
    fittype : KrFitFunction
        The type of fit function to retrieve (e.g., linear, exponential, log-linear).

    Returns
    -------
    fit_function  : function
        The fitting function corresponding to the specified fit type.
    seed_function : function
        The seed function corresponding to the specified fit type.
    '''

    if fittype is KrFitFunction.linear:
        # If the fit type is linear, return the linear fitting function and seed function
        return lin_function, lin_seed

    elif fittype is KrFitFunction.expo:
        # If the fit type is exponential, return the exponential fitting function and seed function
        return expo_function, expo_seed

    elif fittype is KrFitFunction.log_lin:
        # If the fit type is log-linear, return the linear fitting function and seed function
        return lin_function, lin_seed


def prepare_data(fittype, dst):

    '''
    Prepare the data for fitting based on the specified fit type.

    Parameters
    ----------
    fittype : KrFitFunction
        The type of fit function to prepare data for (e.g., linear, exponential, log-linear).
    dst : pd.DataFrame
        The DataFrame containing the data to be prepared for fitting.

    Returns
    -------
    x_data : np.array
        The independent variable data prepared for fitting.
    y_data : np.array
        The dependent variable data prepared for fitting.
    '''

    if fittype is KrFitFunction.linear:
        # If the fit type is linear, return DT (time differences) as x_data and S2e (energy) as y_data
        return dst.DT, dst.S2e

    elif fittype is KrFitFunction.expo:
        # If the fit type is exponential, return DT (time differences) as x_data and S2e (energy) as y_data
        return dst.DT, dst.S2e

    elif fittype is KrFitFunction.log_lin:
        # If the fit type is log-linear, return DT (time differences) as x_data and -log(S2e) as y_data
        return dst.DT, -np.log(dst.S2e)


def transform_parameters(fittype    : KrFitFunction,
                         fit_output : FitFunction):

    '''
    Transform the parameters obtained from the fitting output based on the specified fittype.

    Parameters
    ----------
    fittype : KrFitFunction
        The type of fit function used (e.g., linear, exponential, log-linear).
    fit_output : FitFunction
        Output from IC's fit containing the parameter values, errors, and covariance matrix.

    Returns
    -------
    par : list
        Transformed parameter values.
    err : list
        Transformed parameter errors.
    cov : float
        Transformed covariance value.
    '''

    par = fit_output.values
    err = fit_output.errors
    cov = fit_output.cov[0, 1]

    if fittype == KrFitFunction.log_lin:

        a, b = par
        u_a, u_b = err

        E0 = np.exp(-a)
        s_E0 = np.abs(E0 * u_a)

        lt = 1 / b
        s_lt = np.abs(lt**2 * u_b)

        cov = E0 * lt**2 * cov # Not sure about this

        par = [E0, lt]
        err = [s_E0, s_lt]

        return par, err, cov

    else:

        return par, err, cov


def calculate_residuals(dst     : pd.DataFrame,
                        fittype : KrFitFunction,
                        par     : list[float, float]):

    '''
    Calculate residuals and their standard deviation for the fitted data.

    Parameters
    ----------
    dst : pd.DataFrame
        DataFrame containing the data.
    fittype : KrFitFunction
        The type of fit function used (e.g., linear, exponential, log-linear).
    par : list
        Fitted parameters.

    Returns
    -------
    res : np.array
        Residuals.
    std : float
        Standard deviation of residuals.
    '''

    function = expo_function if fittype == KrFitFunction.log_lin else lin_function

    res = dst.S2e - function(dst.DT, *par)
    std = res.std()

    return res, std




