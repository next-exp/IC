import copy
import warnings
import itertools

import numpy        as np
import pandas       as pd
import scipy.stats  as stats


from   typing               import Tuple, Optional
from .. types.symbols       import KrFitFunction
from .. evm.ic_containers   import FitFunction
from .. core.fit_functions  import polynom, expo
from ..core.core_functions  import in_range, shift_to_bin_centers
from ..core.fit_functions   import fit, get_chi2_and_pvalue


def lin_seed(x : np.array, y : np.array):
    '''
    Estimate the seed for a linear fit of the form y = a + bx.

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

    if x1 == x0: # If same x value, set slope to 0 and use the mean value of y as interceipt
        b = 0
        a = y.mean()
    else:
        b = (y1 - y0) / (x1 - x0)
        a = y0 - b * x0

    return a, b


def expo_seed(x : np.array, y : np.array):
    '''
    Estimate the seed for an exponential fit of the form y = y0*exp(-x/lt).

    Parameters
    ----------
    x : np.array
        Independent variable.
    y : np.array
        Dependent variable.

    Returns
    -------
    seed : tuple
        Seed parameters (y0, lt) for the exponential fit.
    '''
    x, y = zip(*sorted(zip(x, y)))
    y0   = y[0]

    if y0 <= 0 or y[-1] <= 0:
        raise ValueError("y data must be > 0")

    lt = -x[-1] / np.log(y[-1] / y0)

    return y0, lt


def select_fit_variables(fittype : KrFitFunction, dst : pd.DataFrame):
    '''
    Select the data for fitting based on the specified fit type.

    Parameters
    ----------
    fittype : KrFitFunction
        The type of fit function to prepare data for (e.g., linear, exponential, log-linear).
    dst : pd.DataFrame
        The DataFrame containing the data to be prepared for fitting.

    Returns
    -------
    x_data : pd.Series
        The independent variable data prepared for fitting.
    y_data : pd.Series
        The dependent variable data prepared for fitting.
    '''
    if   fittype is KrFitFunction.linear : return dst.DT,         dst.S2e
    elif fittype is KrFitFunction.expo   : return dst.DT,         dst.S2e
    elif fittype is KrFitFunction.log_lin: return dst.DT, -np.log(dst.S2e)


def get_function_and_seed_lt(fittype : KrFitFunction):
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
    linear_function  = lambda x, y0, slope: polynom(x, y0, slope)
    expo_function    = lambda x, e0, lt:    expo   (x, e0, -lt)

    if   fittype is KrFitFunction.linear:  return linear_function,  lin_seed
    elif fittype is KrFitFunction.log_lin: return linear_function,  lin_seed
    elif fittype is KrFitFunction.expo:    return   expo_function, expo_seed


def transform_parameters(fit_output : FitFunction):
    '''
    Transform the parameters obtained from the fitting output into EO and LT.
    When using log_lin fit, we need to convert the intermediate variables into
    the actual physical magnitudes involved in the process.

    Parameters
    ----------
    fit_output : FitFunction
        Output from IC's fit containing the parameter values, errors, and
        covariance matrix.

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

    a, b     = par
    u_a, u_b = err

    E0   = np.exp(-a)
    s_E0 = np.abs(E0 * u_a)
    lt   = 1 / b
    s_lt = np.abs(lt**2 * u_b)
    cov  = E0 * lt**2 * cov # Not sure about this

    par  = [  E0,   lt]
    err  = [s_E0, s_lt]

    return par, err, cov


def create_df_kr_map(bins   : Tuple[np.array, np.array],
                     counts : np.array,
                     n_min  : int,
                     r_max  : float)->pd.DataFrame:
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
        Kr map dataframe with all the info prior to the fits: bin label,
        events per bin, bin in/outside the active volume, bin position
        (X, Y, R), etc.
    '''

    columns    = ['bin', 'counts', 'e0', 'ue0', 'lt', 'ult', 'covariance', 'res_std', 'chi2',
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
                       n_bins  : Optional[np.array] = None)->np.array:
    '''
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
    '''
    if    n_bins is not None: return n_bins;
    elif  nevents < thr: return np.array([50, 50]);
    else: return  np.array([100, 100]);


def get_XY_bins(n_bins   : np.array,
                XYrange  : Tuple[float, float]):
    '''
    Returns the bins that will be used to make the map.

    Parameters
    ---------
    b_nins: np.array
        array of len = 2 containing the number of bins in X and Y
    XYrange: Tuple[float, float]
        Limits (mm) of X and Y for the map computation

    Returns
    ---------
    bins: Tuple[np.array, np.array]
        Bins in each direction (X,Y) (square map).
    '''
    bins_x = np.linspace(*XYrange, n_bins[0]+1)
    bins_y = np.linspace(*XYrange, n_bins[1]+1)
    return bins_x, bins_y


def get_bin_counts_and_event_bin_id(dst  : pd.DataFrame,
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


def calculate_residuals(x          : pd.Series,
                        y          : pd.Series,
                        fit_output : FitFunction):

    '''
    Calculate residuals and their standard deviation for the fitted data.

    Parameters
    ----------
    x : pd.Series
        Independent variable
    y : pd.Series
        Dependent variable
    fit_output : FitFunction
        Container for IC's fit function result

    Returns
    -------
    res : np.array
        Residuals.
    std : float
        Standard deviation of residuals.
    '''

    function = fit_output.fn

    res = y - function(x)
    std = res.std()

    return res, std


def calculate_pval(residuals : np.array):

    '''
    Calculate the p-value for the Shapiro-Wilk normality test of residuals.

    Parameters
    ----------
    residuals : np.array
        Residuals from the fitted model.

    Returns
    -------
    pval : float
        p-value for the Shapiro-Wilk normality test.
    '''

    pval = stats.shapiro(residuals)[1] if (len(residuals) > 10) else 0.

    return pval


def valid_bin_counter(map_df             : pd.DataFrame,
                      validity_parameter : Optional[float] = 0.9):
    '''
    Count the number of valid bins in the map DataFrame and issue a warning
    if the validity threshold is not met.

    Parameters
    ----------
    map_df : pd.DataFrame
        DataFrame containing map data.
    validity_parameter : float
        Threshold for the ratio of valid bins (default set to 0.9).

    Returns
    -------
    valid_per : float
        Percentage of valid bins.
    '''

    inside = len(map_df[map_df['in_active'] == True])
    valid  = len(map_df[map_df['valid']     == True])

    valid_per = valid / inside * 100

    if valid_per <= validity_parameter:
        # If the percentage of valid bins is below the threshold, issue a warning
        warnings.warn(f"{inside-valid} inner bins are not valid. {valid_per:.1f}% are successful.", UserWarning)
    else:
        # If the percentage of valid bins meets the threshold, print a message
        print(f"{inside-valid} inner bins are not valid. {valid_per:.1f}% are successful.")

    return valid_per


def fit_and_fill_map(map_bin : pd.DataFrame,
                     dst     : pd.DataFrame,
                     fittype : KrFitFunction):
    '''
    This function is the core of the map computation. It's the one in charge of looping
    over all the bins, performing the calculations of the lifetime fits and filling the
    krypton map dataframe with all the obtained parameters.

    Basically, it's applied to the dataframe, and bin by bin, it checks if the conditions
    for the map computation are met. If it's the case, it will call the corresponding fit
    function specified in the fittype parameter, calculate all the different map values
    and fill the dataframe with them.

      Parameters
    --------------
    map_bin : pd.DataFrame
         KrMap dataframe
    dst     : pd.DataFrame
         kdst dataframe.
    fittype : str
         Type of fit to perform.

       Returns
    -------------
    kr_map : pd.DataFrame
         DataFrame containing map parameters.
    '''
    if not map_bin['in_active'] or not map_bin['has_min_counts']: return map_bin

    k        = map_bin.bin

    dst_bin  = dst.query(f'bin_index == {k}')

    fit_func, seed = get_function_and_seed_lt(fittype = fittype)
    x, y           = select_fit_variables    (fittype = fittype,
                                              dst     = dst_bin)

    fit_output, _, _, ier = fit(func        = fit_func,
                                x           = x,
                                y           = y,
                                seed        = seed(x, y),
                                full_output = True)

    par, err, cov = transform_parameters(fittype    = fittype,
                                         fit_output = fit_output)

    res, std = calculate_residuals(x, y, fit_output) # Still considering this
    chi2, _  = get_chi2_and_pvalue(y, fit_output.fn(x), len(x)-len(par), std)
    pval     = calculate_pval(res)

    map_bin['e0']          = par[0]
    map_bin['ue0']         = err[0]
    map_bin['lt']          = par[1]
    map_bin['ult']         = err[1]
    map_bin['covariance']  = cov
    map_bin['res_std']     = std
    map_bin['chi2']        = chi2
    map_bin['pval']        = pval
    map_bin['fit_success'] = True if ier in [1, 2, 3, 4] else False
    map_bin['valid']       = map_bin['fit_success'] & map_bin['has_min_counts'] & map_bin['in_active']

    return map_bin


def find_outliers(maps    : pd.DataFrame,
                  x2range : Tuple[float, float] = (0, 2)):
    '''
    For a given maps and deserved range, it returns a mask where values are
    within the interval.

    Parameters
    ---------
    maps: pd.DataFrame
        Map to check the outliers
    x2range : Tuple[float, float]
        Range for chi2

    Returns
    ---------
    mask: pd.Series
        Mask.
    '''
    mask = in_range(maps.chi2, *x2range)
    return mask


def regularize_map(maps    : pd.DataFrame,
                   x2range : Tuple[float, float]):
    '''
    Given a certain chi2 range, this function checks which bins are outside that
    range and substitutes the (probably wrong) values of the map (e0, lt, etc) for
    the mean value of the whole map.

    Parameters
    ---------
    maps: pd.DataFrame
        Map to check the outliers
    x2range : Tuple[float, float]
        Range for chi2

    Returns
    ---------
    new_map: pd.DataFrame
        Regularized map.'''

    new_map   = copy.deepcopy(maps)

    outliers  = maps.in_active
    outliers &= np.logical_not(find_outliers(new_map, x2range))

    new_map['e0'] [outliers] = np.nanmean(maps['e0'])
    new_map['lt'] [outliers] = np.nanmean(maps['lt'])
    new_map['ue0'][outliers] = np.nanmean(maps['ue0'])
    new_map['ult'][outliers] = np.nanmean(maps['ult'])

    return new_map
