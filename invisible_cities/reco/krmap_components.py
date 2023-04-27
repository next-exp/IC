import numpy          as np
import pandas         as pd
import scipy.stats    as stats
import scipy.optimize as so

from typing                               import Tuple
from invisible_cities.core.core_functions import in_range, shift_to_bin_centers


class KrMap():

    ''' Class for the Kr Map container.

        x_bins  (1D-array of floats)  : bins in coordinate X
        y_bins  (1D-array of floats)  : bins in coordinate Y
        counts  (2D-array of ints)    : number of events in every X,Y bin of the map
        e0      (2D-array of floats)  : energy correction for every X,Y bin of the map
        ue0     (2D-array of floats)  : uncertainty for EO
        lt      (2D-array of floats)  : lifetime correction for every X,Y bin of the map
        ult     (2D-array of floats)  : uncertainty for lt
        cov     (2D-array of floats)  : non-diag covariance matrix element
        pval    (2D-array of floats)  : pvalue
        res_std (2D-array of floats)  : std for residuals
        valid   (2D-array of boolean) : success in the bin'''


    def __init__(self,
                 x_bins : np.array,
                 y_bins : np.array,
                 counts : np.array):

        shape        = (len(x_bins)-1, len(y_bins)-1)

        self.x_bins   = x_bins
        self.y_bins   = y_bins
        self.counts   = counts
        self.e0       = np.zeros(shape=shape, dtype=float)
        self.ue0      = np.zeros(shape=shape, dtype=float)
        self.lt       = np.zeros(shape=shape, dtype=float)
        self.ult      = np.zeros(shape=shape, dtype=float)
        self.cov      = np.zeros(shape=shape, dtype=float)
        self.pval     = np.zeros(shape=shape, dtype=float)
        self.res_std  = np.zeros(shape=shape, dtype=float)
        self.valid    = np.zeros(shape=shape, dtype= bool)



    def __str__(self):
        return '''{}(\nx_bins={.x_bins},\ny_bins={.y_bins},\ncounts={.counts},\ne0={.e0},\nue0={.ue0},\nlt={.lt},\nult={.ult},\ncov={.cov},\npval = {.pval},\nres = {.res_std})'''.format(self.__class__.__name__, self, self, self, self, self, self, self, self, self, self)

    __repr__ = __str__



def get_number_of_bins(dst    : pd.DataFrame,
                       thr    : int = 1e6,
                       n_bins : int = None)->int: #Similar to ICAROS
    """
    Computes the number of XY bins to be used in the creation
    of correction map regarding the number of selected events.
    Parameters
    ---------
    dst: pd.DataFrame
        File containing the events for the map computation.
    thr: int (optional)
        Threshold to use 50x50 or 100x100 maps (standard values).
    n_bins: int (optional)
        The number of events to use can be chosen a priori.
    Returns
    ---------
    n_bins: int
        Number of bins in each direction (X,Y) (square map).
    """

    if    n_bins != None: pass;
    elif  len(dst.index._data) < thr: n_bins = 50
    else: n_bins = 100;
    return n_bins


def get_XY_bins(n_bins   : int,
                XYrange  : Tuple[float, float] = (-490, 490)):

    # ATTENTION! The values for XYrange are temporary, ideally I'd suggest
    # to obtain them through the DataBase depending on the chosen detector
    # (the detector parameter 'next100', 'demopp', etc is present in every
    # argument list for the cities on IC). That way, the right geometry is
    # selected when running the city. In case one prefers to have a custom
    # range, it can be specified anyways.
    #
    # I was thinking of something like this:
    # from invisible_cities.database.load_db import DetectorGeo
    # XYrange = DetectorGeo('next100').values[0][0:2]
    #
    # I just left this as a first approach

    """
    Returns the bins that will be used to make the map.
    Parameters
    ---------
    dst: int
        Number of bins to use per axis
    XYrange: Tuple[float, float]
        Limits in X and Y for the map computation
    Returns
    ---------
    n_bins: int
        Number of bins in each direction (X,Y) (square map).
    """

    xbins = np.linspace(*XYrange, n_bins+1)
    ybins = xbins # I'm assuming it's always going to be the same for both
                  # directions, but in case we don't want to build it like
                  # this we can always define two different Xrange, Yrange
                  # and make one np.linspace(...) for each of them.

    return xbins, ybins


def get_binned_data(dst  : pd.DataFrame,
                    bins : Tuple[np.array, np.array]):

    '''This function distributes all the events in the DST into the selected
    bins, and updates the DST in order to include the X, Y bin index of its
    corresponding bin.

      Parameters
    --------------
    dst  : pd.DataFrame
         Krypton dataframe.
    bins : Tuple[np.array, np.array]
         Bins used to compute the map.

       Returns
    -------------
    counts : np.array
         Total number of events falling into each bin
    bin_centers : list
         List contaning np.arrays with the center of the bins


    '''

    counts, bin_edges, bin_index = stats.binned_statistic_dd((dst.X, dst.Y), dst.S2e, bins = bins,
                                                             statistic = 'count', expand_binnumbers = True)


    bin_centers  = [shift_to_bin_centers(axis) for axis in bin_edges] # Not necessary... maybe it's dropped in next version
    bin_index   -= 1
    # dst.assign(xbin_index=bin_index[0], ybin_index=bin_index[1]) pd.assign is not working for me... i also tried with
                                                                 # dst = dst.assign(**{xbinindex:bin_index[0], .....})
                                                                 # but i can't get the dst to update with the new cols
    dst['xbin_index'] = bin_index[0]
    dst['ybin_index'] = bin_index[1]

    return counts


def linear_function(DT, E0, LT):

    '''Given the DriftTime, and the geometrical (E0) and lifetime (LT)
    corrections, this function computes the energy.

    Parameters:
        DT  : np.array
            Drift Time of selected events.
        E0  : np.array
            E0 correction factor.
        LT  : np.array
            Lifetime correction factor.

    Returns:
        E   : np.array
            Energies
    '''

    dt0 = 680 # Middle point of chamber. FUTURE: taken from the database

    E   = E0 - LT * (DT - dt0)
    return E


def function_(fittype):

    if fittype == 'linear':
        return linear_function

    # Additional types to be included


def lifetime_fit_linear(DT : np.array,
                        E  : np.array):

    '''This function performs the linear lifetime fit. It returns the parameters, errors,
    non-diagonal element in cov matrix and a success variable of the fit.'''

    par     = np.nan  * np.ones(2)
    err     = np.nan  * np.ones(2)

    par, var, _, _, ier = so.curve_fit(linear_function, DT, E, full_output=True)

    err[0] = np.sqrt(var[0, 0])
    err[1] = np.sqrt(var[1, 1])
    cov    = var

    success = True if ier in [1, 2, 3, 4] else False # See scipy.optimize.curve_fit documentation

    return par, err, cov[0,1], success


def lifetime_fit(DT      : np.array,
                 E       : np.array,
                 fittype : str):

    ''' Performs the kind of unbined lifetime fit that fittype specifies. Rudimentary.

    Parameters:
        DT      : np.array
            Drift Time of selected events.
        E       : np.array
            Energies of selected events.
        fittype : str.
            Desired fit: can be either 'icaros' (linear fit w/ np.polyfit), 'linear' (linear fit w/ scipy.optimize) or 'exp' (expo fit w/ scipy.optimize).'''

    # if fittype == 'icaros':

    #     return lifetime_fit_icaros(DT, E)

    if fittype == 'linear':

        return lifetime_fit_linear(DT, E)


def calculate_residuals_in_bin(dst, par, fittype):

    '''This function computes the corrected energy based on the correction
    parameters coming from a previous fit, and then calculates the residuals,
    comparing with the measured value. The dst is updated so it contains these
    residuals. Ultimately, the function returns a single std value of the resi-
    duals in order to have an idea of their distribution.'''

    fit_function = function_(fittype=fittype)
    res = dst.S2e - fit_function(dst.DT, *par)
    series = pd.Series(res, index=dst.index)
    dst['residuals'] = series.values
    std = res.std()
    return std


def calculate_pval(residuals):

    pval = stats.shapiro(residuals)[1] if (len(residuals) > 10) else 0.

    return pval


def get_inner_core(bins_x, bins_y, r_max = None):

    r_fid      = r_max if r_max != None else max(max(abs(bins_x)), max(abs(bins_y))) # Either select it or take it from the binning given

    binsizes_x = bins_x[1:] - bins_x[:-1] # Compute bin sizes
    binsizes_y = bins_y[1:] - bins_y[:-1]
    bin_size   = min(max(binsizes_x), max(binsizes_y))

    centers    = [shift_to_bin_centers(axis) for axis in (bins_x, bins_y)] # Compute bin centers

    xx, yy     = np.meshgrid(*centers)
    r          = np.sqrt(xx**2 + yy**2) # Define r based on the bin grid

    mask       = in_range(r, 0, r_fid + bin_size/2) # Select the allowed r values:
                                                    # Those falling within the r_max. Since the binning doesn't have to be
                                                    # the same in X and Y, the additional bin_size/2 makes sure that you're
                                                    # choosing all the pixels that are partially into the chosen r_max (You
                                                    # can have a corner of a bin that falls into the r_max region, but maybe
                                                    # its center doesn't). I believe this way it would be included.

    return mask


def valid_bin_counter(correction_map, r_max = None):

    core_mask     = get_inner_core(correction_map.x_bins, correction_map.y_bins, r_max) # Select inner part (avoid corners)
    inner_count   = core_mask.sum() # Total number of inner bins
    successful    = np.where(core_mask, correction_map.valid,  0) # Succesful bins among the inner bins
    success_count = np.count_nonzero(successful) # Number of successful bins

    return success_count, inner_count




# ==================================================================================================================================================
# ========================================================= NOT USED FOR THE MAP CORE ==============================================================
# ==================================================================================================================================================


def apply_correction_factors(dst, kr_map, function):

    """
        Computes the corrected energy given a certain kr map and
        the corresponding function.
        Parameters
        ----------
        dst : pd.Dataframe
            Dataframe containing the data
        kr_map : Kr_Map object
            Map where the corrections are stored
        function : Callable
            Function to calculate the energy based on DT and correction factors

        Returns
        -------
        Ecorr : pd.Series
            Array containing the corrected energy
        """

    corr_factors_e0 = maps_coefficient_getter(kr_map, 'e0')(dst.X, dst.Y)
    corr_factors_lt = maps_coefficient_getter(kr_map, 'lt')(dst.X, dst.Y)

    Ecorr = function(dst.DT, corr_factors_e0, corr_factors_lt)

    return Ecorr
