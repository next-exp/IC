import numpy          as np
import pandas         as pd
import scipy.stats    as stats
import scipy.optimize as so

from typing                               import Tuple
from invisible_cities.core.core_functions import shift_to_bin_centers


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

