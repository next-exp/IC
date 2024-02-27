import numpy        as np
import pandas       as pd
import scipy.stats  as stats

from   typing       import Tuple, Optional

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


<<<<<<< HEAD
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
=======
def band_selector_and_check(dst       : pd.DataFrame,
                            boot_map   : ASectorMap,
                            norm_strat : NormStrategy              = NormStrategy.max,
                            input_mask : np.array                  = None,
                            range_Z    : Tuple[np.array, np.array] = (10, 550),
                            range_E    : Tuple[np.array, np.array] = (10.0e+3,14e+3),
                            nbins_z    : int                       = 50,
                            nbins_e    : int                       = 50,
                            nsigma_sel : float                     = 3.5,
                            eff_min   : float                      = 0.4,
                            eff_max   : float                      = 0.6
                           )->np.array:
>>>>>>> 8e1bd3cc (Add band_selector_and_check function)
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
                                                          bins=bins, statistic = 'count',
                                                          expand_binnumbers = True)

    counts       = counts.flatten()
    bin_indexes -= 1
    bin_labels   = np.ravel_multi_index(bin_indexes, dims=(n_xbins, n_ybins),
                                        mode='clip', order = 'F')

    return counts, bin_labels
