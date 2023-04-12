import pandas         as pd


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