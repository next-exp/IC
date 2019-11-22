"""
Core functions
This module includes utility functions.
"""
import time
import enum

import numpy as np

from typing import Sequence

from .. types.ic_types import AutoNameEnumBase


class NormMode(enum.Enum):
    first   = 0
    second  = 1
    sumof   = 3
    mean    = 4


class Interval(AutoNameEnumBase):
    open   = enum.auto()
    closed = enum.auto()


def merge_two_dicts(a,b):
    return {**a, **b}


def timefunc(f):
    """
    Decorator for function timing.
    """
    def time_f(*args, **kwargs):
        t0 = time.time()
        output = f(*args, **kwargs)
        print("Time spent in {}: {} s".format(f.__name__,
                                              time.time() - t0))
        return output
    return time_f


def flat(nested_list):
    while hasattr(nested_list[0], "__iter__"):
        nested_list = [item for inner_list in nested_list for item in inner_list]
    return np.array(nested_list)


def lrange(*args):
    """Create a list specified as a range."""
    return list(range(*args))


def trange(*args):
    """Create a tuple specified as a range."""
    return tuple(range(*args))


def relative_difference(x, y, *, norm_mode=NormMode.first):
    if   norm_mode is NormMode.first : return     (x - y) /  x
    elif norm_mode is NormMode.second: return     (x - y) /      y
    elif norm_mode is NormMode.sumof : return     (x - y) / (x + y)
    elif norm_mode is NormMode.mean  : return 2 * (x - y) / (x + y)
    else:
        raise TypeError(f"Unrecognized relative difference option: {norm_mode}")


def in_range(data, minval=-np.inf, maxval=np.inf, left=Interval.closed, right=Interval.open):
    """
    Find values in range [minval, maxval).

    Parameters
    ---------
    data : np.ndarray
        Data set of arbitrary dimension.
    minval : int or float, optional
        Range minimum. Defaults to -inf.
    maxval : int or float, optional
        Range maximum. Defaults to +inf.
    left   : {"open", "close"}
        Use close (default) or open  interval on the lower bound
    right  : {"open", "close"}
        Use open  (default) or close interval on the upper bound

    Returns
    -------
    selection : np.ndarray
        Boolean array with the same dimension as the input. Contains True
        for those values of data in the input range and False for the others.
    """
    if left not in Interval or right not in Interval:
        raise ValueError("left and right must be an instance of Interval")
    lower_bound = data >= minval if left  is Interval.closed else data > minval
    upper_bound = data <= maxval if right is Interval.closed else data < maxval
    return lower_bound & upper_bound


def weighted_mean_and_var(data       : Sequence,
                          weights    : Sequence,
                          unbiased   : bool = False,
                          frequentist: bool = True,
                          axis              = None) -> (float, float):
    """
    Compute mean value and variance of a dataset with given
    weights.

    Parameters
    ----------
    data: Sequence
        Dataset to which mean and variance are measured.
    weights: Sequence
        Weight for each data point.
    unbiased: Boolean, optional
        Whether to use the unbiased estimator of the variance.
    frequentist: Boolean, optional
        This parameter is ignored if `unbiased` is False.
        If True (default) the weights are interpreted as
        frequencies. Otherwise, they are interpreted as
        a measurement of the reliability of the data point.
    axis: None, int or tuple of ints
        Axis or axes along which to average a. The default, axis=None,
        will average over all of the elements of the input array.
        If axis is negative it counts from the last to the first axis.

    Returns
    -------
    ave: float
        Weighted average of data.
    var: float
        Weighted average of data.
    """
    data      = np.array(data   )
    weights   = np.array(weights)

    ave, wsum = np.average( data          , weights=weights, axis=axis, returned=True)
    var       = np.average((data - ave)**2, weights=weights, axis=axis)

    if unbiased:
        if frequentist:
            var *= wsum / (wsum - 1)
        else:
            wsum2 = np.sum(weights**2)
            var *= 1 / (1 - wsum2/wsum**2)
    return ave, var


def weighted_mean_and_std(*args, **kwargs):
    """
    Same as `weighted_mean_and_var`, but returns the
    standard deviation instead of the variance.
    """
    ave, var = weighted_mean_and_var(*args, **kwargs)
    return ave, np.sqrt(var)


def loc_elem_1d(np1d, elem):
    """Given a 1d numpy array, return the location of element elem."""
    return np.where(np1d==elem)[0][0]


def np_range(start, end, stride):
    """Syntactic sugar for np.arange() (included for consistency with
    np_reverse_range).
    """
    return np.arange(start, end, stride)


def np_reverse_range(start, end, stride):
    """Reverse range."""
    return np.arange(start, end, stride)[::-1]


def np_constant(dim, value):
    """Returns a np array of dimension dim with all elements == value."""
    return np.ones(dim) * value


def to_row_vector(x):
    """
    Takes a np.array of shape (n,) and returns a row vector
    of shape (1, n).
    """
    return x[np.newaxis, :]


def to_col_vector(x):
    """
    Takes a np.array of shape (n,) and returns a column
    vector of shape (n, 1).
    """
    return x[:, np.newaxis]


def dict_map(func, dic):
    """Apply map to dictionary values maintaining correspondence.

    Parameters
    ----------
    func : callable
        Function to be applied on values.
    dic : dictionary
        Dictionary on which func is applied.

    Returns
    -------
    mapdic : dictionary
        Contains key: func(value) for each key, value pair in dic.
    """
    return {key: func(val) for key, val in dic.items()}


def df_map(func, df, field):
    """Apply map to some data frame field.

    Parameters
    ----------
    func : callable
        Function to be applied on field values.
    df : pd.DataFrame
        DataFrame containing field.
    field : string
        Label of the DataFrame column.

    Returns
    -------
    mapdf : pd.DataFrame
        Copy of df with the column *field* modified to contain the output of
        func.
    """
    out = df.copy()
    out[field] = list(map(func, out[field]))
    return out


def dict_filter_by_value(cond, dic):
    """Apply filter to dictionary values maintaining correspondence.

    Parameters
    ----------
    cond : callable
        Condition to be satisfied.
    dic : dictionary
        Dictionary on which cond is applied.

    Returns
    -------
    filterdic : dictionary
        Contains the key, value pairs in dic satisfying cond.
    """
    return {key: val for key, val in dic.items() if cond(val)}


def dict_filter_by_key(cond, dic):
    """Apply filter to dictionary keys maintaining correspondence.
    Parameters
    ----------
    cond : callable
        Condition to be satisfied.
    dic : dictionary
        Dictionary on which cond is applied.
    Returns
    -------
    filterdic : dictionary
        Contains the key, value pairs in dic satisfying cond.
    """
    return {key: val for key, val in dic.items() if cond(key)}


def farray_from_string(sfl):
    """Convert a string of space-separated floats to a np array.

    Parameters
    ----------
    sfl : string
        Input of space-separated floats.

    Returns
    -------
    arr : np.ndarray
        Contains the casted floats.
    """
    return np.array(list(map(float, sfl.split())))


def _rebin_array(arr, stride, met=np.sum, remainder=False):
    """
    rebin arr by a factor stride, using method (ex: np.sum or np.mean), keep the remainder in the
    last bin or not
    """
    lenb = int(len(arr) / int(stride))
    if remainder and len(arr) % stride != 0:
        rebinned     = np.empty(lenb + 1)
        rebinned[-1] = met(arr[lenb*stride:])
    else:
        rebinned = np.empty(lenb)
    for i in range(lenb):
        s = i * stride
        f = s + stride
        rebinned[i] = met(arr[s:f])
    return rebinned


def define_window(wf, window_size):
    """Define a window based on a peak. Takes max plus/minus *window_size*."""
    peak = np.argmax(abs(wf - np.mean(wf)))
    return max(0, peak - window_size), min(len(wf), peak + window_size)


def mean_handle_empty(array):
    return np.mean(array) if len(array) else np.nan


def  std_handle_empty(array):
    return np.std (array) if len(array) else np.nan


def shift_to_bin_centers(x):
    """
    Return bin centers, given bin lower edges.
    """
    return x[:-1] + np.diff(x) * 0.5
