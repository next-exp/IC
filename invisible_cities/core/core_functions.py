"""
Core functions
This module includes utility functions.
"""
import numpy as np
import time


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


def in_range(data, minval=-np.inf, maxval=np.inf):
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

    Returns
    -------
    selection : np.ndarray
        Boolean array with the same dimension as the input. Contains True
        for those values of data in the input range and False for the others.
    """
    return (minval <= data) & (data < maxval)


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


def dict_filter(cond, dic):
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
