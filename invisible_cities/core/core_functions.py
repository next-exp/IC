"""
Core functions
This module includes utility functions.
"""
import numpy as np
import pandas as pd
import math

def lrange(*args):
    """Create a list specified as a range."""
    return list(range(*args))


def trange(*args):
    """Create a tuple specified as a range."""
    return tuple(range(*args))


def loc_elem_1d(np1d, elem):
    """Given a 1d numpy array, return the location of element elem."""
    return np.where(np1d==elem)[0][0]


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


def rebin_array(arr, stride):
    """Rebins an array according to some stride.

    Parameters
    ----------
    arr : np.ndarray
        Array of numbers
    stride : int
        Integration step.

    Returns
    -------
    rebinned : np.ndarray
        Rebinned array
    """
    #n = int(math.ceil(len(t) / float(stride)))
    lenb = int(len(arr) / int(stride))
    rebinned = np.empty(lenb)
    for i in range(lenb):
        low = i * stride
        upp = low + stride
        rebinned[i] = np.sum(arr[low:upp])
    return rebinned


def define_window(wf, window_size):
    """Define a window based on a peak. Takes max plus/minus *window_size*."""
    peak = np.argmax(abs(wf - np.mean(wf)))
    return max(0, peak - window_size), min(len(wf), peak + window_size)
