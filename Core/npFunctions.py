"""
numpy functions
JJGC December 2016

"""
from __future__ import print_function

import math
import numpy as np
import pandas as pd


def np_loc1d(np1d, elem):
    """
    Given a 1d numpy array, return the location of element elem
    """
    return np.where(np1d==elem)[0][0]
