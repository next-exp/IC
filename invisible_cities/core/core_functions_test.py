from __future__ import absolute_import

from . import core_functions as core
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def test_simple():
    # First, consider a simple test function
    assert core.lrange(10) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert core.trange(10) == (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    assert core.loc_elem_1d(np.array(core.lrange(10)), 5) == 5
    assert core.dict_map(lambda x: x**2,
    {'a':1,'b':2,'c':3,'d':4}) == {'a': 1,
                                   'b': 4, 'c': 9, 'd': 16}

    d = {'q' : [-1, +1, -1],
         'mass' : [0.511, 105., 1776.],
         'spin' :[0.5, 0.5, 0.5]}

    leptons = pd.DataFrame(d,index=['e-', 'mu+', 'tau-'])
    l2 = core.df_map(lambda x: x*1000, leptons, 'mass')
    assert l2.mass.values[0] == 511

    core.dict_filter(lambda x: x>5,
      {'a':1,'b':20,'c':3,'d':40}) == {'b': 20, 'd': 40}

    core.farray_from_string('1 10 100')[2] == 100.

    core.rebin_array(core.lrange(100), 5)[0] == 10

    mu, sigma = 100, 0.2 # mean and standard deviation
    sgn = np.random.normal(mu, sigma, 10000)
    n, bins, patches = plt.hist(sgn, 50)
    n0, n1 = core.define_window(n, window_size=10)
    peak = core.loc_elem_1d(n, np.max(n))
    assert n0 == peak - 10
    assert n1 == peak + 10
