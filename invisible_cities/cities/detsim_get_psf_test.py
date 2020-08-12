import os
import numpy  as np
import tables as tb
import pandas as pd

from pytest import fixture
from pytest import raises

from invisible_cities.cities.detsim_get_psf import create_xyz_function
from invisible_cities.cities.detsim_get_psf import create_xy_function
from invisible_cities.cities.detsim_get_psf import binedges_from_bincenters


def test_create_xy_function():

    # create discrete function
    H = np.array([[1, 2, 3],
                  [4, 5, 6]])

    xbins = np.array([0, 1, 2])
    ybins = np.array([0, 1, 2, 3])
    bins = [xbins, ybins]

    # test for some values
    x = np.array([1.5, 0.7, 0.1, -1, 1 ])
    y = np.array([2.9, 1.4, 2.2, 3., 20])
    expected_values = np.array([6., 2., 3., 0., 0.])

    func = create_xy_function(H, bins)
    obtained_values = func(x, y)

    assert np.all(obtained_values == expected_values)


def test_create_xy_function_Exceptions():

    # create discrete function
    H = np.array([[1, 2, 3],
                  [4, 5, 6]])

    xbins = np.array([0, 1, 2])
    ybins = np.array([0, 1, 2, 3])
    bins = [xbins, ybins]

    func = create_xy_function(H, bins)

    # test for some values
    x = np.array([1.5, 0.7, 0.1, -1, 1 ])
    y = np.array([2.9, 1.4, 2.2, 3.])

    with raises(Exception) as E:
        assert func(x, y)
    assert str(E.value) == "x and y must have same size"

    xbins = np.array([0, 1, 2, 3])
    bins = [xbins, ybins]
    with raises(Exception) as E:
        assert create_xy_function(H, bins)
    assert str(E.value) == "bins and array shapes not consistent"


def test_create_xyz_function():

    # create discrete function
    H = np.array([[[1, 1], [2, 2], [3, 3]],
                  [[4, 4], [5, 5], [6, 6]]])

    xbins = np.array([0, 1, 2])
    ybins = np.array([0, 1, 2, 3])
    zbins = np.array([10, 20, 30])
    bins = [xbins, ybins, zbins]

    # test for some values
    x = np.array([1.5, 0.7, 0.1, -1, 1 ])
    y = np.array([2.9, 1.4, 2.2, 3., 20])
    z = np.array([10 ,  20, -2 , 15, 5 ])
    expected_values = np.array([6., 2., 0., 0., 0.])

    func = create_xyz_function(H, bins)
    obtained_values = func(x, y, z)

    assert np.all(obtained_values == expected_values)


def test_binedges_from_bincenters():

    centers = np.array([0, 1, 2, 3])
    assert np.all(binedges_from_bincenters(centers) == np.array([0, 0.5, 1.5, 2.5, 3.]))

    # Exceptions
    centers = np.array([])
    with raises(Exception) as E:
        assert binedges_from_bincenters(centers)
    assert str(E.value) == "Inconsistent array shape"

    centers = np.array([[0, 1], [2, 3]])
    with raises(Exception) as E:
        assert binedges_from_bincenters(centers)
    assert str(E.value) == "Inconsistent array shape"

    centers = np.array([10, 9, 8])
    with raises(Exception) as E:
        assert binedges_from_bincenters(centers)
    assert str(E.value) == "Unordered bin centers"
