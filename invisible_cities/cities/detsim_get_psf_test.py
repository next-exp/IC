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





# ####################################
# ############# KRMAP PSF ############
# ####################################
# def test_xyaxis_get_psf_from_krmap(correction_map_filename):
#
#     psf = get_psf_from_krmap(correction_map_filename, factor=1.)
#
#     maps = read_maps(correction_map_filename)
#     xmin, xmax, ymin, ymax, nx, ny, _ = maps.mapinfo.values
#     dx   = (xmax - xmin)/ float(nx)
#     dy   = (ymax - ymin)/ float(ny)
#     e0map = np.nan_to_num(maps.e0.values)
#
#     #select a random slice in, for example, x
#     f = np.random.randint(nx)
#     xslice = e0map[nx//f, :]
#     x = np.full(len(xslice) ,xmin + nx//f * dx)
#     y = ymin + np.arange(0, ny) * dy
#     z = np.zeros(len(x))
#
#     assert np.all(psf(x, y, z) == xslice)
#
#
# ####################################
# ############# SIPM PSF #############
# ####################################
# @fixture(params=["PSF_SiPM_dst_sum_collapsed.h5"])
# def sipmpsf_filename(request, ICDATADIR):
#     return os.path.join(ICDATADIR, request.param)
#
#
# def test_get_sipm_psf_from_file(sipmpsf_filename):
#
#     psffunc = get_sipm_psf_from_file(sipmpsf_filename)
#
#     with tb.open_file(sipmpsf_filename) as h5file:
#         psf = h5file.root.PSF.PSFs.read()
#     sel = (psf["z"] == np.min(psf["z"]))
#     psf = psf[sel]
#     xr, yr, factor = psf["xr"], psf["yr"], psf["factor"]
#
#     assert np.all(psffunc(xr, yr, np.zeros(len(xr))) == factor)
