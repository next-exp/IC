import os
import numpy  as np
import tables as tb
import pandas as pd

from pytest import fixture

from invisible_cities.reco.corrections_new import read_maps

from invisible_cities.cities.detsim_get_psf import create_xyz_function
from invisible_cities.cities.detsim_get_psf import get_psf_from_krmap
from invisible_cities.cities.detsim_get_psf import get_sipm_psf_from_file

####################################
######### CREATE XY FUNC ###########
####################################
@fixture
def extent_and_function():
    xmin, xmax, dx = 12, 100, 3.
    ymin, ymax, dy = 10, 20, 1.
    zmin, zmax, dz =-50, 50, 13.

    xbins = np.arange(xmin, xmax+dz, dx)
    ybins = np.arange(ymin, ymax+dz, dy)
    zbins = np.arange(zmin, zmax+dz, dz)
    bins = [xbins, ybins, zbins]

    func = lambda x, y, z: -x**2 + 5*y**3 - z

    return bins, func


def test_create_xyz_function(extent_and_function):

    bins, func = extent_and_function

    x, y, z = np.meshgrid(bins[0][:-1], bins[1][:-1], bins[2][:-1]) #remove bin extreme
    x, y, z = x.flatten(), y.flatten(), z.flatten()
    f = func(x, y, z)

    #test create_function
    H, _ = np.histogramdd((x, y, z), weights=f, bins=bins)
    fxyz = create_xyz_function(H, bins)

    assert np.all(f == fxyz(x, y, z))


####################################
############# KRMAP PSF ############
####################################
def test_xyaxis_get_psf_from_krmap(correction_map_filename):

    psf = get_psf_from_krmap(correction_map_filename, factor=1.)

    maps = read_maps(correction_map_filename)
    xmin, xmax, ymin, ymax, nx, ny, _ = maps.mapinfo.values
    dx   = (xmax - xmin)/ float(nx)
    dy   = (ymax - ymin)/ float(ny)
    e0map = np.nan_to_num(maps.e0.values)

    #select a random slice in, for example, x
    f = np.random.randint(nx)
    xslice = e0map[nx//f, :]
    x = np.full(len(xslice) ,xmin + nx//f * dx)
    y = ymin + np.arange(0, ny) * dy
    z = np.zeros(len(x))

    assert np.all(psf(x, y, z) == xslice)


####################################
############# SIPM PSF #############
####################################
@fixture(params=["PSF_SiPM_dst_sum_collapsed.h5"])
def sipmpsf_filename(request, ICDATADIR):
    return os.path.join(ICDATADIR, request.param)


def test_get_sipm_psf_from_file(sipmpsf_filename):

    psffunc = get_sipm_psf_from_file(sipmpsf_filename)

    with tb.open_file(sipmpsf_filename) as h5file:
        psf = h5file.root.PSF.PSFs.read()
    sel = (psf["z"] == np.min(psf["z"]))
    psf = psf[sel]
    xr, yr, factor = psf["xr"], psf["yr"], psf["factor"]

    assert np.all(psffunc(xr, yr, np.zeros(len(xr))) == factor)
