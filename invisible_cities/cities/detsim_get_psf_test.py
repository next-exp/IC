import os
import numpy  as np
import tables as tb
import pandas as pd

from pytest import fixture

from invisible_cities.cities.detsim_get_psf import *

####################################
######### CREATE XY FUNC ###########
####################################
@fixture
def extent_and_function():
    xmin, xmax, dx = 12, 100, 3.
    ymin, ymax, dy =-50, 50, 13.

    xaxis = (xmin, xmax, dx)
    yaxis = (ymin, ymax, dy)

    func = lambda x, y: - x**2

    return xaxis, yaxis, func


def test_create_xy_function(extent_and_function):

    xaxis, yaxis, func = extent_and_function

    xmin, xmax, dx = xaxis
    ymin, ymax, dy = yaxis

    xbins = np.arange(xmin, xmax+dx, dx)
    ybins = np.arange(ymin, ymax+dy, dy)

    x, y = np.meshgrid(xbins, ybins)
    x, y = x.flatten(), y.flatten()
    f = func(x, y)

    #test create_function
    H, _ = np.histogramdd((x, y), weights=f, bins=(xbins, ybins))
    callable = create_xy_function(H, xaxis, yaxis)

    noedges = (x<xbins[-2]) & (y<ybins[-2])
    assert (f[noedges] == callable(x[noedges], y[noedges])).all()


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
    x = xmin + nx//f * dx
    y = ymin + np.arange(0, ny) * dy

    assert (psf(x, y) == xslice).all()


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

    assert (psffunc(xr, yr) == factor).all()
