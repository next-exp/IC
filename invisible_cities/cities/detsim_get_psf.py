import numpy  as np
import tables as tb
import pandas as pd

from typing import Tuple, Callable

from invisible_cities.reco.corrections_new import read_maps

##################################
############# PSF ################
##################################
def _psf(dx, dy, dz, factor = 1.):
    """ generic analytic PSF function
    """
    return factor * np.abs(dz) / (2 * np.pi) / (dx**2 + dy**2 + dz**2)**1.5


def create_xy_function(array : np.array,
                       xaxis : Tuple[float],
                       yaxis : Tuple[float])->Callable:
    xmin, xmax, dx = xaxis
    ymin, ymax, dy = yaxis
    def function(x, y, z=0):
        x = np.clip(x, xmin, xmax-0.1*dx)
        y = np.clip(y, ymin, ymax-0.1*dx)
        ix = ((x-xmin)//dx).astype(int)
        iy = ((y-ymin)//dy).astype(int)
        return array[ix, iy]
    return function


def get_psf_from_krmap(filename : str,
                       factor   : float = 1.)->Callable:
    """ reads KrMap and generate a psf function with the E0-map
    """
    maps = read_maps(filename)
    xmin, xmax, ymin, ymax, nx, ny, _ = maps.mapinfo.values
    dx   = (xmax - xmin)/ float(nx)
    dy   = (ymax - ymin)/ float(ny)
    e0map  = factor * np.nan_to_num(np.array(maps.e0), 0.)

    return create_xy_function(e0map, (xmin, xmax, dx), (ymin, ymax, dy))


def get_sipm_psf_from_file(filename : str)->Callable:
    with tb.open_file(filename) as h5file:
        psf = h5file.root.PSF.PSFs.read()

    #select psf z
    sel = (psf["z"] == np.min(psf["z"]))
    psf = psf[sel]
    xr, yr, factor = psf["xr"], psf["yr"], psf["factor"]

    #create binning
    xbins, ybins = np.unique(xr), np.unique(yr)
    dx = (xbins[-1] - xbins[0])/(len(xbins)-1)
    dy = (ybins[-1] - ybins[0])/(len(ybins)-1)
    xbins = np.append(xbins-dx/2., xbins[-1]+dx/2.)
    ybins = np.append(ybins-dy/2., ybins[-1]+dy/2.)

    #histogram
    psf, _ = np.histogramdd((xr, yr), weights=factor, bins=(xbins, ybins))
    xaxis = (np.min(xbins), np.max(xbins), dx)
    yaxis = (np.min(ybins), np.max(ybins), dy)

    return create_xy_function(psf, xaxis, yaxis)
