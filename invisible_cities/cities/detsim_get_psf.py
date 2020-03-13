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
        y = np.clip(y, ymin, ymax-0.1*dy)
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
    xcenters, ycenters = np.unique(xr), np.unique(yr)
    xbins = binedges_from_bincenters(xcenters)
    ybins = binedges_from_bincenters(ycenters)

    #histogram
    psf, _ = np.histogramdd((xr, yr), weights=factor, bins=(xbins, ybins))
    xaxis = (np.min(xbins), np.max(xbins), xbins[1]-xbins[0])
    yaxis = (np.min(ybins), np.max(ybins), ybins[1]-ybins[0])

    return create_xy_function(psf, xaxis, yaxis)



##################################
######### LIGTH TABLE ############
##################################
def binedges_from_bincenters(bincenters):
    ds = np.diff(bincenters)
    if ~np.all(ds == ds[0]):
        raise Exception("Bin distances must be equal")

    d = ds[0]
    return np.arange(bincenters[0]-d/2., bincenters[-1]+d/2.+d, d)


def create_xyz_function(array, xaxis, yaxis, zaxis):
    xmin, xmax, dx = xaxis
    ymin, ymax, dy = yaxis
    zmin, zmax, dz = zaxis
    def function(x, y, z):
        x = np.clip(x, xmin, xmax-0.1*dx)
        y = np.clip(y, ymin, ymax-0.1*dx)
        z = np.clip(z, zmin, zmax-0.1*dz)
        ix = ((x-xmin)//dx).astype(int)
        iy = ((y-ymin)//dy).astype(int)
        iz = ((z-zmin)//dz).astype(int)
        return array[ix, iy, iz]
    return function


def get_ligthtables(filename: str)->Callable:
    ##### Load LT ######
    with tb.open_file(filename) as h5file:
        LT = h5file.root.LightTable.table.read()

    #### XYZ binning #####
    x, y, z = LT["x"], LT["y"], LT["z"]

    xcenters, ycenters, zcenters = np.unique(x), np.unique(y), np.unique(z)
    xbins = binedges_from_bincenters(xcenters)
    ybins = binedges_from_bincenters(ycenters)
    zbins = binedges_from_bincenters(zcenters)

    xaxis = np.min(xbins), np.max(xbins), np.diff(xbins)[0]
    yaxis = np.min(ybins), np.max(ybins), np.diff(ybins)[0]
    zaxis = np.min(zbins), np.max(zbins), np.diff(zbins)[0]

    ###### CREATE XYZ FUNCTION FOR EACH SENSOR ######
    func_per_sensor = []
    sensors = ["FIBER_SENSOR_10000", "FIBER_SENSOR_10001"]
    for sensor in sensors:
        w = LT[sensor]

        H, _ = np.histogramdd((x, y, z), weights=w, bins=[xbins, ybins, zbins])
        fxyz = create_xyz_function(H, xaxis, yaxis, zaxis)

        func_per_sensor.append(fxyz)

    ###### CREATE XYZ CALLABLE FOR LIST OF XYZ FUNCTIONS #####
    def merge_list_of_functions(list_of_functions):
        def merged(x, y, z):
            return [f(x, y, z) for f in list_of_functions]
        return merged

    return merge_list_of_functions(func_per_sensor)
