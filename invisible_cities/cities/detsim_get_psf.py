import numpy  as np
import tables as tb
import pandas as pd

from typing import Tuple
from typing import Callable

from invisible_cities.reco.corrections_new import read_maps

from invisible_cities.core.core_functions  import in_range


###################################
############# UTILS ###############
###################################
def create_xyz_function(H, bins):
    """Given a 3D array and a list of bins for
    each dim, it returns a x,y,z function"""

    xbins, ybins, zbins = bins
    if not H.shape == (len(xbins)-1, len(ybins)-1, len(zbins)-1):
        raise Exception("bins and array shapes not consistent")

    def function(x, y, z):
        if not x.shape==y.shape==z.shape:
            raise Exception("x, y and z must have same size")

        out = np.zeros(x.shape)
        #select values inside bin extremes
        selx = in_range(x, xbins[0], xbins[-1])
        sely = in_range(y, ybins[0], ybins[-1])
        selz = in_range(z, zbins[0], zbins[-1])
        sel = selx & sely & selz

        ix = np.digitize(x[sel], xbins)-1
        iy = np.digitize(y[sel], ybins)-1
        iz = np.digitize(z[sel], zbins)-1

        out[sel] = H[ix, iy, iz]
        return out
    return function


def binedges_from_bincenters(bincenters):
    ds = np.diff(bincenters)
    if ~np.all(ds == ds[0]):
        raise Exception("Bin distances must be equal")

    d = ds[0]
    return np.arange(bincenters[0]-d/2., bincenters[-1]+d/2.+d, d)


##################################
############# PSF ################
##################################
def _psf(dx, dy, dz, factor = 1.):
    """ generic analytic PSF function
    """
    return factor * np.abs(dz) / (2 * np.pi) / (dx**2 + dy**2 + dz**2)**1.5


def get_psf_from_krmap(filename : str,
                       factor   : float = 1.)->Callable:
    """ reads KrMap and generate a psf function with the E0-map
    """
    maps = read_maps(filename)
    xmin, xmax, ymin, ymax, nx, ny, _ = maps.mapinfo.values
    dx   = (xmax - xmin)/ float(nx)
    dy   = (ymax - ymin)/ float(ny)
    e0map  = factor * np.nan_to_num(np.array(maps.e0), 0.)

    xbins = np.arange(xmin, xmax+dx, dx)
    ybins = np.arange(ymin, ymax+dy, dy)
    zbins = np.array([0, 1])

    H = e0map[:, :, np.newaxis]
    return create_xyz_function(H, [xbins, ybins, zbins])


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
    zbins = np.array([0, 1])

    #histogram
    psf, _ = np.histogramdd((xr, yr), weights=factor, bins=(xbins, ybins))

    H = psf[:, :, np.newaxis]
    return create_xyz_function(H, [xbins, ybins, zbins])


##################################
######### LIGTH TABLE ############
##################################
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
    bins  = [xbins, ybins, zbins]

    ###### CREATE XYZ FUNCTION FOR EACH SENSOR ######
    func_per_sensor = []
    sensors = ["FIBER_SENSOR_10000", "FIBER_SENSOR_10001"]
    for sensor in sensors:
        w = LT[sensor]

        H, _ = np.histogramdd((x, y, z), weights=w, bins=bins)
        fxyz = create_xyz_function(H, bins)

        func_per_sensor.append(fxyz)

    ###### CREATE XYZ CALLABLE FOR LIST OF XYZ FUNCTIONS #####
    def merge_list_of_functions(list_of_functions):
        def merged(x, y, z):
            return [f(x, y, z) for f in list_of_functions]
        return merged

    return merge_list_of_functions(func_per_sensor)
