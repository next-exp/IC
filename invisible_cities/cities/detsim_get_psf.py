import numpy  as np
import tables as tb
import pandas as pd

from typing import Callable

from invisible_cities.reco.corrections     import read_maps
from invisible_cities.core.core_functions  import in_range

from invisible_cities.core import system_of_units as units

###################################
############# UTILS ###############
###################################
def create_xyz_function(H    : np.ndarray,
                        bins : list)->Callable:
    """Given a 3D array and a list of bins for
    each dim, it returns a x,y,z function

    Parameters:
        :H: np.ndarray
            3D histogram
        :bins: list[np.ndarray, np.ndarray, np.ndarray]
            list with the bin edges of :H:. The i-element corresponds
            with the bin edges of the i-axis of :H:.
    Returns:
        :function:
            x, y, z function that returns the correspondig histogram value
    """

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

def create_xy_function(H    : np.ndarray,
                       bins : list)->Callable:
    """Given a 2D array and a list of bins for
    each dim, it returns a x,y,z function

    Parameters:
        :H: np.ndarray
            2D histogram
        :bins: list[np.ndarray, np.ndarray]
            list with the bin edges of :H:. The i-element corresponds
            with the bin edges of the i-axis of :H:.
    Returns:
        :function:
            x, y function that returns the correspondig histogram value
    """

    xbins, ybins = bins
    if not H.shape == (len(xbins)-1, len(ybins)-1):
        raise Exception("bins and array shapes not consistent")

    def function(x, y):
        if not x.shape==y.shape:
            raise Exception("x, y and z must have same size")

        out = np.zeros(x.shape)
        #select values inside bin extremes
        selx = in_range(x, xbins[0], xbins[-1])
        sely = in_range(y, ybins[0], ybins[-1])
        sel = selx & sely

        ix = np.digitize(x[sel], xbins)-1
        iy = np.digitize(y[sel], ybins)-1

        out[sel] = H[ix, iy]
        return out
    return function


def binedges_from_bincenters(bincenters: np.ndarray)->np.ndarray:
    """
    computes bin-edges from bin-centers. The extremes of the edges are asigned to
    the extremes of the bin centers.

    Parameters:
        :bincenters: np.ndarray
            bin centers
    Returns:
        :binedges: np.ndarray
            bin edges
    """
    binedges = np.zeros(len(bincenters)+1)

    binedges[1:-1] = (bincenters[1:] + bincenters[:-1])/2.
    binedges[0]  = bincenters[0]
    binedges[-1] = bincenters[-1]

    return binedges

##################################
############# PSF ################
##################################
def get_psf(filename : str,
            drift_velocity_EL : float = 2.5,
            wf_sipm_bin_width : float = 100):
    """
    From PSF filename, returns a function of distance to SIPMs

    Parameters:
        :filename: str
            path to the PSF h5 file
    Returns:
        :get_psf_values: function

        :info:
    """
    PSF    = pd.read_hdf(filename, "/LightTable")
    Config = pd.read_hdf(filename, "/Config")
    EL_dz = float(Config.loc["EL_GAP"])        * units.mm
    pitch = float(Config.loc["pitch_z"].value) * units.mm
    npartitions = int(EL_dz/pitch)

    distance_bins    = np.sort(PSF.index.values)
    psf_max_distance = np.max(distance_bins)
    PSF = PSF.values
    ELtimes = np.arange(pitch/2., EL_dz, pitch)/drift_velocity_EL

    ELtimepartitions_bins = np.arange(0, ELtimes[-1] + wf_sipm_bin_width, wf_sipm_bin_width)
    n_time_bins = len(ELtimepartitions_bins)-1
    indexes    = np.digitize(ELtimes, ELtimepartitions_bins)-1
    _, indexes = np.unique(indexes, return_index=True)

    splitted_PSF = np.split(PSF, indexes[1:], axis=1)

    effective_PSF = [np.sum(cols, axis=1, keepdims=True)*(1/npartitions) for cols in splitted_PSF]
    effective_PSF  = np.hstack(effective_PSF)

    def get_psf_values(distances):
        psf = np.zeros((len(distances), n_time_bins))
        sel = distances<=psf_max_distance
        indexes  = np.digitize(distances[sel], distance_bins)-1
        psf[sel] = effective_PSF[indexes]
        return psf

    info = (EL_dz, pitch, npartitions, n_time_bins)
    return get_psf_values, info



##################################
######### LIGTH TABLE ############
##################################
def get_ligthtables(filename: str,
                    signal  : str)->Callable:
    """
    From LT filename, returns a function of x,y,z for S1 LTs and x,y for S2
    with the values of the LT for each sensor

    Parameters:
        :filename: str
            path to the PSF h5 file
    Returns:
        :merged: function
            (for S1) a function that merge the x, y, z functions for each sensor.
            (each sensor has its own x, y, z distribution of ligth). The output would be
            a np.ndarray with the LT value for x,y,z for each sensor.
            (for S2) the same, being a x,y dependence
    """
    ##### Load LT ######
    with tb.open_file(filename) as h5file:
        LT = h5file.root.LightTable.table.read()
        info = dict(h5file.root.Config.table.read())

    sensor = str(info[b'sensor'], errors="ignore")
    sensors = [name for name in LT.dtype.names if sensor in name and "total" not in name]
    sensors.sort(key=lambda name: int(name.split("_")[-1]))

    if signal == "S1":
        #### XYZ binning #####
        x, y, z = LT["x"], LT["y"], LT["z"]

        xcenters, ycenters, zcenters = np.unique(x), np.unique(y), np.unique(z)
        xbins = binedges_from_bincenters(xcenters)
        ybins = binedges_from_bincenters(ycenters)
        zbins = binedges_from_bincenters(zcenters)
        bins  = [xbins, ybins, zbins]

        ###### CREATE XYZ FUNCTION FOR EACH SENSOR ######
        func_per_sensor = []
        for sensor in sensors:
            w = LT[sensor]
            H, _ = np.histogramdd((x, y, z), weights=w, bins=bins)
            fxyz = create_xyz_function(H, bins)
            func_per_sensor.append(fxyz)

        ###### CREATE XYZ CALLABLE FOR LIST OF XYZ FUNCTIONS #####
        def merge_list_of_functions(list_of_functions):
            def merged(x, y, z):
                return np.array([f(x, y, z) for f in list_of_functions]).T
            return merged
        return merge_list_of_functions(func_per_sensor)

    elif signal == "S2":
        #### XYZ binning #####
        x, y = LT["x"], LT["y"]

        xcenters, ycenters = np.unique(x), np.unique(y)
        xbins = binedges_from_bincenters(xcenters)
        ybins = binedges_from_bincenters(ycenters)
        bins  = [xbins, ybins]

        ###### CREATE XY FUNCTION FOR EACH SENSOR ######
        func_per_sensor = []
        for sensor in sensors:
            w = LT[sensor]
            H, _ = np.histogramdd((x, y), weights=w, bins=bins)
            fxy = create_xy_function(H, bins)
            func_per_sensor.append(fxy)

        ###### CREATE XY CALLABLE FOR LIST OF XY FUNCTIONS #####
        def merge_list_of_functions(list_of_functions):
            def merged(x, y):
                return np.array([f(x, y) for f in list_of_functions]).T
            return merged
        return merge_list_of_functions(func_per_sensor)
