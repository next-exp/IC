import sys
import numpy as np

from invisible_cities.reco.params import Cluster
from invisible_cities.core.system_of_units_c import units

def barycenter(xs, ys, qs, default=np.nan):
    q    = np.sum(qs)
    n    =    len(qs)
    if n and q > 0:
        x    = np.average(xs, weights=qs)
        y    = np.average(ys, weights=qs)
        xvar = np.sum(qs * (xs - x)**2) / (q - 1)
        yvar = np.sum(qs * (ys - y)**2) / (q - 1)
    else:
        x, y, xvar, yvar = [default] * 4
    return Cluster(q, x, y, xvar**0.5, yvar**0.5, n)

def select_sipms(sis, xs, ys, qs):
    return xs[sis], ys[sis], qs[sis]

def discard_sipms(sis, xs, ys, qs):
    return np.delete(xs, sis), np.delete(ys, sis), np.delete(qs, sis)

def get_nearby_sipm_inds(xc, yc, d, xs, ys, qs):
    """return indices of sipms less than d from (xc,yc)"""
    return np.where(np.sqrt((xs - xc)**2 + (ys - yc)**2) <= d)[0]

def corona(xs, ys, qs, Qthr  =  0*units.pes,
                       Qlm   =  5*units.pes,
                       slm   = 15*units.mm ,
                       rmax  = 25*units.mm ,
                       msipm =  3          ):
    """
    corona creates a list of Clusters by
    first , identifying a loc max (gonz wanted more precise than just max sipm)
    second, calling barycenter to find the Cluster given by SiPMs around the max
    third , removing (nondestructively) the sipms contributing to that Cluster
    until there are no more local maxima

    kwargs
    Qthr : SiPMs with less than Qthr pes are ignored
    Qlm  : local maxima must have a SiPM with at least T pes
    slm  : all SiPMs within slm distance from the local max SiPM are used (by
           barycenter) to compute the approximate center of the local max.
    rmax : xs,ys,qs, of SiPMs within rmax of a local max are used by barycenter
           to compute a Cluster.
    msipm: the minimum number of SiPMs found with

    returns
    c    : a list of Clusters
    """
    c  = []
    # Keep SiPMs with at least Qthr pes
    xs, ys, qs = select_sipms(np.where(qs >= Qthr)[0], xs, ys, qs)

    # While there are more local maxima
    while len(qs) > 0:
        mx = np.argmax(qs)       # SiPM with largest Q
        if qs[mx] < Qlm: break   # largest Q remaining is negligible

        # find locmax (the baryc of charge in SiPMs less than slm from mx)
        sis = get_nearby_sipm_inds(xs[mx], ys[mx], slm, xs, ys, qs)
        lm  = barycenter(*select_sipms(sis, xs, ys, qs))

        # rsis is an array of the responsive sipms less than rmax from locmax
        rsis = get_nearby_sipm_inds(lm.X, lm.Y, rmax, xs, ys, qs)

        # if rsis have at least msipms, get the barycenter
        if len(rsis) >= msipm:
            c.append(barycenter(*select_sipms(rsis, xs, ys, qs)))

        # delete the SiPMs contributing to this cluster
        xs, ys, qs = discard_sipms(rsis, xs, ys, qs)

    return c
