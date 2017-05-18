import sys
import numpy as np

from .. core.system_of_units_c import units
from .. core.exceptions        import SipmEmptyList
from .. core.exceptions        import SipmZeroCharge
from .       params            import Cluster

def barycenter(pos, qs):
    if not len(pos): raise SipmEmptyList
    if sum(qs) == 0: raise SipmZeroCharge
    mu  = np.average( pos           , weights=qs, axis=0)
    std = np.average((pos - mu) ** 2, weights=qs, axis=0)
    return Cluster(sum(qs), mu, std, len(qs))

def select_sipms(sis, pos, qs):
    return pos[sis], qs[sis]

def discard_sipms(sis, pos, qs):
    return np.delete(pos, sis, axis=0), np.delete(qs, sis)

def get_nearby_sipm_inds(cs, d, pos, qs):
    """return indices of sipms less than d from (xc,yc)"""
    return np.where(np.linalg.norm(pos - cs, axis=1) <= d)[0]


def corona(pos, qs, Qthr  =  0*units.pes,
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
    pos, qs = select_sipms(np.where(qs >= Qthr)[0], pos, qs)
    # While there are more local maxima
    while len(qs) > 0:
        mx = np.argmax(qs)       # SiPM with largest Q
        if qs[mx] < Qlm: break   # largest Q remaining is negligible

        # find locmax (the baryc of charge in SiPMs less than slm from mx)
        sis = get_nearby_sipm_inds(pos[mx], slm, pos, qs)
        lm  = barycenter(*select_sipms(     sis, pos, qs))

        # rsis is an array of the responsive sipms less than rmax from locmax
        rsis = get_nearby_sipm_inds(lm.pos, rmax, pos, qs)

        # if rsis have at least msipms, get the barycenter
        if len(rsis) >= msipm:
            c.append(barycenter(*select_sipms(rsis, pos, qs)))

        # delete the SiPMs contributing to this cluster
        pos, qs = discard_sipms(rsis, pos, qs)

    return c
