from argparse import Namespace

import numpy  as np
import tables as tb
import pandas as pd


from .. database            import load_db
from .. core                import fit_functions as fitf
from .. core.core_functions import in_range
from .. reco.corrections    import Correction
from .. reco.corrections    import LifetimeCorrection
from .. io  .dst_io         import load_dst
from .. io  .kdst_io        import xy_correction_writer

from .  components import city


@city
def zaira(files_in, file_out,
          event_range,   # not used, by config insists on sending it
          dst_group, dst_node,
          lifetime, u_lifetime,
          xbins, ybins,
          xmin = None, xmax = None,
          ymin = None, ymax = None,
          rmin = None, rmax = None,
          zmin = None, zmax = None,
          emin = None, emax = None):

    fiducial_x     = get_x_limits(xmin, xmax)
    fiducial_y     = get_y_limits(ymin, ymax)
    fiducial_r     = get_r_limits(rmin, rmax)
    fiducial_z     = get_z_limits(zmin, zmax)
    fiducial_e     = get_e_limits(emin, emax)
    lt_corrections = get_lifetime_corrections(lifetime, u_lifetime)

    dsts = [load_dst(input_file, dst_group, dst_node) for input_file in files_in]

    # Correct each dataset with the corresponding lifetime
    for dst, correct in zip(dsts, lt_corrections):
        dst.S2e *= correct(dst.Z.values).value

    # Join datasets
    dst = pd.concat(dsts)

    # select fiducial region
    fid_dst =     dst
    fid_dst = fid_dst[in_range(fid_dst.S2e.values, *fiducial_e)]
    fid_dst = fid_dst[in_range(fid_dst.Z  .values, *fiducial_z)]
    fid_dst = fid_dst[in_range(fid_dst.R  .values, *fiducial_r)]
    fid_dst = fid_dst[in_range(fid_dst.X  .values, *fiducial_x)]
    fid_dst = fid_dst[in_range(fid_dst.Y  .values, *fiducial_y)]

    # Compute corrections and stats
    xycorr, nevt = profile_xy(fid_dst.X.values, fid_dst.Y.values, fid_dst.S2e.values,
                              xbins, ybins, fiducial_x, fiducial_y)

    with tb.open_file(file_out, 'w') as h5out:
        write_xy = xy_correction_writer(h5out)
        write_xy(*xycorr._xs, xycorr._fs, xycorr._us, nevt)

    return Namespace(events_in  = len(    dst),
                     events_out = len(fid_dst))


def get_x_limits(xmin, xmax):
    det_geo = load_db.DetectorGeo()
    x_min   = det_geo.XMIN[0] if xmin is None else xmin
    x_max   = det_geo.XMAX[0] if xmax is None else xmax
    return x_min, x_max


def get_y_limits(ymin, ymax):
    det_geo = load_db.DetectorGeo()
    y_min   = det_geo.YMIN[0] if ymin is None else ymin
    y_max   = det_geo.YMAX[0] if ymax is None else ymax
    return y_min, y_max


def get_z_limits(zmin, zmax):
    det_geo = load_db.DetectorGeo()
    z_min   = det_geo.ZMIN[0] if zmin is None else zmin
    z_max   = det_geo.ZMAX[0] if zmax is None else zmax
    return z_min, z_max


def get_r_limits(rmin, rmax):
    det_geo = load_db.DetectorGeo()
    r_min   =              0  if rmin is None else rmin
    r_max   = det_geo.RMAX[0] if rmax is None else rmax
    return r_min, r_max


def get_e_limits(emin, emax):
    e_min   =              0  if emin is None else emin
    e_max   =         np.inf  if emax is None else emax
    return e_min, e_max


def get_lifetime_corrections(lifetime, u_lifetime):
    lifetimes   = [  lifetime] if not np.shape(  lifetime) else   lifetime
    u_lifetimes = [u_lifetime] if not np.shape(u_lifetime) else u_lifetime
    return tuple(map(LifetimeCorrection, lifetimes, u_lifetimes))


def profile_xy(X, Y, E, xbins, ybins, xrange, yrange):
    xs, ys, es, us = fitf.profileXY(X, Y, E,
                                    xbins , ybins ,
                                    xrange, yrange)
    nevt           = np.histogram2d(X, Y,
                                    (xbins ,  ybins),
                                    (xrange, yrange))[0]
    return Correction((xs, ys), es, us), nevt
