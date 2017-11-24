import numpy  as np

from .  corrections import Correction
from .  corrections import LifetimeXYCorrection
from .. io.dst_io   import load_dst


def load_z_corrections(filename):
    dst = load_dst(filename, "Corrections", "Zcorrections")
    return Correction((dst.z.values,), dst.factor.values, dst.uncertainty.values)


def load_xy_corrections(filename, *,
                        group = "Corrections",
                        node  = "XYcorrections",
                        **kwargs):
    dst  = load_dst(filename, group, node)
    x, y = np.unique(dst.x.values), np.unique(dst.y.values)
    f, u = dst.factor.values, dst.uncertainty.values

    return Correction((x, y),
                      f.reshape(x.size, y.size),
                      u.reshape(x.size, y.size),
                      **kwargs)


def load_lifetime_xy_corrections(filename, *,
                                 group = "Corrections",
                                 node  = "LifetimeXY",
                                 scale = 1,
                                 **kwargs):
    """
    Load the lifetime map from hdf5 file.

    Parameters
    ----------
    filename: str
        Path to the file containing the map.
    group: str
        Name of the group where the table is stored.
    node: str
        Name of the table containing the data.
    scale: float
        Scale factor for the lifetime values.

    Other kwargs are passed to the contructor of LifetimeXYCorrection.
    """
    dst  = load_dst(filename, group, node)
    x, y = np.unique(dst.x.values), np.unique(dst.y.values)
    f, u = dst.factor.values * scale, dst.uncertainty.values * scale

    return LifetimeXYCorrection(f.reshape(x.size, y.size),
                                u.reshape(x.size, y.size),
                                x, y, **kwargs)
