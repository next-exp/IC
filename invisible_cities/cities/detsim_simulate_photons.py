import numpy  as np

from typing import Callable


#######################################
########## PHOTON SIMULATION ##########
#######################################
def generate_s1_photons(energies : np.array,
                        ws       : float) -> np.array:
    """generate s1 photons"""
    return np.random.poisson(energies / ws)


def generate_s2_photons(x              : np.array,
                        el_gain        : float,
                        el_gain_sigma  : float) -> np.array:
    """generate number of EL-photons produced by secondary electrons that reach
    the EL (after drift and diffusion)
    """
    n = len(x)
    nphs = np.random.normal(el_gain, el_gain_sigma, size = n)
    return nphs


def pes_at_sensors(xs         : np.array,
                   ys         : np.array,
                   zs         : np.array,
                   photons    : np.array,
                   x_sensors  : np.array,
                   y_sensors  : np.array,
                   z_sensors  : float   ,
                   psf : Callable) -> np.array:
    """compute the photons that reach each sensor, based on
    the sensor psf"""

    dxs = xs[:, np.newaxis] - x_sensors
    dys = ys[:, np.newaxis] - y_sensors
    dzs = zs[:, np.newaxis] - z_sensors
    photons = photons[:, np.newaxis]

    pes = photons * psf(dxs, dys, dzs)
    pes = np.random.poisson(pes)
    return pes.T
