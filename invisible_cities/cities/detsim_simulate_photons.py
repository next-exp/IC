import numpy  as np

from typing import Callable


#######################################
########## PHOTON SIMULATION ##########
#######################################
def generate_s1_photons(energies : np.ndarray,
                        ws       : float) -> np.ndarray:
    """generate s1 photons"""
    return np.random.poisson(energies / ws)


def generate_s2_photons(x              : np.ndarray,
                        el_gain        : float,
                        el_gain_sigma  : float) -> np.ndarray:
    """generate number of EL-photons produced by secondary electrons that reach
    the EL (after drift and diffusion)
    """
    n = len(x)
    nphs = np.random.normal(el_gain, el_gain_sigma, size = n)
    return nphs


def pes_at_sensors(xs         : np.ndarray,
                   ys         : np.ndarray,
                   zs         : np.ndarray,
                   photons    : np.ndarray,
                   x_sensors  : np.ndarray,
                   y_sensors  : np.ndarray,
                   z_sensors  : float   ,
                   psf : Callable) -> np.ndarray:
    """compute the pes that reach each sensor, based on
    the sensor psf"""

    dxs = xs[:, np.newaxis] - x_sensors
    dys = ys[:, np.newaxis] - y_sensors
    dzs = zs[:, np.newaxis] - z_sensors
    photons = photons[:, np.newaxis]

    pes = photons * psf(dxs, dys, dzs)
    pes = np.random.poisson(pes)
    return pes.T
