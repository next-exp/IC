import numpy as np

from typing import Tuple

from .. core import system_of_units as units

def generate_ionization_electrons(energies    : np.ndarray,
                                  wi          : float = 22.4 * units.eV,
                                  fano_factor : float = 0.15) -> np.ndarray:
    """ Generate secondary ionization electrons from energy deposits

    Parameters:
        :wi: float
            Mean ionization energy
        :fano_factor: float
            Fano-factor. related with the deviation in ionization electrons
        :energies: np.ndarray
            Energy hits
    Returns:
        :nes: np.ndarray
            The ionization electrons per hit

    Comment:
        The quotient energy/wi gives the mean value of the ionization electrons produced
        in a hit (N).
        The number ionization electrons produced in a hit (N) are assumed to be normally
        distributed with mean N and variance (N*F). (Being F the fano_factor).
        If the variance is such that <1, the electrons are assumed to be poisson distributed.
    """
    nes = energies / wi
    var = nes * fano_factor
    pois = var < 1      #See docstring for explanation
    nes[ pois] = np.random.poisson(nes[pois])
    nes[~pois] = np.round(np.random.normal(nes[~pois], np.sqrt(var[~pois])))
    nes[nes<0] = 0
    return nes.astype(int)


def drift_electrons(zs             : np.ndarray,
                    n_electrons    : np.ndarray,
                    lifetime       : float = 12 * units.ms,
                    drift_velocity : float = 1  * units.mm / units.mus) -> np.ndarray:
    """ Returns number of electrons due to lifetime losses from secondary electrons

    Parameters:
        :lifetime: float
            Electron lifetime
        :drift_velocity: float
            Drift velocity at the active volume of the detector
        :zs:
            z coordinate of the ionization hits
        :electrons:
            Number of ionization electrons in each hit
    Returns:
        :nes: np.ndarray
            Number of ionization electrons that reach the EL gap
    """
    @np.vectorize
    def attachment(n_ie, t):
        return np.count_nonzero(-lifetime * np.log(np.random.uniform(size=n_ie)) > t)

    ts  = zs / drift_velocity
    return attachment(n_electrons, ts)


def diffuse_electrons(xs                     : np.ndarray,
                      ys                     : np.ndarray,
                      zs                     : np.ndarray,
                      n_electrons            : np.ndarray,
                      transverse_diffusion   : float = 1.0 * units.mm / units.cm**0.5,
                      longitudinal_diffusion : float = 0.2 * units.mm / units.cm**0.5)\
                      -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Starting from hits with positions xs, ys, zs, and number of electrons,
    apply diffusion and return diffused positions xs, ys, zs for each electron.

    Paramters:
        :transverse_diffusion: float
        :longitudinal_diffusion: float
        :xs, ys, zs: np.ndarray (1D of size: #hits)
            Postions of initial hits
        :electrons:
            Number of ionization electrons per hit before drifting
    Returns:
        :dxs, dys, dzs: np.ndarray (1D of size: #hits x #electrons-per-hit)
            Diffused positions at the EL
    """
    xs = np.repeat(xs, n_electrons)
    ys = np.repeat(ys, n_electrons)
    zs = np.repeat(zs, n_electrons)

    # substitute z<0 to z=0
    sel = zs<0
    zs[sel] = 0

    sqrtz = zs ** 0.5
    dxs  = np.random.normal(xs, sqrtz *   transverse_diffusion)
    dys  = np.random.normal(ys, sqrtz *   transverse_diffusion)
    dzs  = np.random.normal(zs, sqrtz * longitudinal_diffusion)

    return (dxs, dys, dzs)
