import numpy  as np

#######################################
########## PHOTON SIMULATION ##########
#######################################
def generate_s1_photons(ws       : float,
                        energies : np.ndarray) -> np.ndarray:
    """generates s1 photons from the energy hits

    Parameters:
        :ws: float
            scintillation yield
        :energies: np.ndarray
            energy hits
    Returns:
        :: np.ndarray
            s1 photons per hit
    """
    return np.random.poisson(energies / ws)


def generate_s2_photons(el_gain        : float,
                        el_gain_sigma  : float,
                        nes            : np.ndarray) -> np.ndarray:
    """generate number of EL-photons produced by secondary electrons that reach
    the EL (after drift and diffusion)

    Parameters:
        :el_gain: float
            number of phtotons emitted per ionization electron at EL
        :el_gain_sigma: float
            standard deviation of the el_gain.
        :nes: np.ndarray
            number of ionization electrons at arriving at some EL X,Y position.
    Returns:
        :: np.ndarray
            s2 photons electrons at the EL
    """
    return np.random.normal(el_gain*nes, np.sqrt(el_gain_sigma**2*nes))
