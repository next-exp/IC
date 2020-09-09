import numpy as np

from typing import Tuple
from typing import Callable

from scipy.stats  import rv_continuous

##################################
############## PES ###############
##################################
def pes_at_pmts(LT      : Callable  ,
                photons : np.ndarray,
                xs      : np.ndarray,
                ys      : np.ndarray,
                zs      : np.ndarray = None)->np.ndarray:
    """Compute the pes generated in each PMT from photons generated at some point

    Parameters:
        :LT: function
            The Liht Table in functional form
        :photons: np.ndarray
            The photons generated at each hit (in the active volume
            for the S1 and in the EL for S2)
        :xs, ys, zs: np.ndarray
            hit position (zs=None for S2)
    Returns:
        :pes: np.ndarray
            photoelectrons at each PMT produced by each hit.
            Shape is (nsensors, nhits)
    """
    if np.any(zs): #S1
        pes = photons[:, np.newaxis] * LT(xs, ys, zs)
    else:          #S2
        pes = photons[:, np.newaxis] * LT(xs, ys)
    pes = np.random.poisson(pes)
    return pes.T


##################################
############## TIMES #############
##################################
class S1_TIMES(rv_continuous):
    """S1 times distribution generator.
    Following distribution 0.1*exp(t/4.5) + 0.9*exp(t/100)"""

    def __init__(self):
        super().__init__(a=0)

    def _pdf(self, x):
        return (0.1*np.exp(-x/4.5) + 0.9*np.exp(-x/100))*1/(0.1*4.5 + 0.9*100)

generate_S1_time = S1_TIMES()

def generate_S1_times_from_pes(S1pes_at_pmts : np.ndarray)->list:
    """Given the S1pes_at_pmts, this function returns the times at which the pes
    are be distributed (see generate_S1_time function).
    It returns a list whose elements are the times at which the photoelectrons in that PMT
    are generated.

    Parameters:
        :S1pes_at_pmts: np.ndarray
            the pes at each PMT generated by each hit
    Returns:
        :S1times: list[np.ndarray,..]
            Each element are the S1 times for a PMT. If certain sensor
            do not see any pes, the array is empty.
    """
    S1pes_pmt = np.sum(S1pes_at_pmts, axis=1)
    S1times = [generate_S1_time.rvs(size=pes) for pes in S1pes_pmt]
    return S1times
