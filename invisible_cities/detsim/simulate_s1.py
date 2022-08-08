"""
---------------------------
    detsim s1 simulation
---------------------------

This module contains the functions that implement s1 simulation in detsim.
s1 is simulated following the steps:

- compute the number of scintillation photons emitted by track-hits.
- use the s1 lighttable to compute the number of pes at each PMT produced by each hit.
-----------
Important: Note that not delay is assumed between emission of light from the track and the arrival time
at the PMTs. In detsim the s1 photons arrive instantaneously to the PMTs.
This can cause some mismatch between detsim and fullsim s1 variables in large detectors.
----------
- for each pes arriving at the PMT, the s1-times (ie, the times in which they generate signal)
are computed using the s1-distribution plus the hit emission time.
- finally, the s1-times are bufferized in a waveform.
"""

import os
import numpy as np

from typing import Callable

from scipy.optimize import brentq

from .. core import system_of_units as units


def compute_scintillation_photons(energy : np.ndarray,
                                  ws     : float) -> np.ndarray:
    """
    Computes the number of scintillation photons produced
    in each energy deposition hit
    Parameters:
        :energy: np.ndarray
            vector with the energy values of each hit
        :ws: float
            inverse scintillation yield
    Returns:
        np.ndarray
        The number of photons at each hit they are poisson
        distributed with mean energy/ws
    """
    return np.random.poisson(energy / ws)


def compute_s1_pes_at_pmts(xs      : np.ndarray,
                           ys      : np.ndarray,
                           zs      : np.ndarray,
                           photons : np.ndarray,
                           lt      : Callable  )->np.ndarray:
    """
    Compute the pes generated in each PMT from S1 photons
    Parameters:
        :LT: function
            The Light Table in functional form
        :photons: np.ndarray
            The photons generated at each hit
        :xs, ys, zs: np.ndarray
            hit position
    Returns:
        :pes: np.ndarray
            photoelectrons at each PMT produced by all hits.
            shape is (number_of_sensors, number_of_hits)
    """
    pes = photons[:, np.newaxis] * lt(xs, ys, zs)
    return pes.T


def s1_times_pdf(x):
    """
    Implements the s1-times pdf (values in nano-seconds)
    Reference:
    """
    tau1 = 4.5 * units.ns
    tau2 = 100 * units.ns
    p1 = 0.1
    norm = (tau1*p1 + tau2*(1-p1))
    return (1./norm) * (p1*np.exp(-x/tau1) + (1-p1)*np.exp(-x/tau2))
