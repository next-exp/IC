"""Waveform Functions
This module includes functions to manipulate waveforms.
authors: J.J. Gomez-Cadenas, G. Martinez
"""
import numpy as np
from typing import Optional
from typing  import Callable

from .. core.core_functions import define_window
from .. calib               import calib_sensors_functions as csf
from .. sierpe              import blr

def to_adc(wfs, adc_to_pes):
    """
    Convert waveform in pes to adc.

    Parameters
    ----------
    wfs : 2-dim np.ndarray
        The waveform (axis 1) for each sensor (axis 0).
    adc_to_pes : a vector of constants
        Contains the sensor-related information.

    Returns
    -------
    adc_wfs : 2-dim np.ndarray
        The input wfs scaled to adc.
    """
    return wfs * adc_to_pes.reshape(wfs.shape[0], 1)


def to_pes(wfs, adc_to_pes):
    """
    Convert waveform in adc to pes.

    Parameters
    ----------
    wfs : 2-dim np.ndarray
        The waveform (axis 1) for each sensor (axis 0).
    adc_to_pes : a vector of constants
        Contains the sensor-related information.

    Returns
    -------
    pes_wfs : 2-dim np.ndarray
        The input wfs scaled to pes.
    """
    return wfs / adc_to_pes.reshape(wfs.shape[0], 1)


def suppress_wf(waveform, threshold, padding = 0):
    """Put zeros where the waveform is below some threshold.

    Parameters
    ----------
    waveform : 1-dim np.ndarray
        Waveform amplitudes.
    threshold : int or float
        Cut value.

    Returns
    -------
    wf : 1-dim np.ndarray
        A copy of the input waveform with values below threshold set to zero.
    """
    wf = np.copy(waveform)
    below_thr = wf <= threshold
    if padding > 0:
        ## pad around signal
        indices = np.argwhere(np.invert(below_thr)).flatten()
        min_range = np.clip(indices - padding, 0, None)
        max_range = np.clip(indices + padding, None, len(below_thr)-1)
        for min_indx, max_indx in zip(min_range, max_range):
            below_thr[min_indx:max_indx] = False
    wf[below_thr] = 0
    return wf


def noise_suppression(waveforms, thresholds, padding = 0):
    """Put zeros where the waveform is below some threshold.

    Parameters
    ----------
    waveforms : 2-dim np.ndarray
        Waveform amplitudes (axis 1) for each sensor (axis 0).
    thresholds : int or float or sequence of ints or floats
        Cut value for each waveform (sequence) or for all (single number).
    padding : Number of samples before and after signal to keep

    Returns
    -------
    suppressed_wfs : 2-dim np.ndarray
        A copy of the input waveform with values below threshold set to zero.
    """
    if not hasattr(thresholds, "__iter__"):
        thresholds = np.ones(waveforms.shape[0]) * thresholds
    if not hasattr(padding, "__iter__"):
        padding = np.zeros(waveforms.shape[0], dtype = int) + padding
    suppressed_wfs = list(map(suppress_wf, waveforms, thresholds, padding))
    return np.array(suppressed_wfs)


def cwf_from_rwf(pmtrwf, event_list, calib_vectors, deconv_params):
    """Compute CWF from RWF for a given list of events."""

    CWF=[]
    for event in event_list:
        pmt_evt = pmtrwf[event]
        ZWF     = csf.means(pmt_evt[:, :deconv_params.n_baseline]) - pmt_evt
        rep_thr = np.repeat(deconv_params.thr_trigger, ZWF.shape[0])
        CWF.append(np.array(tuple(map(blr.deconvolve_signal, ZWF,
                                      calib_vectors.coeff_c     ,
                                      calib_vectors.coeff_blr   ,
                                      rep_thr                   ))))
    return CWF


def compare_cwf_blr(cwf, pmtblr, event_list, window_size=500):
    """Return differences between the area of the CWF and the BLR in a
    given window, for a list of events, expressed as a percentage.
    """
    DIFF = []
    for event in event_list:
        CWF = cwf[event]
        BLR = pmtblr[event]

        for i in range(len(BLR)):
            t0, t1 = define_window(BLR[i], window_size)
            diff = abs(np.sum(BLR[i][t0:t1]) - np.sum(CWF[i][t0:t1]))
            diff = 100. * diff / np.sum(BLR)
            DIFF.append(diff)

    return np.array(DIFF)


def median_std_method(wfs    : np.ndarray,  
                      nsigma : Optional[float] = 3.) -> np.ndarray:
    """
    Computes the median and standard deviation of the time summed SiPM 
    waveforms and selects the SiPMs that are nsigma over the median.

    Parameters
    ----------
    wfs    : 2D array of shape (n_sipms, n_time_bins) containing the waveforms of each SiPM.
    nsigma : Number of standard deviations above the median, default 3.
    
    Returns
    -------
    Boolean numpy array of shape (n_sipms,) where True indicates that the SiPM is selected.
    """
    charges = np.sum(wfs, axis=1)
    threshold = np.median(charges) + nsigma * np.std(charges)
    return charges >= threshold


def charge_threshold_method(wfs       : np.ndarray,
                            threshold : Optional[float] = 5.) -> np.ndarray:
    """
    Selects the SiPMs whose time summed waveforms are above a threshold.
    
    Parameters
    ----------
    wfs       : 2D array of shape (n_sipms, n_time_bins) containing the waveforms of each SiPM.
    threshold : Charge threshold in PE, default 5.
    
    Returns
    -------
    Boolean numpy array of shape (n_sipms,) where True indicates that the SiPM is selected.
    """
    charges = np.sum(wfs, axis=1)
    return charges >= threshold


def top_n_method(wfs : np.ndarray, 
                 n   : Optional[int] = 10) -> np.ndarray:
    """
    Selects the SiPMs with the top n highest time summed waveforms.
        
    Parameters
    ----------
    wfs       : 2D array of shape (n_sipms, n_time_bins) containing the waveforms of each SiPM.
    n         : Number of most energeticSiPMs to select, default 10.
    
    Returns
    -------
    Boolean numpy array of shape (n_sipms,) where True indicates that the SiPM is selected.
    """
    charges = np.sum(wfs, axis=1)
    idx = np.argsort(charges)[-n:]

    selected_ids = np.zeros_like(charges, dtype=bool)
    selected_ids[idx] = True
    return selected_ids


def kill_isolated_sipms(selected_ids        : np.ndarray,
                        sipm_x              : np.ndarray, 
                        sipm_y              : np.ndarray, 
                        proximity_threshold : float) -> np.ndarray:
    """
    For the SiPMs that have passed the previous selection, scans through the SiPMs to check if they 
    have neighbouring SiPMs - i.e., within the proximity_threshold - that have also passed the selection. 
    If no neighbours are found, the SiPMs are classed as isolated, and are removed.

    Parameters
    ----------
    selected_ids        : Boolean array of shape (n_sipms,) indicating which SiPMs passed the previous selection.
    sipm_x              : 1D array of shape (n_sipms,) containing the x positions of the SiPMs.
    sipm_y              : 1D array of shape (n_sipms,) containing the y positions of the SiPMs.
    proximity_threshold : Distance threshold in mm used to identify isolated SiPMs.

    Returns
    -------
    selected_ids_no_isolated : Boolean array of shape (n_sipms,) where True indicates that the SiPM is selected.
    """
    selected_ids_no_isolated = selected_ids.copy()

    for i in np.where(selected_ids)[0]:
        x, y = sipm_x[i], sipm_y[i]

        distances = np.sqrt((sipm_x - x)**2 + (sipm_y - y)**2)

        n_neighbors = np.sum((distances < proximity_threshold) & selected_ids)

        if n_neighbors <= 1:
            selected_ids_no_isolated[i] = False
            
    return selected_ids_no_isolated