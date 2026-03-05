"""Waveform Functions
This module includes functions to manipulate waveforms.
authors: J.J. Gomez-Cadenas, G. Martinez
"""
import numpy as np
from typing  import Optional
from typing  import Callable
from typing  import Tuple

from .. core.core_functions import define_window, to_col_vector
from .. calib               import calib_sensors_functions as csf
from .. sierpe              import blr
from .. database            import load_db
from .. reco.peak_functions import select_wf_slices_above_time_integrated_thr

from .. types  .symbols     import SiPMSelectionMethod

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


def charge_threshold_method(wfs             : np.ndarray,
                            indices         : np.ndarray,
                            zeroing_thr     : Optional[float] = 2.,
                            integration_thr : Optional[float] = 5.) -> Tuple[np.ndarray, np.ndarray]:
    """
    Selects the SiPMs whose time summed waveforms within the s2 windows are above two thresholds:
        - initial zero suprresion threshold (setting values in each waveform below a value to 0)
        - threshold over each selected integrated slice of the waveforms

    Parameters
    ----------
    wfs       : 2D array of shape (n_sipms, n_time_bins) containing the waveforms of each SiPM.
    threshold : Charge threshold in PE, default 5.

    Returns
    -------
    Tuple of np arrays including all passing sipm ids and the corresponding waveforms
    """
    thr = to_col_vector(np.full(wfs.shape[0], zeroing_thr))

    # zero entries below threshold
    zwfs = np.where(wfs > thr, wfs, 0)

    # returns selected ids and waveforms above integral
    return select_wf_slices_above_time_integrated_thr(zwfs, indices, integration_thr)


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


def apply_circular_padding(selected_ids_no_isolated : np.ndarray,
                           sipm_x                   : np.ndarray,
                           sipm_y                   : np.ndarray,
                           padding_radius           : float) -> np.ndarray:
    """
    For the SiPMs that pass the previous selection, creates circular padding of radius padding_radius,
    selecting all SiPMs within that radius. Stores the union of all selected SiPMs.

    Parameters
    ----------
    selected_ids_no_isolated : Boolean array of shape (n_sipms,) indicating which SiPMs passed the previous selection.
    sipm_x                   : 1D array of shape (n_sipms,) containing the x positions of the SiPMs.
    sipm_y                   : 1D array of shape (n_sipms,) containing the y positions of the SiPMs.
    padding_radius           : Distance threshold in mm used to create circular padding around selected SiPMs.

    Returns
    -------
    sipm_ids_with_signal : Boolean array of shape (n_sipms,) where True indicates that the SiPM is selected.
    """
    sipm_ids_with_signal = np.zeros_like(selected_ids_no_isolated, dtype=bool)

    for i in np.where(selected_ids_no_isolated)[0]:
        x, y = sipm_x[i], sipm_y[i]
        distances = np.sqrt((sipm_x - x)**2 + (sipm_y - y)**2)
        sipm_ids_with_signal |= distances < padding_radius

    return sipm_ids_with_signal


def spatial_selection_method(wfs                 : np.ndarray,
                             indices             : np.ndarray,
                             selection_method    : SiPMSelectionMethod,
                             selection_kwargs    : dict,
                             proximity_threshold : float,
                             padding_radius      : float,
                             run_number          : int,
                             detector_db         : str) -> np.ndarray:
    """
    SiPM selection function, applies SiPM cuts based on user input.
    A first selection of SiPMs is made, isolated SiPMs are removed
    and padding is added around the SiPMs that are left.

    Parameters
    ----------
    wfs                 : 2D array of shape (n_sipms, n_time_bins) containing the waveforms of each SiPM.
    selection_method    : Method used to select SiPMs.
    selection_kwargs    : Dictionary of arguments passed to the selection function.
    proximity_threshold : Threshold used to identify isolated SiPMs.
    padding_radius      : Radial padding added to each SiPM that passes the selections.
    run_number          : Run number used to load the detector database.
    detector_db         : Database used to load the detector geometry.

    Returns
    -------
    selected_ids : Array of shape (n_sipms,) containing the indices of the selected SiPMs.
    selected_wfs : 2D array of shape (n_selected_sipms, n_time_bins) with the waveforms of the selected SiPMs.
    """
    detector_info = load_db.DataSiPM(detector_db, run_number)
    sipm_x = np.array(detector_info.X)
    sipm_y = np.array(detector_info.Y)

    slice_ = slice(indices[0], indices[-1] + 1)
    wfs_   = wfs[:, slice_]

    if selection_method is SiPMSelectionMethod.median_std_method:
        starting_ids_ = median_std_method(wfs_, **selection_kwargs)
    elif selection_method is SiPMSelectionMethod.top_n_method:
        starting_ids_ = top_n_method(wfs_, **selection_kwargs)
    else:
        raise ValueError(f"Selection method {selection_method} not recognized.")

    selected_ids_no_isolated_ = kill_isolated_sipms(
        starting_ids_,
        sipm_x,
        sipm_y,
        proximity_threshold
    )

    sipm_ids_with_signal_ = apply_circular_padding(
        selected_ids_no_isolated_,
        sipm_x,
        sipm_y,
        padding_radius
    )

    selected_ids = np.where(sipm_ids_with_signal_)[0]
    selected_wfs = wfs[selected_ids]

    return selected_ids, selected_wfs

