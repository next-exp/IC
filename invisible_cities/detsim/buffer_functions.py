import numpy  as np
import pandas as pd

from typing import Callable
from typing import     List
from typing import    Tuple
from typing import    Union

from .. reco.peak_functions import indices_and_wf_above_threshold
from .. reco.peak_functions import                 split_in_peaks


def weighted_histogram(data: pd.DataFrame, bins: np.ndarray) -> np.ndarray:
    return np.histogram(data.time, weights=data.charge, bins=bins)[0]


def bin_sensors(sensors   : pd.DataFrame,
                bin_width : float       ,
                t_min     : float       ,
                t_max     : float       ,
                max_buffer: int         ) -> Tuple[np.ndarray, pd.Series]:
    """
    Raw data binning function.

    Parameters
    ----------
    sensors : List of Waveforms for one sensor type
    Should be sorted into one type/binning
    t_min   : float
              Minimum time to be used to define bins.
    t_max   : float
              As t_min but the maximum to be used
    """
    max_time    = min(t_max, t_min + max_buffer)
    min_bin     = np.floor(t_min    / bin_width) * bin_width
    max_bin     = np.ceil (max_time / bin_width) * bin_width

    bins        = np.arange(min_bin, max_bin + bin_width, bin_width)
    bin_sensors = sensors.groupby('sensor_id').apply(weighted_histogram, bins)
    return bins[:-1], bin_sensors


## !! to-do: clarify for non-pmt versions of next
## !! to-do: Check on integral instead of only threshold?
def find_signal_start(wfs          : Union[pd.Series, np.ndarray],
                      bin_threshold: float                       ,
                      stand_off    : int                         ) -> List[int]:
    """
    Finds signal in the binned waveforms and
    identifies candidate triggers.
    """
    if isinstance(wfs, np.ndarray):
        eng_sum = wfs.sum(axis=0)
    else:
        eng_sum = wfs.sum()
    indices = indices_and_wf_above_threshold(eng_sum,
                                             bin_threshold).indices
    ## Just using this and the stand_off for now
    ## taking first above sum threshold.
    ## !! To-do: make more robust with min int? or similar
    all_indx = split_in_peaks(indices, stand_off)
    return [pulse[0] for pulse in all_indx]


def pad_safe(sensors: np.ndarray, padding: Tuple) -> np.ndarray:
    """Pads zeros around each sensor in a 2D array"""
    if not sensors.shape[0]:
        return np.empty((0, padding[0] + padding[1] + 1))
    return np.apply_along_axis(np.pad, 1, sensors, padding, "constant")


def buffer_calculator(buffer_len: float, pre_trigger: float,
                      pmt_binwid: float, sipm_binwid: float) -> Callable:
    """
    Calculates the output buffers for all sensors
    based on a configured buffer length and pretrigger.
    Synchronising the clock between the two sensor types.

    Parameters
    ----------
    buffer_len  : float
                  Length of buffer expected in mus
    pre_trigger : float
                  Time in buffer before identified signal in mus
    pmt_binwid  : float
                  Width in mus of PMT sample integration
    sipm_binwid : float
                  Width in mus of SiPM sample integration
    """
    pmt_buffer_samples  = int(buffer_len  //  pmt_binwid)
    sipm_buffer_samples = int(buffer_len  // sipm_binwid)
    sipm_pretrg         = int(pre_trigger // sipm_binwid)
    sipm_postrg         = sipm_buffer_samples - sipm_pretrg
    pmt_pretrg_base     = int(pre_trigger //  pmt_binwid)
    pmt_postrg_base     = pmt_buffer_samples - pmt_pretrg_base

    def generate_slice(trigger    : int       ,
                       pmt_bins   : np.ndarray,
                       pmt_charge : np.ndarray,
                       sipm_bins  : np.ndarray,
                       sipm_charge: np.ndarray) -> Tuple:
        """
        Synchronises the clocks between the SiPMs and PMTs
        and slices the histograms where this synchronisation
        indicates.
        """
        npmt_bin  = len(pmt_bins)
        nsipm_bin = len(sipm_bins)

        trg_bin    = np.where(sipm_bins <= pmt_bins[trigger])[0][-1]
        bin_corr   = (pmt_bins[trigger] - sipm_bins[trg_bin]) // pmt_binwid
        pmt_pretrg = pmt_pretrg_base + int(bin_corr)
        pmt_postrg = pmt_postrg_base - int(bin_corr)

        pmt_pre    = 0       , trigger - pmt_pretrg
        pmt_pos    = npmt_bin, trigger + pmt_postrg
        pmt_slice  = slice(max(pmt_pre), min(pmt_pos))
        pmt_pad    = -min(pmt_pre), max(0, pmt_pos[1] - npmt_bin)

        sipm_pre   = 0        , trg_bin - sipm_pretrg
        sipm_pos   = nsipm_bin, trg_bin + sipm_postrg
        sipm_slice = slice(max(sipm_pre), min(sipm_pos))
        sipm_pad   = -min(sipm_pre), max(0, sipm_pos[1] - nsipm_bin)

        return (pad_safe( pmt_charge[:,  pmt_slice],  pmt_pad),
                pad_safe(sipm_charge[:, sipm_slice], sipm_pad))


    def position_signal(triggers   :       List                  ,
                        pmt_bins   : np.ndarray                  ,
                        pmt_charge : Union[pd.Series, np.ndarray],
                        sipm_bins  : np.ndarray                  ,
                        sipm_charge: Union[pd.Series, np.ndarray]
                        ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Synchronises the SiPMs and PMTs for each identified
        trigger and calls the padding function to fill with
        zeros where necessary.
        """
        if isinstance(pmt_charge, pd.Series):
            pmt_q  = np.asarray(pmt_charge.tolist())
            sipm_q = np.empty((0,0))\
            if sipm_charge.empty else np.asarray(sipm_charge.tolist())
        else:
            pmt_q  =  pmt_charge
            sipm_q = sipm_charge
        return [generate_slice(trigger, pmt_bins, pmt_q, sipm_bins, sipm_q)
                for trigger in triggers                                    ]
    return position_signal
