import numpy  as np
import tables as tb
import pandas as pd

from typing import Callable
from typing import     List
from typing import    Tuple

from .. database.load_db   import            DataPMT
from .. database.load_db   import           DataSiPM
from .. io      .mcinfo_io import get_sensor_binning


def create_timestamp(event_number: int or float,
                     rate        : float       ) -> float:
    """
    Calculates timestamp for a given Event Number and Rate.

    :param event_number: Value of the current event.
    :param rate: Value of the rate.
    :return: Calculated timestamp
    """
    period = 1 / rate
    timestamp = event_number * period + np.random.uniform(0, period)
    return timestamp


def trigger_times(trigger_indx: List[int] ,
                  event_time  :      float,
                  time_bins   : np.ndarray) -> List[float]:
    """
    Calculates trigger time for all identified candidates
    according to the nexus event time and the time within
    the event where the trigger was identified.

    Parameters
    ----------
    trigger_indx : List[int]
                   The indices in the histogrammed nexus sensor
                   data where triggers were identified.
    event_time   : float
                   Time of nexus event
    time_bins    : np.ndarray
                   The binning used to histogram the nexus sensor
                   data.

    Returns
    -------
    triggered event times : List[float]
    """
    return event_time + time_bins[trigger_indx]


def first_and_last_times(pmt_wfs    : pd.DataFrame,
                         sipm_wfs   : pd.DataFrame,
                         pmt_binwid : float       ,
                         sipm_binwid: float       ) -> Tuple[float, float]:
    """
    Returns the maximum and minimum time of an
    event given the two types of detector.
    """
    min_time  = min(pmt_wfs.time.min(), sipm_wfs.time.min())
    max_time  = max(pmt_wfs.time.max(), sipm_wfs.time.max())
    max_time += min(pmt_binwid        ,         sipm_binwid)
    return min_time, max_time


def sensor_order(detector_db: str, run_number : int,
                 length_pmt : int, length_sipm: int) -> Callable:
    """
    Casts the event sensor info into the correct order
    adding zeros for sensors which didn't see any signal.
    """
    pmt_ids       = DataPMT (detector_db, run_number).SensorID
    sipm_ids      = DataSiPM(detector_db, run_number).SensorID
    n_pmt, n_sipm = get_n_sensors(detector_db, run_number)
    pmt_shape     = (n_pmt , length_pmt )
    sipm_shape    = (n_sipm, length_sipm)
    def ordering(sensor_order : pd.Int64Index  ,
                 sensor_resp  : np.ndarray     ,
                 sensor_shape : Tuple[int, int]) -> np.ndarray:
        sensors = np.zeros(sensor_shape, np.int)
        sensors[sensor_order] = sensor_resp
        return sensors
        
    def order_and_pad(pmt_resp    : pd.Series                          ,
                      sipm_resp   : pd.Series                          ,
                      evt_buffers : List[Tuple[np.ndarray, np.ndarray]]
                      ) -> List[Tuple]:
        pmt_ord  = pmt_ids [ pmt_ids.isin( pmt_resp.index)].index
        sipm_ord = sipm_ids[sipm_ids.isin(sipm_resp.index)].index

        return [(ordering(pmt_ord , pmts , pmt_shape ),
                 ordering(sipm_ord, sipms, sipm_shape))
                    for pmts, sipms in evt_buffers]
    return order_and_pad


def get_n_sensors(detector_db: str, run_number: int) -> Tuple[int, int]:
    """Get the number of sensors for this run"""
    npmt  = DataPMT (detector_db, run_number).shape[0]
    nsipm = DataSiPM(detector_db, run_number).shape[0]
    return npmt, nsipm


def pmt_and_sipm_bin_width(file_name: str) -> Tuple[float, float]:
    """
    returns pmt and sipm bin widths as set in nexus.
    Assumes Pmt + SiPM as in NEW, NEXT100 & DEMO.
    """
    sns_bins = get_sensor_binning(file_name)
    if sns_bins.empty or np.any(sns_bins.bin_width <= 0):
        raise tb.NoSuchNodeError('No useful binning info found')
    pmt_wid  = sns_bins.bin_width[sns_bins.index.str.contains( 'Pmt')].iloc[0]
    sipm_wid = sns_bins.bin_width[sns_bins.index.str.contains('SiPM')].iloc[0]
    return pmt_wid, sipm_wid
