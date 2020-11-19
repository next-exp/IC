"""
-----------------------------------------------------------------------
                                Buffy
-----------------------------------------------------------------------

 -- Sorts MC sensor info into buffers, slays vampires etc

This city reads nexus Monte Carlo Sensor information and sorts the
information into data-like buffers without the addition of
electronics noise.

Uses a configured buffer length, pretrigger and threshold for the
positioning of trigger like signal. If more than one trigger is found
separated from each other by more than a buffer width the nexus event
can be split into multiple data-like triggers.
"""

import pandas as pd
import tables as tb

import warnings

from typing import Callable
from typing import     List

from .. core                   import        system_of_units as units
from .. detsim.sensor_utils    import   first_and_last_times
from .. detsim.sensor_utils    import          get_n_sensors
from .. detsim.sensor_utils    import           sensor_order
from .. detsim.sensor_utils    import pmt_and_sipm_bin_width
from .. io    .event_filter_io import    event_filter_writer
from .. reco                   import          tbl_functions as tbl

from .. dataflow   import                   dataflow as fl

from .  components import                       city
from .  components import                    collect
from .  components import               copy_mc_info
from .  components import calculate_and_save_buffers
from .  components import        mcsensors_from_file
from .  components import                print_every
from .  components import                  wf_binner


@city
def buffy(files_in     , file_out   , compression      , event_range,
          print_mod    , detector_db, run_number       , max_time   ,
          buffer_length, pre_trigger, trigger_threshold):

    npmt, nsipm       = get_n_sensors(detector_db, run_number)
    pmt_wid, sipm_wid = pmt_and_sipm_bin_width_safe_(files_in)
    nsamp_pmt         = int(buffer_length /  pmt_wid)
    nsamp_sipm        = int(buffer_length / sipm_wid)

    extract_tminmax   = fl.map(first_and_last_times_(pmt_wid, sipm_wid),
                               args = ("pmt_resp", "sipm_resp")        ,
                               out  = ("min_time",  "max_time")        )

    bin_calculation   = wf_binner(max_time)
    bin_pmt_wf        = fl.map(binning_set_width(bin_calculation, pmt_wid),
                               args = ("pmt_resp", "min_time", "max_time"),
                               out  = ("pmt_bins", "pmt_bin_wfs")         )

    bin_sipm_wf       = fl.map(binning_set_width(bin_calculation, sipm_wid),
                               args = ("sipm_resp", "min_time", "max_time"),
                               out  = ("sipm_bins", "sipm_bin_wfs")        )

    order_sensors     = fl.map(sensor_order(detector_db, run_number,
                                            nsamp_pmt  , nsamp_sipm) ,
                               args = ("pmt_bin_wfs", "sipm_bin_wfs",
                                       "buffers")                    ,
                               out  = "ordered_buffers"              )

    filter_events     = fl.map(lambda x, y : not any([x.empty, y.empty]),
                               args = ('pmt_resp' , 'sipm_resp')        ,
                               out  = 'event_passed'                    )
    events_with_resp  = fl.count_filter(bool, args="event_passed")

    with tb.open_file(file_out, "w", filters=tbl.filters(compression)) as h5out:
        buffer_calculation = calculate_and_save_buffers(buffer_length    ,
                                                        pre_trigger      ,
                                                        pmt_wid          ,
                                                        sipm_wid         ,
                                                        trigger_threshold,
                                                        h5out            ,
                                                        run_number       ,
                                                        npmt             ,
                                                        nsipm            ,
                                                        nsamp_pmt        ,
                                                        nsamp_sipm       ,
                                                        order_sensors    )

        write_filter   = fl.sink(event_filter_writer(h5out, "detected_events"),
                                 args=("event_number", "event_passed")       )

        event_count_in = fl.spy_count()

        evtnum_collect = collect()

        result = fl.push(source = mcsensors_from_file(files_in   ,
                                                      detector_db,
                                                      run_number )           ,
                         pipe   = fl.pipe(fl.slice(*event_range  ,
                                                   close_all=True)      ,
                                          event_count_in.spy            ,
                                          print_every(print_mod)        ,
                                          filter_events                 ,
                                          fl.branch(write_filter)       ,
                                          events_with_resp.filter       ,
                                          extract_tminmax               ,
                                          bin_pmt_wf                    ,
                                          bin_sipm_wf                   ,
                                          fl.branch("event_number"      ,
                                                    evtnum_collect.sink),
                                          buffer_calculation            )   ,
                         result = dict(events_in   = event_count_in.future  ,
                                       events_resp = events_with_resp.future,
                                       evtnum_list = evtnum_collect.future  ))

        copy_mc_info(files_in, h5out, result.evtnum_list,
                     detector_db, run_number)

        return result


def first_and_last_times_(pmt_binwid: float, sipm_binwid: float):
    def get_event_time_extremes(pmt_resp : pd.DataFrame,
                                sipm_resp: pd.DataFrame):
        return first_and_last_times(pmt_resp  , sipm_resp  ,
                                    pmt_binwid, sipm_binwid)
    return get_event_time_extremes


def binning_set_width(binning_fnc: Callable, bin_width: float):
    def bin_calculation_(sensors: pd.DataFrame,
                         t_min  : float       ,
                         t_max  : float       ):
        return binning_fnc(sensors, bin_width, t_min, t_max)
    return bin_calculation_


def pmt_and_sipm_bin_width_safe_(files_in: List[str]):
    for fn in files_in:
        try:
            pmt_wid, sipm_wid = pmt_and_sipm_bin_width(fn)
            return pmt_wid, sipm_wid
        except (tb.HDF5ExtError, tb.NoSuchNodeError) as e:
            warnings.warn(f' no useful bin widths: {0}'.format(e), UserWarning)
            continue
    return 25 * units.ns, 1 * units.mus
