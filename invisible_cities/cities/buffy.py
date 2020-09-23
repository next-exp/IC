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

import numpy  as np
import tables as tb

from .. detsim.buffer_functions import      buffer_calculator
from .. detsim.sensor_utils     import   first_and_last_times
from .. detsim.sensor_utils     import          get_n_sensors
from .. detsim.sensor_utils     import           sensor_order
from .. detsim.sensor_utils     import pmt_and_sipm_bin_width
from .. detsim.sensor_utils     import          trigger_times
from .. io    .rwf_io           import          buffer_writer
from .. reco                    import          tbl_functions as tbl

from .. dataflow   import            dataflow as fl

from .  components import                city
from .  components import             collect
from .  components import        copy_mc_info
from .  components import mcsensors_from_file
from .  components import         print_every
from .  components import       signal_finder
from .  components import           wf_binner


@city
def buffy(files_in     , file_out   , compression      , event_range,
          print_mod    , detector_db, run_number       , max_time   ,
          buffer_length, pre_trigger, trigger_threshold):

    npmt, nsipm       = get_n_sensors(detector_db, run_number)
    pmt_wid, sipm_wid = pmt_and_sipm_bin_width(files_in[0])
    nsamp_pmt         = int(buffer_length /  pmt_wid)
    nsamp_sipm        = int(buffer_length / sipm_wid)

    extract_tminmax   = fl.map(first_and_last_times                ,
                               args = ("pmt_resp"   ,  "sipm_resp",
                                       "pmt_binwid", "sipm_binwid"),
                               out  = ("min_time", "max_time"))

    bin_calculation   = wf_binner(max_time)
    bin_pmt_wf        = fl.map(bin_calculation                   ,
                               args = ("pmt_resp",  "pmt_binwid",
                                       "min_time",    "max_time"),
                               out  = ("pmt_bins", "pmt_bin_wfs"))

    bin_sipm_wf       = fl.map(bin_calculation                    ,
                               args = ("sipm_resp", "sipm_binwid",
                                       "min_time" ,    "max_time"),
                               out  = ("sipm_bins", "sipm_bin_wfs"))

    find_signal       = fl.map(signal_finder(buffer_length, pmt_wid,
                                             trigger_threshold     ),
                               args = "pmt_bin_wfs"                 ,
                               out  = "pulses"                      )

    event_times       = fl.map(trigger_times                             ,
                               args = ("pulses", "timestamp", "pmt_bins"),
                               out  = "evt_times"                        )

    calculate_buffers = fl.map(buffer_calculator(buffer_length, pre_trigger,
                                                  pmt_wid      ,    sipm_wid),
                                args = ("pulses",
                                        "pmt_bins" ,  "pmt_bin_wfs",
                                        "sipm_bins", "sipm_bin_wfs"),
                                out  = "buffers")

    order_sensors     = fl.map(sensor_order(detector_db, run_number,
                                            nsamp_pmt  , nsamp_sipm) ,
                               args = ("pmt_bin_wfs", "sipm_bin_wfs",
                                       "buffers")                    ,
                               out  = "ordered_buffers"              )

    with tb.open_file(file_out, "w", filters=tbl.filters(compression)) as h5out:
        buffer_writer_ = fl.sink(buffer_writer(h5out                  ,
                                               run_number = run_number,
                                               n_sens_eng = npmt      ,
                                               n_sens_trk = nsipm     ,
                                               length_eng = nsamp_pmt ,
                                               length_trk = nsamp_sipm),
                                 args = ("event_number", "evt_times"  ,
                                         "ordered_buffers"            ))

        event_count_in = fl.spy_count()

        evtnum_collect = collect()

        result = fl.push(source = mcsensors_from_file(files_in   ,
                                                      detector_db,
                                                      run_number )         ,
                         pipe   = fl.pipe(fl.slice(*event_range  ,
                                                   close_all=True)      ,
                                          event_count_in.spy            ,
                                          print_every(print_mod)        ,
                                          extract_tminmax               ,
                                          bin_pmt_wf                    ,
                                          bin_sipm_wf                   ,
                                          find_signal                   ,
                                          event_times                   ,
                                          calculate_buffers             ,
                                          order_sensors                 ,
                                          fl.branch("event_number"      ,
                                                    evtnum_collect.sink),
                                          buffer_writer_                )  ,
                         result = dict(events_in   = event_count_in.future,
                                       evtnum_list = evtnum_collect.future))

        copy_mc_info(files_in, h5out, result.evtnum_list,
                     detector_db, run_number)

        return result
