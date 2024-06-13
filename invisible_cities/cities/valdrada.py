"""
-----------------------------------------------------------------------
                                 valdrada
-----------------------------------------------------------------------

This city finds simulates the trigger procedure at the FPGA over PMT waveforms.
This includes a number of tasks:
    - Use a moving average to obtain the baseline
    - Truncate to integers the deconvolution output
    - Find trigger candidates and compute their characteristics, taking into
      account the unique circumstances derived from working on a continuous scheme
      (for example, delay in signals)
    - Evaluate coincidences between trigger candidates.

A number of general configuration parameters are needed (input as a dict, trigger_config):
- coincidence_window : Time window (in time bins) in which valid trigger coincidences are counted
- discard_width      : Any trigger with width less than this parameter is discarded.
- multipeak          : Dictionary with multipeak protection parameters, otherwise None.
     - q_min      : Integrated charge threshold of a post-trigger peak to discard
                    a trigger due to multipeak protection. In ADC counts.
     - time_min   : Minimum width of a post-trigger peak to discard a trigger due to
                    multipeak protection. In time bins.
     - time_after : For how long is the multipeak protection evaluated after a
                    valid trigger. In mus.

A individual trigger configuration can be given per channel, through a dict
with keys equal to PMT IDs, which marks the validity range of the peak
characteristics:
- q_min, q_max          : Range for the integrated charge of the peak (q_min < q < q_max).
                          In ADC counts.
- time_min, time_max    : Range for the peak width (time_min <= width < time_max).
                          In time mus.
- baseline_dev, amp_max : Range for peak height (baseline_dev < height < amp_max).
                          In ADC counts.
- pulse_valid_ext       : Time allowed for a pulse to go below baseline_dev. In ns.

The result of the city is a dataframe containing the event ID and PMT ID of each
trigger candidate. For each trigger candidate a number of parameters is computed:
 - trigger_time : time bin at which the trigger candidate starts.
 - q            : integrated ADC counts within the peak.
 - width        : Length of the peak in time bins.
 - height       : Maximum ADC values in a given time bien within the peak.
 - baseline     : Value of the baseline at trigger_time.
 - max_height   : Maximum (minimum due to PMT negative signals) height in the wvf.

Additionally, a set of flags are assigned depending on wether the parameters are
within range of the trigger configuration:
 - valid_q    : If peak is within q range.
 - valid_w    : If peak is within width range.
 - valid_h    : If peak is within height range.
 - valid_peak : Only if 'multipeak' is active, True if there isn't a post-trigger
               candidate with the configuration parameters.
 - valid_all  : boolean and of previous valid flags.

 Finally, a series of coincidence-related values are also given:
 - n_coinc       : Number of valid triggers within the coincidence_window,
                   starting at the trigger trigger_time and including the trigger itself.
                   -1 indicates no valid trigger (including the trigger itself).
 - closest_ttime : Time difference to closest valid trigger.
                   -1 if there are none aside from the trigger itself.
 - closest_pmt   : PMT ID of the closest valid trigger.
                   -1 if there are none aside from the trigger itself.
"""

import tables as tb
import numpy  as np

from .. reco                  import tbl_functions        as tbl
from .. io  .run_and_event_io import run_and_event_writer
from .. io  .trigger_io       import       trigger_writer, trigger_dst_writer

from .. dataflow            import dataflow as fl
from .. dataflow.dataflow   import push
from .. dataflow.dataflow   import pipe
from .. dataflow.dataflow   import sink

from .  components import city
from .  components import print_every
from .  components import collect
from .  components import copy_mc_info
from .  components import deconv_pmt_fpga
from .  components import WfType
from .  components import wf_from_files
from .  components import get_number_of_active_pmts

from .. reco.trigger_functions import retrieve_trigger_information
from .. reco.trigger_functions import get_trigger_candidates
from .. reco.trigger_functions import check_trigger_coincidence

from .. core       import system_of_units as units


def check_empty_trigger(triggers) -> bool:
    """
    Filter for valdrada flow.
    The flow stops if there are no candidate triggers/
    """
    return len(triggers) > 0

@city
def valdrada(files_in, file_out, compression, event_range, print_mod, detector_db, run_number,
             trigger_config = dict(), channel_config = dict()):

    # Change time units into number of waveform bins.
    if trigger_config['multipeak'] is not None:
        trigger_config['multipeak']['time_min'  ] /= 25*units.ns
        trigger_config['multipeak']['time_after'] /= 25*units.ns
    for k in channel_config.keys():
        channel_config[k]['time_min']       /= 25*units.ns
        channel_config[k]['time_max']       /= 25*units.ns
        channel_config[k]['pulse_valid_ext'] = int(channel_config[k]['pulse_valid_ext']/25*units.ns)

    #### Define data transformations
    # Raw WaveForm to deconvolved waveforms
    rwf_to_cwf           = fl.map(deconv_pmt_fpga(detector_db, run_number, list(channel_config.keys())),
                                  args = "pmt",
                                  out  = ("cwf", "baseline"))

    # Extract all possible trigger candidates of each PMT
    trigger_on_channels  = fl.map(retrieve_trigger_information(channel_config, trigger_config),
                                  args = ("pmt", "cwf", "baseline", "event_number"),
                                  out  = "triggers")

    # Add coincidence between channels
    check_coincidences   = fl.map(check_trigger_coincidence(trigger_config['coincidence_window']),
                                  item = "triggers")

    # Filter events with zero triggers
    filter_empty_trigger = fl.map(check_empty_trigger,
                                  args = "triggers",
                                  out  = "empty_trigger")

    event_count_in     = fl.spy_count()
    event_count_out    = fl.spy_count()
    events_no_triggers = fl.count_filter(bool, args = "empty_trigger")
    evtnum_collect     = collect()

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:
        # Define writers...
        write_event_info_   = run_and_event_writer(h5out)
        write_trigger_info_ = trigger_writer      (h5out, get_number_of_active_pmts(detector_db, run_number))
        write_trigger_dst_  = trigger_dst_writer  (h5out)
        # ... and make them sinks

        write_event_info   = sink(write_event_info_  , args=( "run_number"  , "event_number"    , "timestamp"   ))
        write_trigger_info = sink(write_trigger_info_, args=( "trigger_type", "trigger_channels"                ))
        write_trigger_dst  = sink(write_trigger_dst_ , args=  "triggers"                                         )

        result = push(source = wf_from_files(files_in, WfType.rwf),
                      pipe   = pipe(fl.slice(*event_range, close_all=True),
                                    print_every(print_mod),
                                    event_count_in.spy,
                                    rwf_to_cwf,
                                    trigger_on_channels,
                                    filter_empty_trigger,
                                    events_no_triggers.filter,
                                    check_coincidences,
                                    event_count_out.spy,
                                    fl.branch("event_number", evtnum_collect.sink),
                                    fl.fork(write_trigger_dst ,
                                            write_event_info  ,
                                            write_trigger_info)),
                      result = dict(events_in   = event_count_in    .future,
                                    events_out  = event_count_out   .future,
                                    evtnum_list = evtnum_collect    .future,
                                    events_pass = events_no_triggers.future))

        if run_number <= 0:
            copy_mc_info(files_in, h5out, result.evtnum_list,
                         detector_db, run_number)

        return result
