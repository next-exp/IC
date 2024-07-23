"""
-----------------------------------------------------------------------
                                Diomira
-----------------------------------------------------------------------

From Germanic, Teodomiro/Teodemaro: famous among its people.

This city simulates the response of the different sensors within the
detector, namely, PMTs and SiPMs. This can be summarized in the
following tasks:
    - Add the sensor baseline.
    - Simulate the noise.
    - Simulate gain fluctuations.
    - Convert (true) photoelectrons to ADC counts.
Besides, and only for PMTs:
    - Emulate the signal-derivative effect of the energy plane
      electronics.
    - Rebin 1-ns waveforms to 25-ns waveforms to match those produced
      by the detector.

On top of that, the city can emulate the trigger algorithm of the
detector. At the present time, only a S2 trigger is implemented, which
processes the data in the same way as the detector and applies the same
filters according to some predefined parameters.
"""

from functools import partial
from itertools import compress

import numpy  as np
import tables as tb

from .. core.configure          import EventRangeType
from .. core.configure          import OneOrManyFiles
from .. core                    import    tbl_functions as tbl
from .. reco                    import   peak_functions as pkf
from .. sierpe                  import fee              as FE
from .. io.rwf_io               import           rwf_writer
from .. io.run_and_event_io     import run_and_event_writer
from .. io. event_filter_io     import  event_filter_writer
from .. filters.trigger_filters import TriggerFilter
from .. database                import load_db
from .. detsim                  import sensor_functions as sf
from .. evm.ic_containers       import TriggerParams
from .. evm.pmaps               import S2
from .. types.ic_types          import minmax
from .. types.ic_types          import NoneType
from .. types.symbols           import WfType

from .. dataflow import dataflow as fl

from .  components import city
from .  components import print_every
from .  components import collect
from .  components import copy_mc_info
from .  components import sensor_data
from .  components import deconv_pmt
from .  components import wf_from_files
from .  components import simulate_sipm_response
from .  components import compute_pe_resolution

from typing import Union
from typing import Optional


@city
def diomira( files_in       : OneOrManyFiles
           , file_out       : str
           , compression    : str
           , event_range    : EventRangeType
           , print_mod      : int
           , detector_db    : str
           , run_number     : int
           , sipm_noise_cut : float
           , filter_padding : int
           , trigger_type   : Union[NoneType, str]
           , trigger_params : Optional[dict]                 = dict()
           , s2_params      : Optional[dict]                 = dict()
           , random_seed    : Optional[Union[NoneType, int]] = None
           ):
    if random_seed is not None:
        np.random.seed(random_seed)

    sd = sensor_data(files_in[0], WfType.mcrd)

    simulate_pmt_response_  = fl.map(simulate_pmt_response (detector_db,
                                                            run_number ),
                                     args="pmt" , out= ("pmt_sim", "blr_sim"))
    simulate_sipm_response_ = fl.map(simulate_sipm_response(detector_db   ,
                                                            run_number    ,
                                                            sd.SIPMWL     ,
                                                            sipm_noise_cut,
                                                            filter_padding),
                                     args="sipm", out="sipm_sim"             )
    trigger_filter_         = select_trigger_filter(trigger_type  ,
                                                    trigger_params,
                                                    s2_params     )
    emulate_trigger_        = fl.map(emulate_trigger(detector_db   ,
                                                     run_number    ,
                                                     trigger_type  ,
                                                     trigger_params,
                                                     s2_params     ),
                                     args="pmt_sim", out="trigger_sim"       )
    trigger_pass            = fl.map(trigger_filter_   ,
                                     args="trigger_sim",
                                     out="trigger_pass")
    trigger_filter          = fl.count_filter(bool, args="trigger_pass")

    with tb.open_file(file_out, "w", filters=tbl.filters(compression)) as h5out:
        RWF        = partial(rwf_writer, h5out, group_name='RD')
        write_pmt  = fl.sink(RWF(table_name      = 'pmtrwf',
                                 n_sensors       = sd.NPMT ,
                                 waveform_length = sd.PMTWL // int(FE.t_sample)),
                                 args= "pmt_sim")
        write_blr  = fl.sink(RWF(table_name      = 'pmtblr',
                                 n_sensors       = sd.NPMT ,
                                 waveform_length = sd.PMTWL // int(FE.t_sample)),
                                 args= "blr_sim")
        write_sipm = fl.sink(RWF(table_name      = 'sipmrwf',
                                 n_sensors       = sd.NSIPM ,
                                 waveform_length = sd.SIPMWL),
                                 args="sipm_sim")

        write_event_info_ = run_and_event_writer(h5out)
        write_evt_filter_ = event_filter_writer (h5out, "trigger")

        write_event_info = fl.sink(write_event_info_,
                                   args=("run_number", "event_number",
                                         "timestamp"                 ))
        write_evt_filter = fl.sink(write_evt_filter_,
                                   args=("event_number", "trigger_pass"))

        event_count_in = fl.spy_count()

        evtnum_collect = collect()

        result = fl.push(source = wf_from_files(files_in, WfType.mcrd),
                         pipe   = fl.pipe(fl.slice(*event_range          ,
                                                   close_all=True)      ,
                                          event_count_in.spy            ,
                                          print_every(print_mod)        ,
                                          simulate_pmt_response_        ,
                                          emulate_trigger_              ,
                                          trigger_pass                  ,
                                          fl.branch(write_evt_filter)   ,
                                          trigger_filter.filter         ,
                                          simulate_sipm_response_       ,
                                          fl.branch("event_number"     ,
                                                    evtnum_collect.sink),
                                          fl.fork(write_pmt       ,
                                                  write_blr       ,
                                                  write_sipm      ,
                                                  write_event_info))     ,
                         result = dict(events_in     = event_count_in.future,
                                       evtnum_list   = evtnum_collect.future,
                                       events_filter = trigger_filter.future))

        if run_number <= 0:
            copy_mc_info(files_in, h5out, result.evtnum_list,
                         detector_db, run_number)

        return result


def simulate_pmt_response(detector, run_number):
    datapmt       = load_db.DataPMT(detector, run_number)
    adc_to_pes    = np.abs(datapmt.adc_to_pes.values).astype(np.double)
    single_pe_rms = datapmt.Sigma.values.astype(np.double)
    pe_resolution = compute_pe_resolution(single_pe_rms, adc_to_pes)

    def simulate_pmt_response(pmtrd):
        rwf, blr = sf.simulate_pmt_response(0, pmtrd[np.newaxis],
                                            adc_to_pes, pe_resolution,
                                            detector, run_number)
        return np.round(rwf).astype(np.int16), np.round(blr).astype(np.int16)
    return simulate_pmt_response



def select_trigger_filter(trigger_type, trigger_params, s2_params):
    if   trigger_type is None:
        def always_pass(*args):
            return True
        return always_pass

    elif trigger_type == "S2":
        chann  =              trigger_params["tr_channels"]
        nchann =              trigger_params["min_number_channels"]
        ratio  =              trigger_params["data_mc_ratio"]
        height = minmax(min = trigger_params["min_height"],
                        max = trigger_params["max_height"])
        charge = minmax(min = trigger_params["min_charge"],
                        max = trigger_params["max_charge"])
        width  = minmax(min = trigger_params["min_width" ],
                        max = trigger_params["max_width" ])
        params = TriggerParams(trigger_channels    =  chann,
                               min_number_channels = nchann,
                               charge              = charge * ratio,
                               height              = height * ratio,
                               width               = width)
        return TriggerFilter(params)

    else:
        raise ValueError(f"Invalid trigger type: {repr(trigger_type)}")


def emulate_trigger(detector_db, run_number, trigger_type, trigger_params, s2_params):
    if   trigger_type is None:
        def do_nothing(*args, **kwargs):
            pass
        return do_nothing
    elif trigger_type == "S2":
        channels   = trigger_params["tr_channels"]
        min_height = trigger_params["min_height"]
        datapmt    = load_db.DataPMT(detector_db, run_number)
        IC_ids     = sf.convert_channel_id_to_IC_id(datapmt, channels).tolist()
        n_baseline = s2_params.pop("n_baseline")
        s2_params  = dict(time         = minmax(min = s2_params["s2_tmin"        ],
                                                max = s2_params["s2_tmax"        ]),
                          stride                    = s2_params["s2_stride"      ] ,
                          length       = minmax(min = s2_params["s2_lmin"        ],
                                                max = s2_params["s2_lmax"        ]),
                          rebin_stride =              s2_params["s2_rebin_stride"])

        for channel in channels:
            if channel not in compress(datapmt.ChannelID, datapmt.Active.values):
                raise ValueError("Cannot trigger on a masked PMT")

        deconvolver = deconv_pmt(detector_db, run_number, n_baseline, IC_ids)

        def get_indices(cwf):
            return pkf.indices_and_wf_above_threshold(cwf, thr=min_height)[0]

        def find_peaks(cwf, idx):
            return pkf.find_peaks(cwf, idx, Pk=S2, pmt_ids=[-1], **s2_params)

        def emulate_trigger(rwfs):
            cwfs = deconvolver(rwfs)
            peak_data = dict()
            for ID, cwf in zip(IC_ids, cwfs):
                idx = get_indices(cwf)
                s2  = find_peaks (cwf, idx)
                peak_data[ID] = s2
            return peak_data

        return emulate_trigger

    else:
        raise ValueError(f"Invalid trigger type: {repr(trigger_type)}")
