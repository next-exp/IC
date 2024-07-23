"""
-----------------------------------------------------------------------
                                 Hypathia
-----------------------------------------------------------------------

From ancient Greek ‘Υπατια: highest, supreme.

This city reads true waveforms from detsim and compute pmaps from them
without simulating the electronics. This includes:
    - Rebin 1-ns waveforms to 25-ns waveforms to match those produced
      by the detector.
    - Produce a PMT-summed waveform.
    - Apply a threshold to the PMT-summed waveform.
    - Find pulses in the PMT-summed waveform.
    - Match the time window of the PMT pulse with those in the SiPMs.
    - Build the PMap object.
"""
import numpy  as np
import tables as tb

from functools import partial

from .. core.configure         import EventRangeType
from .. core.configure         import OneOrManyFiles
from .. core                   import tbl_functions        as tbl
from .. reco                   import peak_functions       as pkf
from .. core . random_sampling import NoiseSampler         as SiPMsNoiseSampler
from .. core                   import system_of_units      as units
from .. detsim                 import sensor_functions     as sf
from .. io   .run_and_event_io import run_and_event_writer
from .. io   .      trigger_io import       trigger_writer
from .. types.symbols          import WfType
from .. types.symbols          import SiPMThreshold

from .. dataflow            import dataflow as fl
from .. dataflow.dataflow   import push
from .. dataflow.dataflow   import pipe
from .. dataflow.dataflow   import sink

from .  components import city
from .  components import print_every
from .  components import collect
from .  components import copy_mc_info
from .  components import zero_suppress_wfs
from .  components import sensor_data
from .  components import wf_from_files
from .  components import get_number_of_active_pmts
from .  components import compute_and_write_pmaps
from .  components import simulate_sipm_response
from .  components import calibrate_sipms
from .  components import get_actual_sipm_thr


@city
def hypathia( files_in        : OneOrManyFiles
            , file_out        : str
            , compression     : str
            , event_range     : EventRangeType
            , print_mod       : int
            , detector_db     : str
            , run_number      : int
            , sipm_noise_cut  : float
            , filter_padding  : int
            , thr_sipm        : float
            , thr_sipm_type   : SiPMThreshold
            , pmt_wfs_rebin   : int
            , pmt_pe_rms      : float
            , s1_lmin         : int  , s1_lmax     : int
            , s1_tmin         : float, s1_tmax     : float
            , s1_rebin_stride : int  , s1_stride   : int
            , thr_csum_s1     : float
            , s2_lmin         : int  , s2_lmax     : int
            , s2_tmin         : float, s2_tmax     : float
            , s2_rebin_stride : int  , s2_stride   : int
            , thr_csum_s2     : float, thr_sipm_s2 : float
            , pmt_samp_wid    : float
            , sipm_samp_wid   : float
            ):

    sipm_thr = get_actual_sipm_thr(thr_sipm_type, thr_sipm, detector_db, run_number)

    #### Define data transformations
    sd = sensor_data(files_in[0], WfType.mcrd)

    # Raw WaveForm to Corrected WaveForm
    mcrd_to_rwf      = fl.map(rebin_pmts(pmt_wfs_rebin),
                              args = "pmt",
                              out  = "rwf")

    # Add single pe fluctuation to pmts
    simulate_pmt = fl.map(partial(sf.charge_fluctuation, single_pe_rms=pmt_pe_rms),
                          args = "rwf",
                          out = "ccwfs")

    # Compute pmt sum
    pmt_sum          = fl.map(pmts_sum, args = 'ccwfs',
                              out  = 'pmt')

    # Find where waveform is above threshold
    zero_suppress    = fl.map(zero_suppress_wfs(thr_csum_s1, thr_csum_s2),
                              args = ("pmt", "pmt"),
                              out  = ("s1_indices", "s2_indices", "s2_energies"))

    # SiPMs simulation
    simulate_sipm_response_  = fl.map(simulate_sipm_response(detector_db, run_number,
                                                             sd.SIPMWL, sipm_noise_cut,
                                                             filter_padding),
                                     item="sipm")

    # Sipm calibration function expects waveform as int16
    discretize_signal = fl.map(lambda rwf: np.round(rwf).astype(np.int16),
                              item="sipm")

    # SiPMs calibration
    sipm_rwf_to_cal  = fl.map(calibrate_sipms(detector_db, run_number, sipm_thr),
                              item = "sipm")

    event_count_in  = fl.spy_count()
    event_count_out = fl.spy_count()

    evtnum_collect  = collect()

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:

        # Define writers...
        write_event_info_   = run_and_event_writer(h5out)
        write_trigger_info_ = trigger_writer      (h5out, get_number_of_active_pmts(detector_db, run_number))

        # ... and make them sinks
        write_event_info   = sink(write_event_info_  , args=(   "run_number",     "event_number", "timestamp"   ))
        write_trigger_info = sink(write_trigger_info_, args=( "trigger_type", "trigger_channels"                ))

        compute_pmaps, empty_indices, empty_pmaps = compute_and_write_pmaps(
                                             detector_db, run_number, pmt_samp_wid, sipm_samp_wid,
                                             s1_lmax, s1_lmin, s1_rebin_stride, s1_stride, s1_tmax, s1_tmin,
                                             s2_lmax, s2_lmin, s2_rebin_stride, s2_stride, s2_tmax, s2_tmin, thr_sipm_s2,
                                             h5out, sipm_rwf_to_cal)

        result = push(source = wf_from_files(files_in, WfType.mcrd),
                      pipe   = pipe(fl.slice(*event_range, close_all=True),
                                    print_every(print_mod),
                                    event_count_in.spy,
                                    mcrd_to_rwf,
                                    simulate_pmt,
                                    pmt_sum,
                                    zero_suppress,
                                    simulate_sipm_response_,
                                    discretize_signal,
                                    compute_pmaps,
                                    event_count_out.spy,
                                    fl.branch("event_number", evtnum_collect.sink),
                                    fl.fork(write_event_info,
                                            write_trigger_info)),
                     result = dict(events_in   = event_count_in .future,
                                   events_out  = event_count_out.future,
                                   evtnum_list = evtnum_collect .future,
                                   over_thr    = empty_indices  .future,
                                   full_pmap   = empty_pmaps    .future))

        if run_number <= 0:
            copy_mc_info(files_in, h5out, result.evtnum_list,
                         detector_db, run_number)


def rebin_pmts(rebin_stride):
    def rebin_pmts(rwf):
        rebinned_wfs = rwf
        if rebin_stride > 1:
            # dummy data for times and widths
            times     = np.zeros(rwf.shape[1])
            widths    = times
            waveforms = rwf
            _, _, rebinned_wfs = pkf.rebin_times_and_waveforms(times, widths, waveforms, rebin_stride=rebin_stride)
        return rebinned_wfs
    return rebin_pmts


def pmts_sum(rwfs):
    return rwfs.sum(axis=0)
