"""
-----------------------------------------------------------------------
                                 Irene
-----------------------------------------------------------------------

From ancient Greek, Εἰρήνη: Peace.

This city finds the signal pulses within the waveforms produced by the
detector or by diomira in the case of Monte Carlo data.
This includes a number of tasks:
    - Remove the signal-derivative effect of the PMT waveforms.
    - Calibrate PMTs and produced a PMT-summed waveform.
    - Remove the baseline from the SiPM waveforms and calibrate them.
    - Apply a threshold to the PMT-summed waveform.
    - Find pulses in the PMT-summed waveform.
    - Match the time window of the PMT pulse with those in the SiPMs.
    - Build the PMap object.
"""
import tables as tb

from .. reco                  import tbl_functions        as tbl
from .. core.random_sampling  import NoiseSampler         as SiPMsNoiseSampler
from .. core                  import system_of_units      as units
from .. io  .run_and_event_io import run_and_event_writer
from .. io  .trigger_io       import       trigger_writer

from .. dataflow            import dataflow as fl
from .. dataflow.dataflow   import push
from .. dataflow.dataflow   import pipe
from .. dataflow.dataflow   import sink

from .  components import city
from .  components import print_every
from .  components import collect
from .  components import copy_mc_info
from .  components import deconv_pmt
from .  components import calibrate_pmts
from .  components import calibrate_sipms
from .  components import zero_suppress_wfs
from .  components import WfType
from .  components import wf_from_files
from .  components import get_number_of_active_pmts
from .  components import compute_and_write_pmaps


@city
def irene(files_in, file_out, compression, event_range, print_mod, detector_db, run_number,
          n_baseline, n_mau, thr_mau, thr_sipm, thr_sipm_type,
          s1_lmin, s1_lmax, s1_tmin, s1_tmax, s1_rebin_stride, s1_stride, thr_csum_s1,
          s2_lmin, s2_lmax, s2_tmin, s2_tmax, s2_rebin_stride, s2_stride, thr_csum_s2, thr_sipm_s2,
          pmt_samp_wid=25*units.ns, sipm_samp_wid=1*units.mus):
    if   thr_sipm_type.lower() == "common":
        # In this case, the threshold is a value in pes
        sipm_thr = thr_sipm

    elif thr_sipm_type.lower() == "individual":
        # In this case, the threshold is a percentual value
        noise_sampler = SiPMsNoiseSampler(detector_db, run_number)
        sipm_thr      = noise_sampler.compute_thresholds(thr_sipm)

    else:
        raise ValueError(f"Unrecognized thr type: {thr_sipm_type}. "
                          "Only valid options are 'common' and 'individual'")

    #### Define data transformations

    # Raw WaveForm to Corrected WaveForm
    rwf_to_cwf       = fl.map(deconv_pmt(detector_db, run_number, n_baseline),
                              args = "pmt",
                              out  = "cwf")

    # Corrected WaveForm to Calibrated Corrected WaveForm
    cwf_to_ccwf      = fl.map(calibrate_pmts(detector_db, run_number, n_mau, thr_mau),
                              args = "cwf",
                              out  = ("ccwfs", "ccwfs_mau", "cwf_sum", "cwf_sum_mau"))

    # Find where waveform is above threshold
    zero_suppress    = fl.map(zero_suppress_wfs(thr_csum_s1, thr_csum_s2),
                              args = ("cwf_sum", "cwf_sum_mau"),
                              out  = ("s1_indices", "s2_indices", "s2_energies"))

    # Remove baseline and calibrate SiPMs
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
                                         s2_lmax, s2_lmin, s2_rebin_stride, s2_stride, s2_tmax, s2_tmin,
                                         thr_sipm_s2,
                                         h5out, compression, sipm_rwf_to_cal)

        result = push(source = wf_from_files(files_in, WfType.rwf),
                      pipe   = pipe(fl.slice(*event_range, close_all=True),
                                    print_every(print_mod),
                                    event_count_in.spy,
                                    rwf_to_cwf,
                                    cwf_to_ccwf,
                                    zero_suppress,
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
            copy_mc_info(files_in, h5out, result.evtnum_list)

        return result
