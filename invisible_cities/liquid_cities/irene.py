import numpy  as np
import tables as tb

from .. types.ic_types import minmax
from .. database       import load_db

from .. reco                  import tbl_functions        as tbl
from .. reco                  import  peak_functions      as pkf
from .. core.random_sampling  import NoiseSampler         as SiPMsNoiseSampler
from .. io  .        pmaps_io import          pmap_writer
from .. io.        mcinfo_io  import       mc_info_writer
from .. io  .run_and_event_io import run_and_event_writer
from .. io  .trigger_io       import       trigger_writer

from .. dataflow            import dataflow as fl
from .. dataflow.dataflow   import push
from .. dataflow.dataflow   import pipe
from .. dataflow.dataflow   import sink

from .  components import city
from .  components import print_every
from .  components import deconv_pmt
from .  components import calibrate_pmts
from .  components import calibrate_sipms
from .  components import zero_suppress_wfs
from .  components import WfType
from .  components import wf_from_files


@city
def irene(files_in, file_out, compression, event_range, print_mod, run_number,
          n_baseline, raw_data_type,
          n_mau, thr_mau, thr_sipm, thr_sipm_type,
          s1_lmin, s1_lmax, s1_tmin, s1_tmax, s1_rebin_stride, s1_stride, thr_csum_s1,
          s2_lmin, s2_lmax, s2_tmin, s2_tmax, s2_rebin_stride, s2_stride, thr_csum_s2, thr_sipm_s2):
    if   thr_sipm_type.lower() == "common":
        # In this case, the threshold is a value in pes
        sipm_thr = thr_sipm

    elif thr_sipm_type.lower() == "individual":
        # In this case, the threshold is a percentual value
        noise_sampler = SiPMsNoiseSampler(run_number)
        sipm_thr      = noise_sampler.compute_thresholds(thr_sipm)

    else:
        raise ValueError(f"Unrecognized thr type: {thr_sipm_type}. "
                          "Only valid options are 'common' and 'individual'")

    #### Define data transformations

    # Raw WaveForm to Corrected WaveForm
    rwf_to_cwf       = fl.map(deconv_pmt(run_number, n_baseline),
                              args = "pmt",
                              out  = "cwf")

    # Corrected WaveForm to Calibrated Corrected WaveForm
    cwf_to_ccwf      = fl.map(calibrate_pmts(run_number, n_mau, thr_mau),
                              args = "cwf",
                              out  = ("ccwfs", "ccwfs_mau", "cwf_sum", "cwf_sum_mau"))

    # Find where waveform is above threshold
    zero_suppress    = fl.map(zero_suppress_wfs(thr_csum_s1, thr_csum_s2),
                              args = ("cwf_sum", "cwf_sum_mau"),
                              out  = ("s1_indices", "s2_indices", "s2_energies"))

    # Remove baseline and calibrate SiPMs
    sipm_rwf_to_cal  = fl.map(calibrate_sipms(run_number, sipm_thr),
                              item = "sipm")

    # Build the PMap
    compute_pmap     = fl.map(build_pmap(run_number,
                                         s1_lmax, s1_lmin, s1_rebin_stride, s1_stride, s1_tmax, s1_tmin,
                                         s2_lmax, s2_lmin, s2_rebin_stride, s2_stride, s2_tmax, s2_tmin, thr_sipm_s2),
                              args = ("ccwfs", "s1_indices", "s2_indices", "sipm"),
                              out  = "pmap")

    ### Define data filters

    # Filter events with zero peaks
    empty_indices   = fl.count_filter(check_nonempty_indices, args = ("s1_indices", "s2_indices"))

    event_count_in  = fl.spy_count()
    event_count_out = fl.spy_count()

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:

        # Define writers...
        write_event_info_   = run_and_event_writer(h5out)
        write_mc_           = mc_info_writer      (h5out) if run_number <= 0 else (lambda *_: None)
        write_pmap_         = pmap_writer         (h5out, compression=compression)
        write_trigger_info_ = trigger_writer      (h5out, get_number_of_active_pmts(run_number))

        # ... and make them sinks
        write_event_info   = sink(write_event_info_  , args=(   "run_number",     "event_number", "timestamp"))
        write_mc           = sink(write_mc_          , args=(           "mc",     "event_number"             ))
        write_pmap         = sink(write_pmap_        , args=(         "pmap",     "event_number"             ))
        write_trigger_info = sink(write_trigger_info_, args=( "trigger_type", "trigger_channels"             ))

        return push(source = wf_from_files(files_in, WfType.rwf),
                    pipe   = pipe(
                                fl.slice(*event_range, close_all=True),
                                print_every(print_mod),
                                event_count_in.spy,
                                rwf_to_cwf,
                                cwf_to_ccwf,
                                zero_suppress,
                                empty_indices.filter,
                                sipm_rwf_to_cal,
                                compute_pmap,
                                event_count_out.spy,
                                fl.fork(write_pmap,
                                        write_mc,
                                        write_event_info,
                                        write_trigger_info)),
                    result = dict(events_in  = event_count_in .future,
                                  events_out = event_count_out.future,
                                  empty_pmap = empty_indices  .future))



def build_pmap(run_number,
               s1_lmax, s1_lmin, s1_rebin_stride, s1_stride, s1_tmax, s1_tmin,
               s2_lmax, s2_lmin, s2_rebin_stride, s2_stride, s2_tmax, s2_tmin, thr_sipm_s2):
    s1_params = dict(time        = minmax(min = s1_tmin,
                                          max = s1_tmax),
                    length       = minmax(min = s1_lmin,
                                          max = s1_lmax),
                    stride       = s1_stride,
                    rebin_stride = s1_rebin_stride)

    s2_params = dict(time        = minmax(min = s2_tmin,
                                          max = s2_tmax),
                    length       = minmax(min = s2_lmin,
                                          max = s2_lmax),
                    stride       = s2_stride,
                    rebin_stride = s2_rebin_stride)

    datapmt = load_db.DataPMT(run_number)
    pmt_ids = datapmt.SensorID[datapmt.Active.astype(bool)].values

    def build_pmap(ccwf, s1_indx, s2_indx, sipmzs): # -> PMap
        return pkf.get_pmap(ccwf, s1_indx, s2_indx, sipmzs,
                            s1_params, s2_params, thr_sipm_s2, pmt_ids)

    return build_pmap


def get_number_of_active_pmts(run_number):
    datapmt = load_db.DataPMT(run_number)
    return np.count_nonzero(datapmt.Active.values.astype(bool))


def check_nonempty_indices(s1_indices, s2_indices):
    return s1_indices.size and s2_indices.size
