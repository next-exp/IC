from operator  import add
from functools import partial

import numpy  as np
import tables as tb

from .. core.system_of_units_c import units
from .. io.         hist_io    import          hist_writer
from .. io.run_and_event_io    import run_and_event_writer
from .. icaro.hst_functions    import shift_to_bin_centers
from .. database               import load_db
from .. reco                   import           tbl_functions as tbl
from .. reco                   import calib_functions         as cf
from .. reco                   import calib_sensors_functions as csf

from .. dataflow import dataflow as fl

from .  components import city
from .  components import WfType
from .  components import print_every
from .  components import sensor_data
from .  components import wf_from_files
from .  components import waveform_binner
from .  components import waveform_integrator


@city
def moriana(files_in, file_out, compression, event_range, print_mod, run_number,
            raw_data_type, proc_mode,
            min_bin, max_bin, bin_width,
            number_integrals, integral_start, integral_width, integrals_period,
            n_mau = 100):
    raw_data_type_ = getattr(WfType, raw_data_type.lower())

    if proc_mode not in ("subtract_mode", "subtract_median"):
        raise ValueError(f"Unrecognized processing mode: {proc_mode}")

    bin_edges   = np.arange(min_bin, max_bin, bin_width)
    bin_centres = shift_to_bin_centers(bin_edges)
    sd          = sensor_data(files_in[0], raw_data_type_)
    nsipm       = sd.NSIPM
    wf_length   = sd.SIPMWL
    shape       = nsipm, len(bin_centres)
    sampling    = 1 * units.mus

    (light_limits,
      dark_limits) = cf.valid_integral_limits(sampling        ,
                                              number_integrals,
                                              integral_start  ,
                                              integral_width  ,
                                              integrals_period,
                                              wf_length       )

    subtract_baseline = fl.map(csf.sipm_processing[proc_mode], args="sipm", out="bls")
    integrate_light   = fl.map(waveform_integrator(light_limits))
    integrate_dark    = fl.map(waveform_integrator( dark_limits))
    bin_waveforms     = fl.map(waveform_binner    (  bin_edges ))
    sum_histograms    = fl.reduce(add, np.zeros(shape, dtype=np.int))
    accumulate_light  = sum_histograms()
    accumulate_dark   = sum_histograms()
    event_count       = fl.spy_count()

    with tb.open_file(file_out, "w", filters=tbl.filters(compression)) as h5out:
        write_event_info    = run_and_event_writer(h5out)
        write_run_and_event = fl.sink(write_event_info, args=("run_number", "event_number", "timestamp"))
        write_hist          = partial(hist_writer,
                                      h5out,
                                      group_name  = "HIST",
                                      n_sensors   = nsipm,
                                      bin_centres = bin_centres)

        out = fl.push(
            source = wf_from_files(files_in, raw_data_type_),
            pipe   = fl.pipe(fl.slice(*event_range, close_all=True),
                             event_count.spy,
                             print_every(print_mod),
                             subtract_baseline,
                             fl.fork(("bls", integrate_light, bin_waveforms, accumulate_light   .sink),
                                     ("bls", integrate_dark , bin_waveforms, accumulate_dark    .sink),
                                                                             write_run_and_event      )),

            result = dict(events_in   = event_count     .future,
                          spe         = accumulate_light.future,
                          dark        = accumulate_dark .future,
                          event_count =      event_count.future)
        )

        write_hist(table_name = "sipm_spe" )(out.spe )
        write_hist(table_name = "sipm_dark")(out.dark)

    return out
