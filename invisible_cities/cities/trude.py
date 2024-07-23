"""
-----------------------------------------------------------------------
                                Trude
-----------------------------------------------------------------------

From Germanic, Gertrud, hypocorism: female friend.

This city produces the light and dark spectrum of SiPMs for dedicated
calibration runs. This is achieved by selecting regions in the SiPM
waveforms where LED pulses are expected and regions which end 2
microseconds before, respectively and integrating the content within
each region. The regions before LED pulses should only contain
electronics noise and dark counts giving the zero external light
approximation whereas those in time with the pulses will contain one
or more detected photoelectrons. The waveform integrals are split into
two groups: those with expected photoelectrons (light) and those
without expected photoelectrons (dark). Each group produces a different
spectrum.

The spectra can be produced in two flavours:
    - Using the mode   to subtract the baseline.
    - Using the median to subtract the baseline.
The tasks performed are:
    - Subtract the baseline.
    - Deconvolve the waveform.
    - Slice the waveforms.
    - Split into light and dark.
    - Integrate slices.
    - Histrogram the result.
"""
from operator  import add
from functools import partial

import numpy  as np
import tables as tb

from .. core                 import         system_of_units as units
from .. core.configure       import          EventRangeType
from .. core.configure       import          OneOrManyFiles
from .. io.    histogram_io  import             hist_writer
from .. io.run_and_event_io  import    run_and_event_writer
from .. core.core_functions  import    shift_to_bin_centers
from .. core                 import           tbl_functions as tbl
from .. calib                import         calib_functions as cf
from .. calib                import calib_sensors_functions as csf
from .. types.symbols        import                  WfType
from .. types.symbols        import           SiPMCalibMode

from .. dataflow import dataflow as fl

from .  components import city
from .  components import print_every
from .  components import sensor_data
from .  components import wf_from_files
from .  components import waveform_binner
from .  components import waveform_integrator


@city
def trude( files_in         : OneOrManyFiles
         , file_out         : str
         , compression      : str
         , event_range      : EventRangeType
         , print_mod        : int
         , detector_db      : str
         , run_number       : int
         , proc_mode        : SiPMCalibMode
         , min_bin          : float
         , max_bin          : float
         , bin_width        : float
         , number_integrals : int
         , integral_start   : float
         , integral_width   : float
         , integrals_period : float
         ):
    if proc_mode not in SiPMCalibMode:
        raise ValueError(f"Unrecognized processing mode: {proc_mode}")

    bin_edges   = np.arange(min_bin, max_bin, bin_width)
    bin_centres = shift_to_bin_centers(bin_edges)
    sd          = sensor_data(files_in[0], WfType.rwf)
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
    sum_histograms    = fl.reduce(add, np.zeros(shape, dtype=int))
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
            source = wf_from_files(files_in, WfType.rwf),
            pipe   = fl.pipe(fl.slice(*event_range, close_all=True),
                             event_count.spy,
                             print_every(print_mod),
                             subtract_baseline,
                             fl.fork(("bls", integrate_light, bin_waveforms, accumulate_light   .sink),
                                     ("bls", integrate_dark , bin_waveforms, accumulate_dark    .sink),
                                                                             write_run_and_event      )),

            result = dict(events_in   = event_count     .future,
                          spe         = accumulate_light.future,
                          dark        = accumulate_dark .future)
        )

        write_hist(table_name = "sipm_spe" )(out.spe )
        write_hist(table_name = "sipm_dark")(out.dark)
        cf.copy_sensor_table(files_in[0], h5out)

    return out
