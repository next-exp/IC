"""
-----------------------------------------------------------------------
                               Ercilia
-----------------------------------------------------------------------

From Latin, Hersilia, hypocorism: "hair" or "tenderness,"

This city produces calibration spectra for fiber-coupled SiPMs. The
calibration is achieved by identifying signal peaks in the fiber
waveforms and integrating the waveform content within a configurable
window around each peak. The resulting integrals provide a measure of
the charge associated with individual light pulses transmitted through
the optical fibers.

The peak-finding procedure is designed to identify isolated optical
signals while suppressing baseline fluctuations and electronics noise.
For each detected peak, a region centred on the peak position is
selected and integrated to determine the pulse charge. The collection
of pulse charges is then histogrammed to produce the calibration
spectrum.

The tasks performed are:
    - Subtract the baseline.
    - Search for peaks.
    - Determine integration windows.
    - Integrate around detected peaks.
    - Collect pulse charges.
    - Histogram the result.
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
from .  components import peak_charge_binner


@city
def ercilia( files_in         : OneOrManyFiles
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
    sd          = sensor_data(files_in[0], WfType.rwf, detector_db)
    nfiber       = sd.NPMT  # Use PMT since SensorData tuple is only configured for SiPMs and PMTs
    wf_length   = sd.PMTWL  # Use PMT since SensorData tuple is only configured for SiPMs and PMTs
    shape       = nfiber, len(bin_centres)
    
    sampling    = 25 * units.ns  # Maybe not hard code this?


    subtract_baseline = fl.map(
        partial(csf.subtract_and_flip, proc_mode=proc_mode),
        args="pmt",
        out="bls"
    )
    # Add a peak finder here, Define limits via peak finder, not via valid_integral_limits
    extract_charges = fl.map(
        cf.integrate_peaks_ercilia,
        args="bls",
        out="charges"
    )
    bin_charges = fl.map(
        peak_charge_binner(bin_edges),
        args="charges",
        out="hist"
    )
    
    sum_histograms   = fl.reduce(add, np.zeros(shape, dtype=int))
    accumulate_light = sum_histograms()
    event_count      = fl.spy_count()

    with tb.open_file(file_out, "w", filters=tbl.filters(compression)) as h5out:
        write_event_info    = run_and_event_writer(h5out)
        write_run_and_event = fl.sink(write_event_info, args=("run_number", "event_number", "timestamp"))
        write_hist          = partial(hist_writer,
                                      h5out,
                                      group_name  = "HIST",
                                      n_sensors   = nfiber,
                                      bin_centres = bin_centres)


        out = fl.push(
            source = wf_from_files(files_in, WfType.rwf, detector_db),

            pipe = fl.pipe(
                fl.slice(*event_range, close_all=True),
                event_count.spy,
                print_every(print_mod),
                subtract_baseline,
                extract_charges,
                bin_charges,
                fl.fork(
                    ("hist", accumulate_light.sink),
                    write_run_and_event
                )
            ),

            result = dict(
                events_in = event_count.future,
                spe       = accumulate_light.future
            )
        )

        write_hist(table_name = "fiber_spe" )(out.spe )
        cf.copy_sensor_table(files_in[0], h5out)

    return out
