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
from .. io.amplification_io  import amplification_writer
from .. core.core_functions  import    shift_to_bin_centers
from .. core                 import           tbl_functions as tbl
from .. calib                import         calib_functions as cf
from .. calib                import calib_sensors_functions as csf
from .. types.symbols        import                  WfType
from .. types.symbols        import           SiPMCalibMode
from .. types.symbols        import              SensorType

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
           , amplification    : bool       = False
           , sensor_type      : SensorType = SensorType.PMT
           ):
    if proc_mode not in SiPMCalibMode:
        raise ValueError(f"Unrecognized processing mode: {proc_mode}")

    if amplification and sensor_type is SensorType.SIPM:
        raise ValueError(
            "amplification=True is not supported with sensor_type=SensorType.SIPM "
            "(tracking plane); no LG channel exists for the sipm plane"
        )

    bin_edges   = np.arange(min_bin, max_bin, bin_width)
    bin_centres = shift_to_bin_centers(bin_edges)
    sd          = sensor_data(files_in[0], WfType.rwf, detector_db, sensor_type=sensor_type)
    nfiber      = sd.NPMT
    wf_length   = sd.PMTWL
    shape       = nfiber, len(bin_centres)

    # Peak-finder parameters are sample-count based, so they differ
    # between the fiber barrel (25 ns sampling) and tracking plane (1 µs sampling).
    if sensor_type is SensorType.SIPM:
        peak_finder_kwargs = dict(
            pre_samples    = 1,
            post_samples   = 2,
            min_distance   = 4,
            merge_distance = 2,
        )
    else:
        peak_finder_kwargs = dict(
            pre_samples    = 5,
            post_samples   = 10,
            min_distance   = 10,
            merge_distance = 30,
        )

    subtract_baseline = fl.map(
        partial(csf.subtract_and_flip, proc_mode=proc_mode, flip=sensor_type is not SensorType.SIPM),
        args="pmt",
        out="bls"
    )
    extract_charges = fl.map(
        partial(cf.integrate_peaks_ercilia, **peak_finder_kwargs),
        args="bls",
        out="charges"
    )
    bin_charges = fl.map(
        peak_charge_binner(bin_edges),
        args="charges",
        out="hist"
    )

    if amplification:
        subtract_baseline_lg = fl.map(
            partial(csf.subtract_and_flip, proc_mode=proc_mode),
            args="pmt_lg",
            out="bls_lg"
        )
        extract_pairs = fl.map(
            cf.integrate_hg_lg_pairs_ercilia,
            args=("bls", "bls_lg"),
            out=("channels", "areas_hg", "areas_lg")
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

        pipe_steps = [
            fl.slice(*event_range, close_all=True),
            event_count.spy,
            print_every(print_mod),
            subtract_baseline,
            extract_charges,
            bin_charges,
        ]
        fork_branches = [("hist", accumulate_light.sink), write_run_and_event]

        if amplification:
            write_amp_row       = amplification_writer(h5out)
            write_amplification = fl.sink(write_amp_row,
                                           args=("event_number", "channels", "areas_hg", "areas_lg"))
            pipe_steps   += [subtract_baseline_lg, extract_pairs]
            fork_branches.append(write_amplification)

        pipe_steps.append(fl.fork(*fork_branches))

        out = fl.push(
            source = wf_from_files(files_in, WfType.rwf, detector_db,
                                    amplification=amplification, sensor_type=sensor_type),
            pipe   = fl.pipe(*pipe_steps),
            result = dict(
                events_in = event_count.future,
                spe       = accumulate_light.future
            )
        )

        write_hist(table_name = "fiber_spe")(out.spe)
        cf.copy_sensor_table(files_in[0], h5out)

    return out