"""
-----------------------------------------------------------------------
                                Berenice
-----------------------------------------------------------------------

From Βερενíκη, Ancient Macedonian form of the Attic Greek, Φερενíκη: she, who brings victory.

This city produces the spectrum of dark noise of the SiPMs. This is
achieved by binning either the pe or ADC content of each raw WF sample.
Some slices will have only electronic noise while others will contain
one or more dark counts. The resulting spectra give a representation
of the SiPM charge in the absence of external light above detector
ambient conditions.
The spectrum is produced in three flavours:
    - Using the mode   to subtract the baseline without calibrating.
    - Using the mode   to subtract the baseline and     calibrating.
    - Using the median to subtract the baseline and     calibrating.
The tasks performed are:
    - Subtract the baseline   (with different algorithms).
    - Calibrate the waveforms (not in the first case).
    - Slice the waveforms.
    - Integrate slices.
    - Histogram the result.
"""
from operator  import add
from functools import partial

import numpy  as np
import tables as tb

from .. core .configure        import          EventRangeType
from .. core .configure        import          OneOrManyFiles
from .. core .core_functions   import    shift_to_bin_centers
from .. core                   import           tbl_functions as tbl
from .. io   .    histogram_io import             hist_writer
from .. io   .run_and_event_io import    run_and_event_writer
from .. calib                  import         calib_functions as cf
from .. calib                  import calib_sensors_functions as csf
from .. database               import                 load_db
from .. types.symbols          import                  WfType

from .. dataflow import dataflow as fl

from .  components import city
from .  components import print_every
from .  components import sensor_data
from .  components import wf_from_files
from .  components import waveform_binner


@city
def berenice( files_in    : OneOrManyFiles
            , file_out    : str
            , compression : str
            , event_range : EventRangeType
            , print_mod   : int
            , detector_db : str
            , run_number  : int
            , min_bin     : float
            , max_bin     : float
            , bin_width   : float
            ):
    bin_edges   = np.arange(min_bin, max_bin, bin_width)
    bin_centres = shift_to_bin_centers(bin_edges)
    nsipm       = sensor_data(files_in[0], WfType.rwf).NSIPM
    shape       = nsipm, len(bin_centres)

    subtract_mode         = fl.map(csf.subtract_mode            )
    calibrate_with_mode   = fl.map(mode_calibrator  (detector_db, run_number))
    calibrate_with_median = fl.map(median_calibrator(detector_db, run_number))

    bin_waveforms         = fl.map(waveform_binner  (bin_edges ))
    sum_histograms        = fl.reduce(add, np.zeros(shape, dtype=int))

    accumulate_adc        = sum_histograms()
    accumulate_mode       = sum_histograms()
    accumulate_median     = sum_histograms()

    event_count = fl.spy_count()

    with tb.open_file(file_out, 'w', filters=tbl.filters(compression)) as h5out:
        write_event_info    = run_and_event_writer(h5out)
        write_run_and_event = fl.sink(write_event_info, args=("run_number", "event_number", "timestamp"))

        write_hist = partial(hist_writer,
                             h5out,
                             group_name  = 'HIST',
                             n_sensors   = nsipm,
                             bin_centres = bin_centres)

        out = fl.push(
            source = wf_from_files(files_in, WfType.rwf),
            pipe   = fl.pipe(fl.slice(*event_range, close_all=True),
                             event_count.spy,
                             print_every(print_mod),
                             fl.fork(("sipm", subtract_mode        , bin_waveforms, accumulate_adc     .sink),
                                     ("sipm", calibrate_with_mode  , bin_waveforms, accumulate_mode    .sink),
                                     ("sipm", calibrate_with_median, bin_waveforms, accumulate_median  .sink),
                                                                                    write_run_and_event      )),

            result = dict(events_in   = event_count      .future,
                          adc         = accumulate_adc   .future,
                          mode        = accumulate_mode  .future,
                          median      = accumulate_median.future))

        write_hist(table_name = "adc"   )(out.adc   )
        write_hist(table_name = "mode"  )(out.mode  )
        write_hist(table_name = "median")(out.median)
        cf.copy_sensor_table(files_in[0], h5out)

    return out


def mode_calibrator(detector_db, run_number):
    adc_to_pes = load_db.DataSiPM(detector_db, run_number).adc_to_pes.values
    def calibrate_with_mode(wfs):
        return csf.sipm_subtract_mode_and_calibrate(wfs, adc_to_pes)
    return calibrate_with_mode


def median_calibrator(detector_db, run_number):
    adc_to_pes = load_db.DataSiPM(detector_db, run_number).adc_to_pes.values
    def calibrate_with_median(wfs):
        return csf.sipm_subtract_median_and_calibrate(wfs, adc_to_pes)
    return calibrate_with_median
