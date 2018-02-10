"""
code: sipmpdf.py
description: Generates binned spectra of sipm rwf - mean
and (rwf - mean)-mau
credits: see ic_authors_and_legal.rst in /doc

last revised:
"""
from argparse  import Namespace
from functools import partial

import numpy  as np

from .                        import calib_functions         as cf
from .. reco                  import calib_sensors_functions as csf
from .. io  .         hist_io import          hist_writer
from .. io  .run_and_event_io import run_and_event_writer
from .. core.core_functions   import shift_to_bin_centers

from .. cities.base_cities import CalibratedCity
from .. cities.base_cities import EventLoop


class Sipmpdf(CalibratedCity):
    """
    Generates binned spectra of sipm rwf - mean
    and (rwf - mean)-mau
    Reads: Raw data waveforms.
    Produces: Histograms of pedestal subtracted waveforms.
    """
    parameters = tuple("""min_bin max_bin bin_width adc_only""".split())

    def __init__(self, **kwds):
        """
        sipmPDF Init:
        1. inits base city
        2. inits counters
        3. gets sensor parameters
        """
        super().__init__(**kwds)

        self.cnt.init(n_events_tot = 0)
        self.sp       = self.get_sensor_params(self.input_files[0])
        conf          = self.conf
        self.histbins = np.arange(conf.min_bin, conf.max_bin, conf.bin_width)

        ## ADC plots?
        self.adc_only = conf.adc_only
        
        self.sipm_processing_adc    = csf.sipm_processing["subtract_mode"]
        self.sipm_processing_mode   = csf.sipm_processing["subtract_mode_calibrate"]
        self.sipm_processing_median = csf.sipm_processing["subtract_median_calibrate"]

    def event_loop(self, NEVT, dataVectors):
        """
        actions:
        1. loops over all the events in each file.
        2. write event/run to file
        3. write histogram info to file (to reduce memory usage)
        """
        write       = self.writers
        sipmrwf     = dataVectors.sipm
        events_info = dataVectors.events

        ## Where we'll be saving the binned info for each channel
        shape          = sipmrwf.shape[1], len(self.histbins) - 1
        if self.adc_only:
            sipm_adc_zs    = np.zeros(shape, dtype=np.int)
        else:
            sipm_mode_zs   = np.zeros(shape, dtype=np.int)
            sipm_median_zs = np.zeros(shape, dtype=np.int)

        for evt in range(NEVT):
            self.conditional_print(evt, self.cnt.n_events_tot)

            what_next = self.event_range_step()
            if what_next is EventLoop.skip_this_event: continue
            if what_next is EventLoop.terminate_loop : break
            self.cnt.n_events_tot += 1

            wfs = sipmrwf[evt]

            # Zeroed sipm waveforms in pe
            if self.adc_only:
                sipm_adc    = self.sipm_processing_adc   (wfs)
                sipm_adc_zs    += cf.bin_waveforms(sipm_adc   , self.histbins)
            else:
                sipm_mode   = self.sipm_processing_mode  (wfs, self.sipm_adc_to_pes)
                sipm_median = self.sipm_processing_median(wfs, self.sipm_adc_to_pes)

                sipm_mode_zs   += cf.bin_waveforms(sipm_mode  , self.histbins)
                sipm_median_zs += cf.bin_waveforms(sipm_median, self.histbins)

            # write stuff
            event, timestamp = self.event_and_timestamp(evt, events_info)
            write.run_and_event(self.run_number, event, timestamp)

        if self.adc_only:
            write.adc_spec(sipm_adc_zs)
        else:
            write.sipm (sipm_mode_zs  )
            write.medsi(sipm_median_zs)


    def get_writers(self, h5out):
        cf.copy_sensor_table(self.input_files[0], h5out)
        
        bin_centres = shift_to_bin_centers(self.histbins)
        HIST        = partial(hist_writer,
                              h5out,
                              group_name  = 'HIST',
                              n_sensors   = self.sp.NSIPM,
                              bin_centres = bin_centres)

        if self.adc_only:
            writers = Namespace(
                run_and_event = run_and_event_writer(h5out),
                adc_spec      = HIST(table_name  = 'sipm_adc'))
            return writers

        writers = Namespace(
            run_and_event = run_and_event_writer(h5out),
            sipm          = HIST(table_name  = 'sipm_mode'  ),
            medsi         = HIST(table_name  = 'sipm_median'))

        return writers

    def write_parameters(self, h5out):
        pass

    def display_IO_info(self):
        super().display_IO_info()
        print(self.sp)
