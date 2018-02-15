"""
code: sipmpdf.py
description: Generates binned spectra of sipm rwf - mean
and (rwf - mean)-mau
credits: see ic_authors_and_legal.rst in /doc

last revised:
"""
import sys

from argparse import Namespace

from functools import partial

import numpy  as np
import tables as tb

from .                         import calib_functions as cf
from .. reco                   import calib_sensors_functions as csf
from .. io.         hist_io    import          hist_writer
from .. io.run_and_event_io    import run_and_event_writer
from .. icaro.hst_functions    import shift_to_bin_centers

from ..  cities.base_cities import CalibratedCity
from ..  cities.base_cities import EventLoop


class Sipmpdf(CalibratedCity):
    """
    Generates binned spectra of sipm rwf - mean
    and (rwf - mean)-mau
    Reads: Raw data waveforms.
    Produces: Histograms of pedestal subtracted waveforms.
    """

    parameters = tuple("""min_bin max_bin bin_wid""".split())

    def __init__(self, **kwds):
        """
        sipmPDF Init:
        1. inits base city
        2. inits counters
        3. gets sensor parameters
        """

        super().__init__(**kwds)

        self.cnt.init(n_events_tot = 0)
        self.sp = self.get_sensor_params(self.input_files[0])

        ## The bin range for the histograms
        min_bin = self.conf.min_bin
        max_bin = self.conf.max_bin
        bin_wid = self.conf.bin_wid
        self.histbins = np.arange(min_bin, max_bin, bin_wid)

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
        shape    = sipmrwf.shape[1], len(self.histbins) - 1
        bsipmzs  = np.zeros(shape, dtype=np.int)
        bsipmmzs = np.zeros(shape, dtype=np.int)

        for evt in range(NEVT):
            self.conditional_print(evt, self.cnt.n_events_tot)

            what_next = self.event_range_step()
            if what_next is EventLoop.skip_this_event: continue
            if what_next is EventLoop.terminate_loop : break
            self.cnt.n_events_tot += 1

            ## Zeroed sipm waveforms in pe
            ## sipmzs   = self.calibrate_with_mean(sipmrwf[evt])
            sipmzs   = csf.sipm_processing['pdf'](sipmrwf[evt],
                                                      self.sipm_adc_to_pes)
            bsipmzs += cf.bin_waveforms(sipmzs, self.histbins)

            ## Difference from the MAU
            sipmmzs   = csf.sipm_processing['pdfMM'](sipmrwf[evt],
                                                         self.sipm_adc_to_pes)
            bsipmmzs += cf.bin_waveforms(sipmmzs, self.histbins)

            # write stuff
            event, timestamp = self.event_and_timestamp(evt, events_info)
            write.run_and_event(self.run_number, event, timestamp)

        write.sipm (bsipmzs)
        write.mausi(bsipmmzs)


    def get_writers(self, h5out):
        bin_centres = shift_to_bin_centers(self.histbins)
        HIST        = partial(hist_writer,
                              h5out,
                              group_name  = 'HIST',
                              n_sensors   = self.sp.NSIPM,
                              n_bins      = len(bin_centres),
                              bin_centres = bin_centres)

        writers = Namespace(
            run_and_event = run_and_event_writer(h5out),
            sipm          = HIST(table_name  = 'sipm'),
            mausi         = HIST(table_name  = 'sipmMAU'))

        return writers

    def write_parameters(self, h5out):
        pass

    def display_IO_info(self):
        super().display_IO_info()
        print(self.sp)

    def _copy_sensor_table(self, h5in):
        # Copy sensor table if exists (needed for GATE)
        if 'Sensors' not in h5in.root: return

        group = self.output_file.create_group(self.output_file.root,
                                              "Sensors")
        datapmt  = h5in.root.Sensors.DataPMT
        datasipm = h5in.root.Sensors.DataSiPM
        datapmt .copy(newparent=group)
        datasipm.copy(newparent=group)
