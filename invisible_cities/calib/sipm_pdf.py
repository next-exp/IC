"""
code: sipmPDF.py
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

from .. io.         hist_io    import          hist_writer
from .. io.run_and_event_io    import run_and_event_writer

from ..  cities.base_cities import CalibratedCity
from ..  cities.base_cities import EventLoop


class Sipm_pdf(CalibratedCity):
    """
    Generates binned spectra of sipm rwf - mean
    and (rwf - mean)-mau
    """

    parameters = tuple("""min_bin max_bin bin_wid""".split())

    def __init__(self, **kwds):
        """sipmPDF Init:
        1. inits base city
        2. inits counters
        3. gets sensor parameters
        """

        super().__init__(**kwds)

        self.cnt.init(n_events_tot = 0)
        self.sp = self.get_sensor_params(self.input_files[0])

        ## The bin range for the histograms
        conf = self.conf
        min_bin = conf.min_bin
        max_bin = conf.max_bin
        bin_wid = conf.bin_wid
        self.histbins = np.arange(min_bin, max_bin, bin_wid)

    def event_loop(self, NEVT, dataVectors):
        """actions:
        1. loops over all the events in each file.
        2. write event/run to file
        3. write histogram info to file (to reduce memory usage)
        """
        write = self.writers
        sipmrwf      = dataVectors.sipm
        events_info = dataVectors.events

        ## Where we'll be saving the binned info for each channel
        bsipmzs = np.zeros((sipmrwf.shape[1], len(self.histbins)-1), dtype=np.int)
        bsipmmzs = np.zeros((sipmrwf.shape[1], len(self.histbins)-1), dtype=np.int)

        for evt in range(NEVT):
            self.conditional_print(evt, self.cnt.n_events_tot)

            what_next = self.event_range_step()
            if what_next is EventLoop.skip_this_event: continue
            if what_next is EventLoop.terminate_loop : break
            self.cnt.n_events_tot += 1

            ## Zeroed sipm waveforms in pe
            sipmzs = self.calibrated_signal_sipm(sipmrwf[evt], cal=1)
            bsipmzs += self.bin_waveform(sipmzs)

            ## Difference from the MAU
            sipmmzs = self.calibrated_signal_sipm(sipmrwf[evt], cal=2)
            bsipmmzs += self.bin_waveform(sipmmzs)

            # write stuff
            event, timestamp = self.event_and_timestamp(evt, events_info)
            write.run_and_event(self.run_number, event, timestamp)
        write.sipm(bsipmzs)
        write.mausi(bsipmmzs)


    def bin_waveform(self, waveData):
        """ Bins the current event data and adds it
        to the file level bin array """

        binData = np.array([np.histogram(sipm, self.histbins)[0] for sipm in waveData])

        return binData


    def get_writers(self, h5out):
        HIST = partial(hist_writer,  h5out,   group_name='HIST')
        writers = Namespace(
            run_and_event = run_and_event_writer(h5out),
            sipm  = HIST(table_name='sipm' ,
                        n_sensors=self.sp.NSIPM , n_bins=len(self.histbins)-1,
                        bin_centres=(self.histbins+0.05)[:-1]),
            mausi = HIST(table_name='sipmMAU' ,
                        n_sensors=self.sp.NSIPM , n_bins=len(self.histbins)-1,
                        bin_centres=(self.histbins+0.05)[:-1])
            )

        return writers

    def write_parameters(self, h5out):
        pass

    def display_IO_info(self):
        super().display_IO_info()
        print(self.sp)

    def _copy_sensor_table(self, h5in):
        # Copy sensor table if exists (needed for GATE)
        if 'Sensors' in h5in.root:
            self.sensors_group = self.output_file.create_group(
                self.output_file.root, "Sensors")
            datapmt = h5in.root.Sensors.DataPMT
            datapmt.copy(newparent=self.sensors_group)
            datasipm = h5in.root.Sensors.DataSiPM
            datasipm.copy(newparent=self.sensors_group)
