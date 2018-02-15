"""
code: sipmgain.py
description: Generates spectra using integrals
at configuration file specified regions (will fit eventually)
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
from ..  core               import system_of_units as units


class Sipmgain(CalibratedCity):
    """
    Generates spectra using integrals in each buffer
    according to start, end, width and repetition
    given in the configuration file.
    input: RAW data (pulsed LED normally)
    output: Spectra (spe spectra)
    TODO: add fits and output results.
    """

    parameters = tuple("""proc_mode min_bin max_bin bin_wid number_integrals start_integral width_integral period_integrals""".split())

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

        ## the mode of processing (temp).
        ## Can be 'gain': subtract mode
        ## or 'gainM': subtract mean
        self.pmode = conf.proc_mode
        
        min_bin = conf.min_bin
        max_bin = conf.max_bin
        bin_wid = conf.bin_wid
        self.bin_half_wid = bin_wid / 2.
        self.histbins = np.arange(min_bin, max_bin, bin_wid)

        ## The integral limits
        s_wid = 1 * units.mus
        self.l_limits, self.d_limits = cf.int_limits(s_wid,
                                                         conf.number_integrals,
                                                         conf.start_integral,
                                                         conf.width_integral,
                                                         conf.period_integrals)


    def event_loop(self, NEVT, dataVectors):
        """actions:
        1. loops over all the events in each file.
        2. write event/run to file
        3. write histogram info to file (to reduce memory usage)
        """
        write = self.writers
        sipmrwf      = dataVectors.sipm
        events_info  = dataVectors.events

        ## Make sure the limits are valid for these data.
        ## Need to be in range. Should probably give a warning.
        ## If the start of a limit is out of range the end must be
        ## removed too and vice versa
        ## Correlated
        self.l_limits = cf.filter_limits(self.l_limits, len(sipmrwf[0][0]))
        ## Anti-correlated
        self.d_limits = cf.filter_limits(self.d_limits, len(sipmrwf[0][0]))
        ###

        ## Where we'll be saving the binned info for each channel
        bsipmspe = np.zeros((sipmrwf.shape[1], len(self.histbins)-1),
                            dtype=np.int)
        bsipmdar = np.zeros((sipmrwf.shape[1], len(self.histbins)-1),
                            dtype=np.int)

        for evt in range(NEVT):
            self.conditional_print(evt, self.cnt.n_events_tot)

            what_next = self.event_range_step()
            if what_next is EventLoop.skip_this_event: continue
            if what_next is EventLoop.terminate_loop : break
            self.cnt.n_events_tot += 1

            ## Get the pedestal subtracted ADC WF
            sipmadc = csf.sipm_processing[self.pmode](sipmrwf[evt])

            ## Values of the integrals, [:, ::2] since those between limits
            ## automatically calculated too.
            # LED correlated.
            led_ints = cf.spaced_integrals(sipmadc, self.l_limits)[:, ::2]
            bsipmspe += cf.bin_waveforms(led_ints, self.histbins)

            # Dark (anti-correlated)
            dar_ints = cf.spaced_integrals(sipmadc, self.d_limits)[:, ::2]
            bsipmdar += cf.bin_waveforms(dar_ints, self.histbins)

            # write stuff
            event, timestamp = self.event_and_timestamp(evt, events_info)
            write.run_and_event(self.run_number, event, timestamp)
        write.led(bsipmspe)
        write.dar(bsipmdar)


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
            led  = HIST(table_name='sipmspe'),
            dar  = HIST(table_name='sipmdar'))

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
