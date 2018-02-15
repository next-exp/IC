"""
code: pmtgain.py
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
from .. io.         hist_io    import          hist_writer
from .. io.run_and_event_io    import run_and_event_writer
from .. icaro.hst_functions    import shift_to_bin_centers

from ..  cities.base_cities import CalibratedCity
from ..  cities.base_cities import EventLoop
from ..  core               import system_of_units as units
from ..  sierpe             import fee


class Pmtgain(CalibratedCity):
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

        conf = self.conf

        ## the mode of processing (temp).
        ## Can be 'gain': just deconvolve and subtract ped
        ## or 'gainM': deconvolve, subtract ped, subtract MAU
        self.pmode = conf.proc_mode

        ## The bin range for the histograms
        min_bin = conf.min_bin
        max_bin = conf.max_bin
        bin_wid = conf.bin_wid
        self.bin_half_wid = bin_wid / 2.
        self.histbins = np.arange(min_bin, max_bin, bin_wid)

        ## The integral limits
        s_wid = fee.t_sample
        self.l_limits, self.d_limits = cf.int_limits(s_wid,
                                                         conf.number_integrals,
                                                         conf.start_integral,
                                                         conf.width_integral,
                                                         conf.period_integrals)
        ## Dict of functions for PMT processing
        def deconv_pmt_mau(self, RWF):
            CWF = self.deconv_pmt(RWF)
            return csf.pmt_subtract_mau(CWF, n_MAU=self.n_MAU)
        self.pmt_processing = {
            'gain'  : self.deconv_pmt,#Just deconvolve and remove ped
            'gainM' : deconv_pmt_mau#Deconv-ped+mau difference
        }


    def event_loop(self, NEVT, dataVectors):
        """actions:
        1. loops over all the events in each file.
        2. write event/run to file
        3. write histogram info to file (to reduce memory usage)
        """
        write = self.writers
        pmtrwf      = dataVectors.pmt
        events_info = dataVectors.events

        ## Make sure the limits are valid for these data.
        ## Need to be in range. Should probably give a warning.
        ## If the start of a limit is out of range the end must be
        ## removed too and vice versa
        ## Correlated
        self.l_limits = cf.filter_limits(self.l_limits, len(pmtrwf[0][0]))
        ## Anti-correlated
        self.d_limits = cf.filter_limits(self.d_limits, len(pmtrwf[0][0]))
        ###

        ## Where we'll be saving the binned info for each channel
        bpmtspe = np.zeros((len(self.pmt_active), len(self.histbins)-1),
                            dtype=np.int)
        bpmtdar = np.zeros((len(self.pmt_active), len(self.histbins)-1),
                            dtype=np.int)

        for evt in range(NEVT):
            self.conditional_print(evt, self.cnt.n_events_tot)

            what_next = self.event_range_step()
            if what_next is EventLoop.skip_this_event: continue
            if what_next is EventLoop.terminate_loop : break
            self.cnt.n_events_tot += 1

            ## Get the pedestal subtracted ADC WF
            pmtadc = self.pmt_processing[self.pmode](pmtrwf[evt])

            ## Values of the integrals, [:, ::2] since those between limits
            ## automatically calculated too.
            # LED correlated.
            led_ints = cf.spaced_integrals(pmtadc, self.l_limits)[:, ::2]
            bpmtspe += cf.bin_waveforms(led_ints, self.histbins)

            # Dark (anti-correlated)
            dar_ints = cf.spaced_integrals(pmtadc, self.d_limits)[:, ::2]
            bpmtdar += cf.bin_waveforms(dar_ints, self.histbins)

            # write stuff
            event, timestamp = self.event_and_timestamp(evt, events_info)
            write.run_and_event(self.run_number, event, timestamp)
        write.led(bpmtspe)
        write.dar(bpmtdar)


    def get_writers(self, h5out):
        bin_centres = shift_to_bin_centers(self.histbins)
        HIST        = partial(hist_writer,
                              h5out,
                              group_name  = 'HIST',
                              n_sensors   = len(self.pmt_active),
                              n_bins      = len(bin_centres),
                              bin_centres = bin_centres)
        writers = Namespace(
            run_and_event = run_and_event_writer(h5out),
            led  = HIST(table_name='pmtspe'),
            dar  = HIST(table_name='pmtdar'))

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
