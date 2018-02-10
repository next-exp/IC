"""
code: pmtgain.py
description: Generates spectra using integrals
at configuration file specified regions (will fit eventually)
credits: see ic_authors_and_legal.rst in /doc

last revised:
"""
from argparse  import Namespace
from functools import partial

import numpy  as np

from .                         import      calib_functions as cf
from .. reco                   import calib_sensors_functions as csf
from .. io   .         hist_io import          hist_writer
from .. io   .run_and_event_io import run_and_event_writer
from .. icaro.hst_functions    import shift_to_bin_centers

from .. core  .system_of_units_c import units
from .. cities.base_cities       import CalibratedCity
from .. cities.base_cities       import EventLoop
from .. sierpe                   import fee


class Pmtgain(CalibratedCity):
    """
    Generates spectra using integrals in each buffer
    according to start, end, width and repetition
    given in the configuration file.
    input: RAW data (pulsed LED normally)
    output: Spectra (spe spectra)
    TODO: add fits and output results.
    """
    parameters = tuple("""proc_mode
                          min_bin max_bin bin_width number_integrals
                          integral_start integral_width integrals_period""".split())

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

        # the mode of processing (temp).
        # Can be 'gain': just deconvolve and subtract ped
        # or 'gainM': deconvolve, subtract ped, subtract MAU
        if   conf.proc_mode == "gain":
            self.pmt_processing = self.deconv_pmt
        elif conf.proc_mode == "gain_mau":
            self.pmt_processing = self.deconv_pmt_mau
        elif conf.proc_mode == "gain_nodeconv":
            self.pmt_processing = self.without_deconv
        else:
            raise ValueError(f"Unrecognized processing mode: {conf.proc_mode}")

        self.histbins = np.arange(conf.min_bin, conf.max_bin, conf.bin_width)

        # The integral limits
        sampling = fee.t_sample
        l_limits, d_limits = cf.integral_limits(sampling             ,
                                                conf.number_integrals,
                                                conf.integral_start  ,
                                                conf.integral_width  ,
                                                conf.integrals_period)

        # Make sure the limits are valid for these data.
        # Need to be in range. Should probably give a warning.
        # If the start of a limit is out of range the end must be
        # removed too and vice versa
        wf_length = self.sp.PMTWL
        self.l_limits = cf.filter_limits(l_limits, wf_length) #      correlated
        self.d_limits = cf.filter_limits(d_limits, wf_length) # anti-correlated

    def deconv_pmt_mau(self, RWF):
        CWF = self.deconv_pmt(RWF)
        return csf.pmt_subtract_mau(CWF, n_MAU=self.n_MAU)

    def without_deconv(self, RWF):
        CWF = csf.subtract_mode(RWF)
        return CWF[self.pmt_active_list]

    def event_loop(self, NEVT, dataVectors):
        """actions:
        1. loops over all the events in each file.
        2. write event/run to file
        3. write histogram info to file (to reduce memory usage)
        """
        write       = self.writers
        pmtrwf      = dataVectors.pmt
        events_info = dataVectors.events

        ## Where we'll be saving the binned info for each channel
        shape    = len(self.pmt_active_list), len(self.histbins) - 1
        pmt_spe  = np.zeros(shape, dtype=np.int)
        pmt_dark = np.zeros(shape, dtype=np.int)

        for evt in range(NEVT):
            self.conditional_print(evt, self.cnt.n_events_tot)

            what_next = self.event_range_step()
            if what_next is EventLoop.skip_this_event: continue
            if what_next is EventLoop.terminate_loop : break
            self.cnt.n_events_tot += 1

            # Get the pedestal subtracted ADC WF
            pmt_adc = self.pmt_processing(pmtrwf[evt])

            # Values of the integrals, [:, ::2] since those between limits
            # automatically calculated too.

            # LED (correlated)
            led_ints  = cf.spaced_integrals(pmt_adc, self.l_limits)[:, ::2]
            pmt_spe  += cf.bin_waveforms(led_ints, self.histbins)

            # Dark (anti-correlated)
            dark_ints  = cf.spaced_integrals(pmt_adc, self.d_limits)[:, ::2]
            pmt_dark  += cf.bin_waveforms(dark_ints, self.histbins)

            event, timestamp = self.event_and_timestamp(evt, events_info)
            write.run_and_event(self.run_number, event, timestamp)

        write.led (pmt_spe )
        write.dark(pmt_dark)


    def get_writers(self, h5out):
        ## Copy sensor info (good for non DB tests)
        cf.copy_sensor_table(self.input_files[0], h5out)

        bin_centres = shift_to_bin_centers(self.histbins)
        HIST        = partial(hist_writer,
                              h5out,
                              group_name  = 'HIST',
                              n_sensors   = len(self.pmt_active_list),
                              bin_centres = bin_centres)

        writers = Namespace(run_and_event = run_and_event_writer(h5out),
                            led           = HIST(table_name='pmt_spe' ),
                            dark          = HIST(table_name='pmt_dark'))

        return writers

    def write_parameters(self, h5out):
        pass

    def display_IO_info(self):
        super().display_IO_info()
        print(self.sp)
