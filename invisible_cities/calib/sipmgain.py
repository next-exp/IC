"""
code: sipmgain.py
description: Generates spectra using integrals
at configuration file specified regions (will fit eventually)
credits: see ic_authors_and_legal.rst in /doc

last revised:
"""
from argparse  import Namespace
from functools import partial

import numpy  as np

from .. core.system_of_units_c import units
from .                         import calib_functions         as cf
from .. reco                   import calib_sensors_functions as csf
from .. io  .         hist_io  import          hist_writer
from .. io  .run_and_event_io  import run_and_event_writer
from .. core.core_functions    import shift_to_bin_centers

from .. cities.base_cities import CalibratedCity
from .. cities.base_cities import EventLoop


class Sipmgain(CalibratedCity):
    """
    Generates spectra using integrals in each buffer
    according to start, end, width and repetition
    given in the configuration file.
    input: RAW data (pulsed LED normally)
    output: Spectra (spe spectra)
    TODO: add fits and output results.
    """
    parameters = tuple("""proc_mode min_bin max_bin bin_width number_integrals
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

        ## The bin range for the histograms
        conf = self.conf

        ## the mode of processing (temp).
        if conf.proc_mode not in ("subtract_mode", "subtract_median"):
            raise ValueError(f"Unrecognized processing mode: {conf.proc_mode}")
        self.sipm_processing = csf.sipm_processing[conf.proc_mode]

        self.histbins = np.arange(conf.min_bin, conf.max_bin, conf.bin_width)

        # The integral limits
        sampling = 1 * units.mus
        l_limits, d_limits = cf.integral_limits(sampling             ,
                                                conf.number_integrals,
                                                conf.integral_start  ,
                                                conf.integral_width  ,
                                                conf.integrals_period)

        # Make sure the limits are valid for these data.
        # Need to be in range. Should probably give a warning.
        # If the start of a limit is out of range the end must be
        # removed too and vice versa
        wf_length = self.sp.SIPMWL
        self.l_limits = cf.filter_limits(l_limits, wf_length) #      correlated
        self.d_limits = cf.filter_limits(d_limits, wf_length) # anti-correlated

    def event_loop(self, NEVT, dataVectors):
        """actions:
        1. loops over all the events in each file.
        2. write event/run to file
        3. write histogram info to file (to reduce memory usage)
        """
        write       = self.writers
        sipmrwf     = dataVectors.sipm
        events_info = dataVectors.events

        ## Where we'll be saving the binned info for each channel
        shape     = sipmrwf.shape[1], len(self.histbins) - 1
        sipm_spe  = np.zeros(shape, dtype=np.int)
        sipm_dark = np.zeros(shape, dtype=np.int)

        for evt in range(NEVT):
            self.conditional_print(evt, self.cnt.n_events_tot)

            what_next = self.event_range_step()
            if what_next is EventLoop.skip_this_event: continue
            if what_next is EventLoop.terminate_loop : break
            self.cnt.n_events_tot += 1

            ## Get the pedestal subtracted ADC WF
            sipm_adc = self.sipm_processing(sipmrwf[evt])

            # Values of the integrals, [:, ::2] since those between limits
            # automatically calculated too.

            # LED (correlated)
            led_ints  = cf.spaced_integrals(sipm_adc, self.l_limits)[:, ::2]
            sipm_spe += cf.bin_waveforms(led_ints, self.histbins)

            # Dark (anti-correlated)
            dark_ints  = cf.spaced_integrals(sipm_adc, self.d_limits)[:, ::2]
            sipm_dark += cf.bin_waveforms(dark_ints, self.histbins)

            event, timestamp = self.event_and_timestamp(evt, events_info)
            write.run_and_event(self.run_number, event, timestamp)

        write.led (sipm_spe )
        write.dark(sipm_dark)


    def get_writers(self, h5out):
        ## Copy sensor info (good for non DB tests)
        cf.copy_sensor_table(self.input_files[0], h5out)
        
        bin_centres = shift_to_bin_centers(self.histbins)
        HIST        = partial(hist_writer,
                              h5out,
                              group_name  = 'HIST',
                              n_sensors   = self.sp.NSIPM,
                              bin_centres = bin_centres)

        writers = Namespace(run_and_event = run_and_event_writer(h5out),
                            led           = HIST(table_name='sipm_spe' ),
                            dark          = HIST(table_name='sipm_dark'))

        return writers

    def write_parameters(self, h5out):
        pass

    def display_IO_info(self):
        super().display_IO_info()
        print(self.sp)
