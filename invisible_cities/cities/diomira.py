"""
code: diomira.py
description: simulation of the response of the energy and tracking planes
for the NEXT detector.
credits: see ic_authors_and_legal.rst in /doc

last revised: JJGC, 10-July-2017
"""

from functools import partial
from argparse  import Namespace

import numpy as np


from .. io.mcinfo_io            import mc_info_writer
from .. io.run_and_event_io     import run_and_event_writer
from .. io.rwf_io               import rwf_writer
from .. filters.trigger_filters import TriggerFilter
from .. reco                    import wfm_functions as wfm
from .  base_cities             import MonteCarloCity
from .  base_cities             import EventLoop



class Diomira(MonteCarloCity):
    """
    The city of DIOMIRA simulates:
    1. the response of the energy and traking plane sensors.
    2. the response of the trigger
    """

    parameters = tuple("""filter_padding trigger_type""".split())

    def __init__(self, **kwds):
        """Diomira Init:
        1. inits base city
        2. inits counters
        3. get sensor parameters
        4. inits the noise sampler and the sipms thesholds
        5. instantiates the Trigger Filter.
        """
        super().__init__(**kwds)
        conf = self.conf

        self.cnt.init(n_events_tot = 0,
                      nevt_out     = 0)

        self.sipm_noise_cut = conf.sipm_noise_cut

        self.filter_padding = conf.filter_padding

        # thresholds in adc counts with baselines
        self.sipms_thresholds = self.sipm_noise_cut *  self.sipm_adc_to_pes
        self.sipms_thresholds[:, np.newaxis] += self.noise_sampler.baselines

        self.trigger_filter   = TriggerFilter(self.trigger_params)
        self.trigger_type = conf.trigger_type

    def event_loop(self, NEVT, dataVectors):
        """
        loop over events:
        1. simulate pmt and sipm response
        2. write RWF and CWF for PMTs
        3. write SiPMR
        4. write event info and MC info to file
        """

        write       = self.writers
        pmtrd       = dataVectors.pmt
        sipmrd      = dataVectors.sipm
        mc_info     = dataVectors.mc
        events_info = dataVectors.events

        for evt in range(NEVT):
            self.conditional_print(self.cnt.n_events_tot, self.cnt.nevt_out)

            # Count events in and break if necessary before filtering
            what_next = self.event_range_step()
            if what_next is EventLoop.skip_this_event: continue
            if what_next is EventLoop.terminate_loop : break
            self.cnt.n_events_tot += 1

            # Simulate detector response
            dataPMT, blrPMT = self.simulate_pmt_response(evt, pmtrd)
            dataSiPM_noisy  = self.simulate_sipm_response(evt, sipmrd)

            dataSiPM        = wfm.noise_suppression(dataSiPM_noisy       ,
                                                    self.sipms_thresholds,
                                                    self.filter_padding  )

            RWF = dataPMT.astype(np.int16)
            BLR = blrPMT.astype(np.int16)

            # simulate trigger
            if self.trigger_type == 'S2':
                peak_data = self.emulate_trigger(RWF)
                # filter events as a function of trigger
                if not self.trigger_filter(peak_data):
                    continue

            self.cnt.nevt_out += 1

            #write
            event_number, timestamp = self.event_and_timestamp(evt, events_info)
            if self.monte_carlo:
                write.mc(mc_info, event_number)

            write.run_and_event(self.run_number, event_number, timestamp)
            write.rwf(RWF)
            write.cwf(BLR)
            write.sipm(dataSiPM)

    def write_parameters(self, h5out):
        """Write deconvolution parameters to output file"""
        self.write_simulation_parameters_table(h5out)

    def get_writers(self, h5out):
        RWF = partial(rwf_writer,  h5out,   group_name='RD')
        writers = Namespace(
            run_and_event = run_and_event_writer(h5out),
            mc            = mc_info_writer(h5out) if self.monte_carlo else None,
        # 3 variations on the  RWF writer theme
            rwf  = RWF(table_name='pmtrwf' , n_sensors=self.sp.NPMT , waveform_length=self.sp.PMTWL),
            cwf  = RWF(table_name='pmtblr' , n_sensors=self.sp.NPMT , waveform_length=self.sp.PMTWL),
            sipm = RWF(table_name='sipmrwf', n_sensors=self.sp.NSIPM, waveform_length=self.sp.SIPMWL),
        )
        return writers

    def display_IO_info(self):
        super().display_IO_info()
        print(self.sp)
