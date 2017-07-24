"""
code: diomira.py
description: simulation of the response of the energy and tracking planes
for the NEXT detector.
credits: see ic_authors_and_legal.rst in /doc

last revised: JJGC, 10-July-2017
"""
import sys

from glob      import glob
from time      import time
from functools import partial
from argparse  import Namespace

import numpy as np
import tables as tb

from .. core.system_of_units_c  import units
from .. core.exceptions         import ParameterNotSet

from .. io.mc_io                import mc_track_writer
from .. io.run_and_event_io     import run_and_event_writer
from .. io.rwf_io               import rwf_writer
from .. io.fee_io               import write_FEE_table
from .. filters.trigger_filters import TriggerFilter

from .. reco                    import wfm_functions as wfm
from .. reco                    import tbl_functions as tbl
from .. reco.sensor_functions   import convert_channel_id_to_IC_id
from .. reco                    import tbl_functions    as tbl
from .. reco                    import peak_functions_c as cpf
from .. evm.nh5                 import FEE
from .. evm.nh5                 import RunInfo
from .. evm.nh5                 import EventInfo
from .. evm.ic_containers       import PeakData

from .. database                import load_db          as db
from .. types.ic_types          import minmax
from .  base_cities             import MonteCarloCity
from .. filters.trigger_filters import TriggerFilter

class Diomira(MonteCarloCity):
    """
    The city of DIOMIRA simulates:
    1. the response of the energy and traking plane sensors.
    2. the response of the trigger
    """

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

        self.cnt.set_name('diomira')
        self.cnt.set_counter('nmax', value=self.conf.nmax)
        self.cnt.init_counters(('n_events_tot', 'nevt_out'))

        self.sipm_noise_cut   = conf.sipm_noise_cut

        # thresholds in adc counts
        self.sipms_thresholds = self.sipm_noise_cut *  self.sipm_adc_to_pes

        self.trigger_filter   = TriggerFilter(self.trigger_params)


    def event_loop(self, NEVT, first_event_no, dataVectors):
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
        mc_tracks   = dataVectors.mc
        events_info = dataVectors.events

        for evt in range(NEVT):
            # Count events in and break if necessary before filtering
            if self.max_events_reached(self.cnt.counter_value('n_events_tot')):
                break
            else:
                self.cnt.increment_counter('n_events_tot')

            # Simulate detector response
            dataPMT, blrPMT = self.simulate_pmt_response(evt, pmtrd,
                                                         self.sipm_adc_to_pes)
            dataSiPM_noisy = self.simulate_sipm_response(evt, sipmrd,
                                                         self.noise_sampler,
                                                         self.sipm_adc_to_pes)
            dataSiPM = wfm.noise_suppression(dataSiPM_noisy, self.sipms_thresholds)
            RWF = dataPMT.astype(np.int16)
            BLR = blrPMT.astype(np.int16)

            # simulate trigger
            peak_data = self.emulate_trigger(RWF)
            # filter events as a function of trigger

            if not self.trigger_filter(peak_data):
                continue
            self.cnt.increment_counter('nevt_out')

            #write
            event_number, timestamp = self.event_and_timestamp(evt, events_info)
            local_event_number = event_number + first_event_no

            if self.monte_carlo:
                write.mc(mc_tracks, local_event_number)

            write.run_and_event(self.run_number, local_event_number, timestamp)
            write.rwf(RWF)
            write.cwf(BLR)
            write.sipm(dataSiPM)

            self.conditional_print(self.cnt.counter_value('n_events_tot'),
            self.cnt.counter_value('nevt_out'))

    def write_parameters(self, h5out):
        """Write deconvolution parameters to output file"""
        self.write_simulation_parameters_table(h5out)

    def get_writers(self, h5out):
        """Get the writers needed by Diomira"""

        RWF = partial(rwf_writer,  h5out,   group_name='RD')
        writers = Namespace(
            run_and_event = run_and_event_writer(h5out),
            mc            =      mc_track_writer(h5out) if self.monte_carlo else None,
        # 3 variations on the  RWF writer theme
            rwf  = RWF(table_name='pmtrwf' , n_sensors=self.sp.NPMT , waveform_length=self.sp.PMTWL),
            cwf  = RWF(table_name='pmtblr' , n_sensors=self.sp.NPMT , waveform_length=self.sp.PMTWL),
            sipm = RWF(table_name='sipmrwf', n_sensors=self.sp.NSIPM, waveform_length=self.sp.SIPMWL),
        )
        return writers

def display_IO_info(self):
    """display info"""
    super().display_IO_info()
    print(self.sp)
