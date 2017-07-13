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

import tables as tb

from .. core.configure         import configure
from .. core.random_sampling   import NoiseSampler as SiPMsNoiseSampler
from .. core.system_of_units_c import units
from .. core.exceptions        import ParameterNotSet

from .. io.mc_io            import mc_track_writer
from .. io.run_and_event_io import run_and_event_writer
from .. io.rwf_io           import rwf_writer
from .. io.fee_io           import write_FEE_table

from .. reco        import wfm_functions as wfm
from .. reco        import tbl_functions as tbl
from .. evm.nh5     import FEE
from .. evm.nh5     import RunInfo
from .. evm.nh5     import EventInfo

from .  base_cities import SensorResponseCity


class Diomira(SensorResponseCity):
    """
    The city of DIOMIRA simulates the response of the energy and
    traking plane sensors.

    """

    def go(self):
        t0 = time()
        nevt_tot = self.run()
        t1 = time()
        dt = t1 - t0
        print("run {} evts in {} s, time/event = {}".format(nevt_tot, dt, dt/nevt_tot))

    def run(self):

        self.display_IO_info()
        sp = self.get_sensor_rd_params(self.input_files[0])
        print(sp)

        # Create instance of the noise sampler
        self.noise_sampler = SiPMsNoiseSampler(self.run_number, sp.SIPMWL, True)

        # thresholds in adc counts
        self.sipms_thresholds = (self.sipm_noise_cut
                              *  self.sipm_adc_to_pes)

        # loop over input files
        with tb.open_file(self.output_file, "w",
                          filters = tbl.filters(self.compression)) as h5out:

            # Create writers
            RWF = partial(rwf_writer,  h5out,   group_name='RD')
            writers = Namespace(
                run_and_event = run_and_event_writer(h5out),
                mc            =      mc_track_writer(h5out) if self.monte_carlo else None,
                # 3 variations on the  RWF writer theme
                rwf  = RWF(table_name='pmtrwf' , n_sensors=sp.NPMT , waveform_length=sp.PMTWL),
                cwf  = RWF(table_name='pmtblr' , n_sensors=sp.NPMT , waveform_length=sp.PMTWL),
                sipm = RWF(table_name='sipmrwf', n_sensors=sp.NSIPM, waveform_length=sp.SIPMWL),
            )

            # Create and store Front-End Electronics parameters (for the PMTs)
            write_FEE_table(h5out)

            n_events_tot = self._file_loop(writers)
        return n_events_tot

    def _file_loop(self, writers):
        n_events_tot = 0
        for filename in self.input_files:
            first_event_no = self.event_number_from_input_file_name(filename)
            print("Opening file {filename} with first event no {first_event_no}"
                  .format(**locals()))
            with tb.open_file(filename, "r") as h5in:
                # NEVT is the total number of events in pmtrd and sipmrd
                # pmtrd = pmrtd[events][NPMT][rd_waveform]
                # sipmrd = sipmrd[events][NPMT][rd_waveform]
                NEVT, pmtrd, sipmrd = self.get_rd_vectors(h5in)
                events_info = self.get_run_and_event_info(h5in)
                n_events_tot = self._event_loop(NEVT, pmtrd, sipmrd, events_info,
                                                writers,
                                                n_events_tot, h5in, first_event_no)
        return n_events_tot


    def _event_loop(self, NEVT, pmtrd, sipmrd, events_info,
                    write,
                    n_events_tot, h5in, first_event_no):

        for evt in range(NEVT):
            # Simulate detector response
            dataPMT, blrPMT = self.simulate_pmt_response(evt, pmtrd)
            dataSiPM_noisy = self.simulate_sipm_response(evt, sipmrd, self.noise_sampler)
            dataSiPM = wfm.noise_suppression(dataSiPM_noisy, self.sipms_thresholds)

            event_number, timestamp = self.event_and_timestamp(evt, events_info)
            local_event_number = event_number + first_event_no

            write.mc(h5in.root.MC.MCTracks, local_event_number)
            write.run_and_event(self.run_number, local_event_number, timestamp)
            write.rwf(dataPMT.astype(int))
            write.cwf( blrPMT.astype(int))
            write.sipm(dataSiPM)

            n_events_tot += 1
            self.conditional_print(evt, n_events_tot)
            if self.max_events_reached(n_events_tot):
                break
        return n_events_tot
