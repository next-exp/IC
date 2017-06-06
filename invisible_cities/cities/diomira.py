"""
code: diomira.py
description: simulation of the response of the energy and tracking planes
for the NEXT detector.
author: J.J. Gomez-Cadenas
IC core team: Jacek Generowicz, JJGC, G. Martinez, J.A. Hernando, J.M Benlloch
package: invisible cities. See release notes and licence
last changed: 09-11-2017
"""
import sys
from glob import glob
from time import time

import tables as tb

from .. core.configure         import configure
from .. core.configure         import print_configuration
from .. core.random_sampling   import NoiseSampler as SiPMsNoiseSampler
from .. core.system_of_units_c import units
from .. core.exceptions        import ParameterNotSet

from .. io.mc_io            import mc_track_writer
from .. io.run_and_event_io import run_and_event_writer
from .. io.rwf_io           import rwf_writer

from .. reco        import wfm_functions as wfm
from .. reco        import tbl_functions as tbl
from .. reco.params import SensorParams
from .. reco.nh5    import FEE
from .. reco.nh5    import RunInfo
from .. reco.nh5    import EventInfo

from .  base_cities import SensorResponseCity


class Diomira(SensorResponseCity):
    """
    The city of DIOMIRA simulates the response of the energy and
    traking plane sensors.

    """
    def __init__(self,
                 run_number     = 0,
                 files_in       = None,
                 file_out       = None,
                 sipm_noise_cut = 3 * units.pes,
                 first_evt      = 0):
        """
        -1. Inits the machine
        -2. Loads the data base to access calibration data
        -3. Sets all switches to default value
        """

        SensorResponseCity.__init__(self,
                                    run_number     = run_number,
                                    files_in       = files_in,
                                    file_out       = file_out,
                                    sipm_noise_cut = sipm_noise_cut,
                                    first_evt      = first_evt)
        self.check_files()


    def run(self, nmax):

        self.display_IO_info(nmax)
        sensor_params = self.get_sensor_rd_params(self.input_files[0])
        self.print_configuration(sensor_params)

        # Create instance of the noise sampler
        self.noise_sampler = SiPMsNoiseSampler(sensor_params.SIPMWL, True)
        # thresholds in adc counts
        self.sipms_thresholds = (self.sipm_noise_cut
                              *  self.sipm_adc_to_pes)

        # loop over input files
        first = False
        with tb.open_file(self.output_file, "w",
                          filters = tbl.filters(self.compression)) as h5out:

            # Create writers
            write_run_and_event = run_and_event_writer(h5out)
            write_mc            =      mc_track_writer(h5out) if self.monte_carlo else None
            write_rwf           =           rwf_writer(h5out,
                                                       group_name      = 'RD',
                                                       table_name      = 'pmtrwf',
                                                       n_sensors       = sensor_params.NPMT,
                                                       waveform_length = sensor_params.PMTWL)
            write_cwf           =           rwf_writer(h5out,
                                                       group_name      = 'RD',
                                                       table_name      = 'pmtblr',
                                                       n_sensors       = sensor_params.NPMT,
                                                       waveform_length = sensor_params.PMTWL)
            write_sipm          =           rwf_writer(h5out,
                                                       group_name      = 'RD',
                                                       table_name      = 'sipmrwf',
                                                       n_sensors       = sensor_params.NSIPM,
                                                       waveform_length = sensor_params.SIPMWL)
            # Create and store Front-End Electronics parameters (for the PMTs)
            self._create_FEE_table(h5out)
            self.  store_FEE_table()

            n_events_tot = self._main_loop(write_run_and_event,
                                           write_mc, write_rwf, write_cwf, write_sipm, nmax)
        return n_events_tot

    # TODO rename as file loop
    # TODO pack writers into a sensible container
    def _main_loop(self, write_run_and_event, write_mc, write_rwf, write_cwf, write_sipm, nmax):
        n_events_tot = 0
        for filename in self.input_files:
            print("Opening file {}".format(filename))
            with tb.open_file(filename, "r") as h5in:
                NEVT, pmtrd, sipmrd = self.get_rd_vectors(h5in)
                events_info = self.get_run_and_event_info(h5in)
                n_events_tot = self._event_loop(NEVT, pmtrd, sipmrd, events_info,
                                                write_run_and_event, write_mc, write_rwf, write_cwf, write_sipm,
                                                nmax, n_events_tot, h5in)
        return n_events_tot

    def _event_loop(self, NEVT, pmtrd, sipmrd, events_info,
                    write_run_and_event, write_mc, write_rwf, write_cwf, write_sipm,
                    nmax, n_events_tot, h5in):

        for evt in range(NEVT):
            write_mc(h5in.root.MC.MCTracks, n_events_tot)
            event, timestamp = self.event_and_timestamp(evt, events_info)
            write_run_and_event(self.run_number, event, timestamp)
            # simulate PMT and SiPM response
            # RWF and BLR
            dataPMT, blrPMT = self.simulate_pmt_response(evt, pmtrd)
            dataSiPM_noisy = self.simulate_sipm_response(evt, sipmrd, self.noise_sampler)
            dataSiPM = wfm.noise_suppression(dataSiPM_noisy, self.sipms_thresholds)
            # append the data to pytable vectors
            write_rwf(dataPMT.astype(int))
            write_cwf( blrPMT.astype(int))
            write_sipm(dataSiPM)
            n_events_tot += 1
            self.conditional_print(evt, n_events_tot)
            if self.max_events_reached(nmax, n_events_tot):
                break
        return n_events_tot

    def _create_FEE_table(self, h5out):
        # create a table to store Energy plane FEE
        FEE_group = h5out.create_group(h5out.root, "FEE")
        self.fee_table = h5out.create_table(FEE_group, "FEE", FEE,
                          "EP-FEE parameters",
                          tbl.filters("NOCOMPR"))


def DIOMIRA(argv=sys.argv):
    """DIOMIRA DRIVER"""

    CFP = configure(argv)
    files_in = glob(CFP.FILE_IN)
    files_in.sort()

    diomira = Diomira(first_evt      = CFP.FIRST_EVT,
                      files_in       = files_in,
                      file_out       = CFP.FILE_OUT,
                      sipm_noise_cut = CFP.NOISE_CUT)

    nevts = CFP.NEVENTS if not CFP.RUN_ALL else -1
    t0 = time()
    nevt = diomira.run(nmax=nevts)
    t1 = time()
    dt = t1 - t0

    if nevt > 0:
        print("run {} evts in {} s, time/event = {}".format(nevt, dt, dt/nevt))
