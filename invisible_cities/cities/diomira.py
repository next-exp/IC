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
from .. core.mctrk_functions   import mc_track_writer
from .. core.exceptions        import ParameterNotSet

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

    def run(self, nmax):
        """
        Run the machine
        nmax is the max number of events to run
        """

        n_events_tot = 0
        # run the machine only if in a legal state
        self.check_files()
        if not self.sipm_noise_cut:
            raise ParameterNotSet('must set sipm_noise_cut before running')

        self.display_IO_info(nmax)

        # loop over input files
        first = False
        with tb.open_file(self.output_file, "w",
                          filters = tbl.filters(self.compression)) as h5out:

            mctracks_writer = mc_track_writer(h5out)

            for ffile in self.input_files:

                print("Opening file {}".format(ffile))
                filename = ffile

                with tb.open_file(filename, "r") as h5in:
                    pmtrd  = h5in.root.pmtrd
                    sipmrd = h5in.root.sipmrd

                    NEVT, NPMT, PMTWL = pmtrd.shape
                    PMTWL_FEE = int(PMTWL / self.FE_t_sample)
                    NEVENTS_DST, NSIPM, SIPMWL = sipmrd.shape
                    sensor_param = SensorParams(NPMT   = NPMT,
                                                PMTWL  = PMTWL_FEE,
                                                NSIPM  = NSIPM,
                                                SIPMWL = SIPMWL)

                    # last row copied from MCTracks table
                    mctrack_row = 0

                    print("Events in file = {}".format(NEVENTS_DST))

                    if not first:
                        # print configuration, create vectors
                        # init SiPM noiser, store FEE table
                        self.print_configuration(sensor_param, PMTWL)
                        self.set_output_store(h5out, nmax, sensor_param)

                        # add run number
                        run = self.runInfot.row
                        run['run_number'] = self.run_number
                        run.append()

                        # Create instance of the noise sampler
                        self.noise_sampler = SiPMsNoiseSampler(SIPMWL, True)
                        # thresholds in adc counts
                        self.sipms_thresholds = (self.sipm_noise_cut
                                              *  self.sipm_adc_to_pes)
                        # store FEE parameters in table
                        self.store_FEE_table()

                        first = True

                    # loop over events in the file. Break when nmax is reached
                    for evt in range(NEVT):
                        # copy corresponding MCTracks to output MCTracks table
                        mctracks_writer(h5in.root.MC.MCTracks,
                                        n_events_tot, self.first_evt)

                        # simulate PMT and SiPM response
                        # RWF and BLR
                        dataPMT, blrPMT = self.simulate_pmt_response(
                                        evt, pmtrd)
                        # append the data to pytable vectors
                        self.pmtrwf.append(
                            dataPMT.astype(int).reshape(1, NPMT, PMTWL_FEE))
                        self.pmtblr.append(
                             blrPMT.astype(int).reshape(1, NPMT, PMTWL_FEE))
                        # SiPMs
                        dataSiPM = self.simulate_sipm_response(
                                    evt, sipmrd, self.noise_sampler)
                        # return a noise-supressed waveform
                        dataSiPM = wfm.noise_suppression(
                                    dataSiPM, self.sipms_thresholds)

                        self.sipmrwf.append(
                            dataSiPM.astype(int).reshape(1, NSIPM, SIPMWL))

                        # add evt number
                        self.write_evt_number(evt)

                        n_events_tot +=1
                        if n_events_tot % self.nprint == 0:
                            print('event in file = {}, total = {}'
                                  .format(evt, n_events_tot))

                        if n_events_tot >= nmax and nmax > -1:
                            print('reached maximum number of events (={})'
                                  .format(nmax))
                            break

        self.pmtrwf.flush()
        self.sipmrwf.flush()
        self.pmtblr.flush()

        return n_events_tot

    def write_evt_number(self, evt):
        evt_row = self.evtsInfot.row
        evt_row['evt_number'] = evt + self.first_evt
        evt_row['timestamp'] = 0
        evt_row.append()


    def set_output_store(self, h5out, nmax, sp):

        # RD group
        RD = h5out.create_group(h5out.root, "RD")
        # MC group
        MC = h5out.root.MC
        # create a table to store Energy plane FEE
        self.fee_table = h5out.create_table(MC, "FEE", FEE,
                          "EP-FEE parameters",
                          tbl.filters("NOCOMPR"))
        # create vectors
        self.pmtrwf = h5out.create_earray(
                    RD,
                    "pmtrwf",
                    atom = tb.Int16Atom(),
                    shape = (0, sp.NPMT, sp.PMTWL),
                    expectedrows = nmax,
                    filters = tbl.filters(self.compression))

        self.pmtblr = h5out.create_earray(
                    RD,
                    "pmtblr",
                    atom = tb.Int16Atom(),
                    shape = (0, sp.NPMT, sp.PMTWL),
                    expectedrows = nmax,
                    filters = tbl.filters(self.compression))

        self.sipmrwf = h5out.create_earray(
                    RD,
                    "sipmrwf",
                    atom = tb.Int16Atom(),
                    shape = (0, sp.NSIPM, sp.SIPMWL),
                    expectedrows = nmax,
                    filters = tbl.filters(self.compression))

        # run group
        RUN = h5out.create_group(h5out.root, "Run")
        self.runInfot = h5out.create_table(RUN, "RunInfo", RunInfo,
                          "Run info",
                          tbl.filters("NOCOMPR"))
        self.evtsInfot = h5out.create_table(RUN, "events", EventInfo,
                          "Events info",
                          tbl.filters("NOCOMPR"))

    def print_configuration(self, sp, PMTWL):
        print_configuration({"# PMT"        : sp.NPMT,
                             "PMT WL"       : PMTWL,
                             "PMT WL (FEE)" : sp.PMTWL,
                             "# SiPM"       : sp.NSIPM,
                             "SIPM WL"      : sp.SIPMWL})



def DIOMIRA(argv=sys.argv):
    """DIOMIRA DRIVER"""

    CFP = configure(argv)
    files_in = glob(CFP.FILE_IN)
    files_in.sort()

    fpp = Diomira(first_evt      = CFP.FIRST_EVT,
                  files_in       = files_in,
                  file_out       = CFP.FILE_OUT,
                  sipm_noise_cut = CFP.NOISE_CUT)

    nevts = CFP.NEVENTS if not CFP.RUN_ALL else -1
    t0 = time()
    nevt = fpp.run(nmax=nevts)
    t1 = time()
    dt = t1 - t0

    print("run {} evts in {} s, time/event = {}".format(nevt, dt, dt/nevt))

    return nevts, nevt

if __name__ == "__main__":
    DIOMIRA(sys.argv)
