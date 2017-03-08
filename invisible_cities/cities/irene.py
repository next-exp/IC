from __future__ import print_function
import sys

from glob import glob
from time import time

import numpy  as np
import tables as tb


from invisible_cities.core.configure import configure, print_configuration
from invisible_cities.core.system_of_units_c import units

from invisible_cities.reco.pmap_io import pmap_writer, S12, S2Si
from invisible_cities.core.system_of_units_c import units
from invisible_cities.cities.base_cities import PmapCity, SensorParams
from invisible_cities.cities.base_cities import S12Params as S12P
from invisible_cities.core.mctrk_functions import MCTrackWriter


class Irene(PmapCity):
    """Perform fast processing from raw data to pmaps.

    Raw data pmtrwf and sipmrwf.
    It is optimized for speed (use of CYTHON functions) and intended
    for fast processing of data.
    """
    def __init__(self,
                 run_number  = 0,
                 files_in    = None,
                 file_out    = None,
                 compression = 'ZLIB4',
                 nprint      = 10000,
                 n_baseline  = 28000,
                 thr_trigger = 5.0 * units.adc,
                 n_MAU       = 100,
                 thr_MAU     = 3.0 * units.adc,
                 thr_csum_s1 = 0.2 * units.adc,
                 thr_csum_s2 = 1.0 * units.adc,
                 n_MAU_sipm  = 100,
                 thr_sipm    = 5.0 * units.pes,
                 s1_params   = None,
                 s2_params   = None,
                 thr_sipm_s2 = 30. * units.pes):

        PmapCity.__init__(self,
                          run_number  = run_number,
                          files_in    = files_in,
                          file_out    = file_out,
                          compression = compression,
                          nprint      = nprint,
                          n_baseline  = n_baseline,
                          thr_trigger = thr_trigger,
                          n_MAU       = n_MAU,
                          thr_MAU     = thr_MAU,
                          thr_csum_s1 = thr_csum_s1,
                          thr_csum_s2 = thr_csum_s2,
                          n_MAU_sipm  = n_MAU_sipm,
                          thr_sipm    = thr_sipm,
                          s1_params   = s1_params,
                          s2_params   = s2_params,
                          thr_sipm_s2 = thr_sipm_s2)

        # counter for empty events
        self.empty_events = 0 # counts empty events in the energy plane

    def run(self, nmax, print_empty=True):
        """
        Run Irene
        nmax is the max number of events to run
        if print_empty = True, count the number of empty events
        """

        n_events_tot = 0
        self.check_files()

        # check that S1 and S2 params are defined
        if (not self.s1_params) or (not self.s2_params):
            raise IOError('must set S1/S2 parameters before running')

        self.display_IO_info(nmax)

        first = False
        with pmap_writer(self.output_file, self.compression) as write:
            if self.monte_carlo:
                mctrack_writer = MCTrackWriter(write.file)
            # loop over input files
            for ffile in self.input_files:
                print("Opening", ffile, end="... ")
                filename = ffile
                with tb.open_file(filename, "r") as h5in:

                    # access RWF
                    pmtrwf  = h5in.root.RD. pmtrwf
                    sipmrwf = h5in.root.RD.sipmrwf

                    if not self.monte_carlo:
                        self.eventsInfo = h5in.root.Run.events
                    else:
                        # last row copied from MCTracks table
                        mctrack_row = 0

                    NEVT, NPMT,   PMTWL =  pmtrwf.shape
                    NEVT, NSIPM, SIPMWL = sipmrwf.shape
                    sensor_param = SensorParams(NPMT   = NPMT,
                                                PMTWL  = PMTWL,
                                                NSIPM  = NSIPM,
                                                SIPMWL = SIPMWL)
                    print("Events in file = {}".format(NEVT))

                    if not first:
                        self.print_configuration(sensor_param)
                        first = True
                    # loop over all events in file unless reach nmax
                    for evt in range(NEVT):
                        if self.monte_carlo:
                            # copy corresponding MCTracks to output MCTracks table
                            mctrack_writer.copy_mctracks(h5in.root.MC.MCTracks,
                                          n_events_tot)


                        # deconvolve
                        CWF = self.deconv_pmt(pmtrwf[evt])
                        # calibrated PMT sum
                        csum, csum_mau = self.calibrated_pmt_sum(CWF)
                        #ZS sum for S1 and S2
                        s1_ene, s1_indx = self.csum_zs(
                                          csum_mau,
                                          threshold = self.thr_csum_s1)
                        s2_ene, s2_indx = self.csum_zs(
                                          csum,
                                          threshold = self.thr_csum_s2)

                        # In a few rare cases s2_ene is empty
                        # this is due to empty energy plane events
                        # a protection is set to avoid a crash
                        if np.sum(s2_ene) == 0:
                            self.empty_events += 1
                            continue

                        # SiPMs signals
                        sipmzs = self.calibrated_signal_sipm(sipmrwf[evt])
                        # PMAPS
                        S1, S2 = self.find_S12(s1_ene, s1_indx,   s2_ene, s2_indx)
                        Si     = self.find_S2Si(S2, sipmzs)

                        event, timestamp = self.event_and_timestamp(evt,
                                                n_events_tot)
                        # write to file
                        write(self.run_number, event, timestamp, S1, S2, Si)

                        n_events_tot += 1
                        self.conditional_print(evt, n_events_tot)

                        if n_events_tot >= nmax > -1:
                            print('reached max nof of events (={})'
                                  .format(nmax))
                            break

        if print_empty:
            print('Energy plane empty events (skipped) = {}'.format(
                   self.empty_events))
        return n_events_tot

    def display_IO_info(self, nmax):
        PmapCity.display_IO_info(self, nmax)
        print("""
                 S1 parameters {}""" .format(self.s1_params))
        print("""
                 S2 parameters {}""" .format(self.s2_params))
        print("""
                 S2Si parameters
                 threshold min charge per SiPM = {s.thr_sipm} pes
                 threshold min charge in  S2   = {s.thr_sipm_s2} pes
                          """.format(s=self))

    def event_and_timestamp(self, evt, n_events_tot):
        event = n_events_tot   # TODO fixed in new version of DIOMIRA
        timestamp = int(time()) # TODO fixed in new verison of DIOMIRA
        if not self.monte_carlo:
            evtInfo = self.eventsInfo[evt]
            event = evtInfo[0]
            timestamp = evtInfo[1]
        return event, timestamp


def IRENE(argv = sys.argv):
    """IRENE DRIVER"""

    # get parameters dictionary
    CFP = configure(argv)

    # parameters for s1 searches
    s1par = S12P(tmin   = CFP.S1_TMIN * units.mus,
                 tmax   = CFP.S1_TMAX * units.mus,
                 stride = CFP.S1_STRIDE,
                 lmin   = CFP.S1_LMIN,
                 lmax   = CFP.S1_LMAX,
                 rebin  = False)

    # parameters for s2 searches
    s2par = S12P(tmin   = CFP.S2_TMIN * units.mus,
                 tmax   = CFP.S2_TMAX * units.mus,
                 stride = CFP.S2_STRIDE,
                 lmin   = CFP.S2_LMIN,
                 lmax   = CFP.S2_LMAX,
                 rebin  = True)

    #class instance
    irene = Irene(run_number=CFP.RUN_NUMBER)

    # input files
    # TODO detect non existing files and raise sensible message
    files_in = glob(CFP.FILE_IN)
    files_in.sort()
    irene.set_input_files(files_in)

    # output file
    irene.set_output_file(CFP.FILE_OUT)
    irene.set_compression(CFP.COMPRESSION)
    # print frequency
    irene.set_print(nprint=CFP.NPRINT)

    # parameters of BLR
    irene.set_blr(n_baseline  = CFP.NBASELINE,
                  thr_trigger = CFP.THR_TRIGGER * units.adc)

    # parameters of calibrated sums
    irene.set_csum(n_MAU = CFP.NMAU,
                   thr_MAU = CFP.THR_MAU * units.adc,
                   thr_csum_s1 =CFP.THR_CSUM_S1 * units.pes,
                   thr_csum_s2 =CFP.THR_CSUM_S2 * units.pes)

    # MAU and thresholds for SiPms
    irene.set_sipm(n_MAU_sipm= CFP.NMAU_SIPM,
                   thr_sipm=CFP.THR_SIPM)

    # parameters for PMAP searches
    irene.set_pmap_params(s1_params   = s1par,
                          s2_params   = s2par,
                          thr_sipm_s2 = CFP.THR_SIPM_S2)


    t0 = time()
    nevts = CFP.NEVENTS if not CFP.RUN_ALL else -1
    # run
    nevt = irene.run(nmax=nevts, print_empty=CFP.PRINT_EMPTY_EVENTS)
    t1 = time()
    dt = t1 - t0

    if nevt > 0:
        print("run {} evts in {} s, time/event = {}".format(nevt, dt, dt/nevt))

    return nevts, nevt, irene.empty_events

if __name__ == "__main__":
    IRENE(sys.argv)
