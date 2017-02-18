from __future__ import print_function
import sys

from glob import glob
from time import time

import numpy as np
import tables as tb

from   invisible_cities.core.nh5 import S12, S2Si
import invisible_cities.core.tbl_functions as tbl
from   invisible_cities.core.configure \
       import configure, define_event_loop, print_configuration

from   invisible_cities.core.system_of_units_c import SystemOfUnits

from   invisible_cities.cities.base_cities import PmapCity
from   invisible_cities.cities.base_cities import S12Params as S12P

units = SystemOfUnits()


# Parameters for S1/S2 searches
# tmin and tmax define the time interval of the searches
# lmin and lmax define the minimum and maximum width of the signal
# stride define the tolerable size of the "hole" between to samples above
# threshold when performing the search
# S12Params = namedtuple('S12Params', 'tmin tmax stride lmin lmax')
# S12P = S12Params

class Irene(PmapCity):
    """
    The city of IRENE performs a fast processing directly
    from raw data (pmtrwf and sipmrwf) to PMAPS.
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

    #def set_pmap_store(self, pmap_file_name, compression='ZLIB4'):
    def _set_pmap_store(self, pmap_file):
        """Set the output file."""
        # open pmap store
        #super().set_output_file(pmap_file_name, compression=compression)
        #pmap_file = self.h5out

        # self.compression = compression
        # pmap_file = tb.open_file(pmap_file_name, "w",
        #                           filters = tbl.filters(compression))


        # create a group
        pmapsgroup = pmap_file.create_group(
            pmap_file.root, "PMAPS")

        # create tables to store pmaps
        self.s1t  = pmap_file.create_table(
            pmapsgroup, "S1", S12, "S1 Table",
            tbl.filters(self.compression))

        self.s2t  = pmap_file.create_table(
            pmapsgroup, "S2", S12, "S2 Table",
            tbl.filters(self.compression))

        self.s2sit = pmap_file.create_table(
            pmapsgroup, "S2Si", S2Si, "S2Si Table",
            tbl.filters(self.compression))

        self.s1t  .cols.event.create_index()
        self.s2t  .cols.event.create_index()
        self.s2sit.cols.event.create_index()

        # Create group for run info
        if self.run_number >0:
            rungroup     = pmap_file.create_group(
               pmap_file.root, "Run")

            self.runInfo = pmap_file.create_table(
                rungroup, "runInfo", RunInfo, "runInfo",
                tbl.filters(self.compression))

            self.evtInfot = pmap_file.create_table(
                rungroup, "events", EventInfo, "events",
                tbl.filters(self.compression))


    def _store_s12(self, S12, s12_table, event):
        row = s12_table.row
        for i in S12:
            time = S12[i][0]
            ene  = S12[i][1]
            assert len(time) == len(ene)
            for j in range(len(time)):
                row["event"] = event
                if self.run_number > 0:
                    evtInfo = self.eventsInfo[evt]
                    row["evtDaq"] = evtInfo[0]
                else:
                    row["evtDaq"] = event
                row["peak"] = i
                row["time"] = time[j]
                row["ene"]  =  ene[j]
                row.append()
        s12_table.flush()

    def _store_s2si(self, S2Si, event):
        row = self.s2sit.row
        for i in S2Si:
            sipml = S2Si[i]
            for sipm in sipml:
                nsipm = sipm[0]
                ene   = sipm[1]
                for j, E in enumerate(ene):
                    if E > 0:
                        row["event"] = event
                        if self.run_number > 0:
                            evtInfo = self.eventsInfo[evt]
                            row["evtDaq"] = evtInfo[0]
                        else:
                            row["evtDaq"] = event
                        row["peak"]    = i
                        row["nsipm"]   = nsipm
                        row["nsample"] = j
                        row["ene"]     = E
                        row.append()
        self.s2t.flush()

    def _store_pmaps(self, event, evt, S1, S2, S2Si):
        """Store PMAPS."""
        if self.run_number > 0:
            # Event info
            row     = self.evtInfot.row
            evtInfo = self.eventsInfo[evt]

            row['evt_number'] = evtInfo[0]
            row['timestamp']  = evtInfo[1]
            row.append()
            self.evtInfot.flush()

        self._store_s12(S1, self.s1t, event)
        self._store_s12(S2, self.s2t, event)
        self._store_s2si(S2Si, event)

    def run(self, nmax, print_empty=True):
        """
        Run Irene
        nmax is the max number of events to run
        if print_empty = True, count the number of empty events
        """

        # TODO replace IOError with IC Exceptions

        # TODO checks like the ones in the following block are
        # repeated in the run method of all cities. See whether this
        # can be abstracted.

        n_events_tot = 0

        # check that input/output files are defined
        if not self.input_files:
            raise IOError('input file list is empty, must set before running')
        if not self.output_file:
            raise IOError('must set output file before running')

        # check that S1 and S2 params are defined
        if (not self.s1_params) or (not self.s2_params):
            raise IOError('must set S1/S2 parameters before running')

        print("""
                 IRENE will run a max of {} events
                 Storing PMAPS in {}
                 Input Files = {}"""
              .format(nmax, self.output_file, self.input_files))

        print("""
                 S1 parameters {}""" .format(self.s1_params))

        print("""
                 S2 parameters {}""" .format(self.s2_params))

        print("""
                 S2Si parameters
                 threshold min charge per SiPM = {s.thr_sipm} pes
                 threshold min charge in  S2   = {s.thr_sipm_s2} pes
                          """.format(s=self))

        # loop over input files
        first = False
        with tb.open_file(self.output_file, "w",
                          filters=tbl.filters(self.compression)) as\
                          pmap_file:

            self._set_pmap_store(pmap_file) # prepare the pmap store

            for ffile in self.input_files:
                print("Opening", ffile, end="... ")
                filename = ffile
                with tb.open_file(filename, "r") as h5in:
                    # access RWF
                    pmtrwf  = h5in.root.RD.pmtrwf
                    sipmrwf = h5in.root.RD.sipmrwf

                    if self.run_number > 0:
                        self.eventsInfo = h5in.root.Run.events

                    NEVT, NPMT, PMTWL   = pmtrwf.shape
                    NEVT, NSIPM, SIPMWL = sipmrwf.shape
                    print("Events in file = {}".format(NEVT))

                    if first == False:
                        print_configuration({"# PMT"  : NPMT,
                                             "PMT WL" : PMTWL,
                                             "# SiPM" : NSIPM,
                                             "SIPM WL": SIPMWL})

                        first = True

                        # loop over all events in file unless reach nmax
                    for evt in range(NEVT):
                        # deconvolve
                        CWF = self.deconv_pmt(pmtrwf[evt])
                        # calibrated PMT sum
                        csum, csum_mau = self.calibrated_pmt_sum(CWF)
                        #ZS sum for S1 and S2
                        s2_ene, s2_indx = self.csum_zs(csum,
                                              threshold=self.thr_csum_s2)
                        s1_ene, s1_indx = self.csum_zs(csum_mau,
                                              threshold=self.thr_csum_s1)

                        # In a few rare cases s2_ene is empty
                        # this is due to empty energy plane events
                        # a protection is set to avoid a crash
                        if np.sum(s2_ene) == 0:
                            self.empty_events +=1
                            continue

                        # SiPMs signals
                        sipmzs = self.calibrated_signal_sipm(sipmrwf[evt])
                        # PMAPS
                        S1, S2 = self.find_S12(s1_ene,
                                               s1_indx,
                                               s2_ene,
                                               s2_indx)
                        S2Si = self.find_S2Si(S2, sipmzs)

                        # store PMAPS
                        self._store_pmaps(n_events_tot, evt, S1, S2, S2Si)

                        n_events_tot += 1
                        if n_events_tot%self.nprint == 0:
                            print('event in file = {}, total = {}'
                                  .format(evt, n_events_tot))

                        if n_events_tot >= nmax and nmax > -1:
                            print('reached max nof of events (={})'
                                  .format(nmax))
                            break


        #if pmap_file:
            if self.run_number > 0:
                row = self.runInfot.row
                row['run_number'] = self.run_number
                row.append()
                self.runInfot.flush()
            #pmap_file.close()

        if print_empty:
            print('Energy plane empty events (skipped) = {}'.format(
                   self.empty_events))
        return n_events_tot


def IRENE(argv = sys.argv):
    """IRENE DRIVER"""

    # get parameters dictionary
    CFP = configure(argv)

    # parameters for s1 searches
    s1par = S12P(tmin   = CFP['S1_TMIN'] * units.mus,
                 tmax   = CFP['S1_TMAX'] * units.mus,
                 stride = CFP['S1_STRIDE'],
                 lmin   = CFP['S1_LMIN'],
                 lmax   = CFP['S1_LMAX'],
                 rebin  = False)

    # parameters for s2 searches
    s2par = S12P(tmin   = CFP['S2_TMIN'] * units.mus,
                 tmax   = CFP['S2_TMAX'] * units.mus,
                 stride = CFP['S2_STRIDE'],
                 lmin   = CFP['S2_LMIN'],
                 lmax   = CFP['S2_LMAX'],
                 rebin  = True)

    #class instance
    irene = Irene(run_number=CFP['RUN_NUMBER'])

    # input files
    files_in = glob(CFP['FILE_IN'])
    files_in.sort()
    irene.set_input_files(files_in)

    # output file
    irene.set_output_file(CFP['FILE_OUT'])
    irene.set_compression(CFP['COMPRESSION'])
    # print frequency
    irene.set_print(nprint=CFP['NPRINT'])

    # parameters of BLR
    irene.set_blr(n_baseline  = CFP['NBASELINE'],
                  thr_trigger = CFP['THR_TRIGGER'] * units.adc)

    # parameters of calibrated sums
    irene.set_csum(n_MAU = CFP['NMAU'],
                   thr_MAU = CFP['THR_MAU'] * units.adc,
                   thr_csum_s1 =CFP['THR_CSUM_S1'] * units.pes,
                   thr_csum_s2 =CFP['THR_CSUM_S2'] * units.pes)

    # MAU and thresholds for SiPms
    irene.set_sipm(n_MAU_sipm= CFP['NMAU_SIPM'],
                   thr_sipm=CFP['THR_SIPM'])

    # parameters for PMAP searches
    irene.set_pmap_params(s1_params   = s1par,
                          s2_params   = s2par,
                          thr_sipm_s2 = CFP['THR_SIPM_S2'])


    t0 = time()
    nevts = CFP['NEVENTS'] if not CFP['RUN_ALL'] else -1
    # run
    nevt = irene.run(nmax=nevts, print_empty=CFP['PRINT_EMPTY_EVENTS'])
    t1 = time()
    dt = t1 - t0

    print("run {} evts in {} s, time/event = {}".format(nevt, dt, dt/nevt))


if __name__ == "__main__":
    IRENE(sys.argv)
