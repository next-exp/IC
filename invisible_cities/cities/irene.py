from __future__ import print_function
import sys

from glob import glob
from time import time

from collections import namedtuple

import numpy as np
import tables as tb

from   invisible_cities.core.nh5 import RunInfo, EventInfo
from   invisible_cities.core.nh5 import S12, S2Si
import invisible_cities.core.mpl_functions as mpl
import invisible_cities.core.pmaps_functions as pmp
import invisible_cities.core.tbl_functions as tbl
from   invisible_cities.core.configure \
       import configure, define_event_loop, print_configuration

import invisible_cities.sierpe.blr as blr
import invisible_cities.core.peak_functions_c as cpf
from   invisible_cities.core.system_of_units_c import SystemOfUnits
from   invisible_cities.database import load_db

from   invisible_cities.cities.base_cities import DeconvolutionCity

units = SystemOfUnits()


S12Params = namedtuple('S12Params', 'tmin tmax stride lmin lmax')
S12P = S12Params

class Irene(DeconvolutionCity):
    """
    The city of IRENE performs a fast processing directly
    from raw data (pmtrwf and sipmrwf) to PMAPS.
    It is optimized for speed (use of CYTHON functions) and intended
    for fast processing of data.
    """
    def __init__(self,
                 run_number  = 0,
                 files_in    = None,
                 n_baseline  = 28000,
                 thr_trigger = 5 * units.adc,
                 n_MAU       =   100,
                 thr_MAU     = 3 * units.adc):
        """
        Init the machine with the run number.
        Load the data base to access calibration and geometry.
        Sets all switches to default value.
        """

        DeconvolutionCity.__init__(self, run_number, files_in,
                                   n_baseline, thr_trigger, n_MAU,
                                   thr_MAU)

        # CSUM default values (override with set_CSUM)
        self.thr_csum     = 1 * units.pes

        # default parameters
        self.ymax         =     200
        self.S1_start     =     100
        self.S1_end       =     100.5
        self.S1_ymax      =       5

        self.s1_params   = None
        self.s2_params   = None
        self.thr_zs      = None
        self.thr_sipm_s2 = None

        # counters
        self.empty_events = 0 # counts empty events in the energy plane

    def set_pmap_store(self, pmap_file_name, compression='ZLIB4'):
        """Set the input files."""
        # open pmap store
        self.pmap_file = tb.open_file(
            pmap_file_name, "w", filters=tbl.filters(compression))

        # create a group
        pmapsgroup = self.pmap_file.create_group(
            self.pmap_file.root, "PMAPS")

        # create tables to store pmaps
        self.s1t  = self.pmap_file.create_table(
            pmapsgroup, "S1", S12, "S1 Table",
            tbl.filters(compression))

        self.s2t  = self.pmap_file.create_table(
            pmapsgroup, "S2", S12, "S2 Table",
            tbl.filters(compression))

        self.s2sit = self.pmap_file.create_table(
            pmapsgroup, "S2Si", S2Si, "S2Si Table",
            tbl.filters(compression))

        self.s1t  .cols.event.create_index()
        self.s2t  .cols.event.create_index()
        self.s2sit.cols.event.create_index()

        # Create group for run info
        if self.run_number >0:
            rungroup     = self.pmap_file.create_group(
               self.pmap_file.root, "Run")

            self.runInfo = self.pmap_file.create_table(
                rungroup, "runInfo", RunInfo, "runInfo",
                tbl.filters(compression))

            self.evtInfot = self.pmap_file.create_table(
                rungroup, "events", EventInfo, "events",
                tbl.filters(compression))

    def set_BLR(self, n_baseline=38000, thr_trigger=5 * units.adc):
        """Parameters of the BLR."""
        self.n_baseline  = n_baseline
        self.thr_trigger = thr_trigger

    def set_MAU(self, n_MAU=100, thr_MAU=3 * units.adc):
        """Parameters of the MAU used to remove low frequency noise."""
        self.  n_MAU =   n_MAU
        self.thr_MAU = thr_MAU

    def set_CSUM(self, thr_csum=1 * units.pes):
        """Parameter for ZS in the calibrated sum."""
        self.thr_csum = thr_csum

    def set_s12(self, s1, s2):
        """Parameters for S1 search."""
        self.s1_params = s1
        self.s2_params = s2

    def set_sipm(self, thr_zs=5 * units.pes, thr_sipm_s2=50 * units.pes):
        """Parameters for SiPM analysis."""
        self.thr_zs      = thr_zs
        self.thr_sipm_s2 = thr_sipm_s2

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

    def store_pmaps(self, event, evt, S1, S2, S2Si):
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
        Run the machine
        nmax is the max number of events to run
        store_pmaps decides whether to store pmaps or not
        """

        # TODO replace IOError with IC Exceptions

        n_events_tot = 0
        # check the state of the machine
        if not self.input_files:
            raise IOError('input file list is empty')

        if not self.input_files:
            raise IOError('must set files before running')
        if (not self.s1_params) or (not self.s2_params):
            raise IOError('must set S1/S2 parameters before running')
        if not self.thr_zs:
            raise IOError('must set Sipm parameters before running')

        print("""
                 IRENE will run a max of {} events
                 Storing PMAPS in {}
                 Input Files = {}"""
              .format(nmax, self.pmap_file.filename, self.input_files))

        print("""
                 S1 parameters {}""" .format(self.s1_params))

        print("""
                 S2 parameters {}""" .format(self.s2_params))

        print("""
                 S2Si parameters
                 threshold min charge per SiPM = {s.thr_zs} pes
                 threshold min charge in  S2   = {s.thr_sipm_s2} pes
                          """.format(s=self))

        # loop over input files
        first = False
        for ffile in self.input_files:
            print("Opening", ffile, end="... ")
            filename = ffile
            with tb.open_file(filename, "r+") as h5in:
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

                    self.signal_t = np.arange(0., PMTWL * 25, 25)
                    first = True
                # loop over all events in file unless reach nmax
                for evt in range(NEVT):
                    # deconvolve
                    CWF = blr.deconv_pmt(
                      pmtrwf[evt],
                      self.coeff_c,
                      self.coeff_blr,
                      n_baseline  = self.n_baseline,
                      thr_trigger = self.thr_trigger)
                    # calibrated PMT sum
                    csum, _ = cpf.calibrated_pmt_sum(
                      CWF,
                      self.adc_to_pes,
                        n_MAU = self.  n_MAU,
                      thr_MAU = self.thr_MAU)
                    # SiPMs
                    # Supress samples below threshold (in pes)
                    wfzs_ene, wfzs_indx = cpf.wfzs(
                        csum, threshold=self.thr_csum)

                    # In a few rare cases wfzs_ene is empty
                    # this is due to empty energy plane events
                    # a protection is set to avoid a crash

                    if np.sum(wfzs_ene) == 0:
                        self.empty_events +=1
                        continue
                    # find S1
                    S1 = cpf.find_S12(wfzs_ene,
                                      wfzs_indx,
                                      rebin = False,
                                      **self.s1_params._asdict())
                    # find S2
                    S2 = cpf.find_S12(wfzs_ene,
                                      wfzs_indx,
                                      rebin = True,
                                      **self.s2_params._asdict())
                    # SiPMs zero suppression
                    sipmzs = cpf.signal_sipm(
                        sipmrwf[evt],
                        self.sipm_adc_to_pes,
                        thr   = self.thr_zs,
                        n_MAU = self.n_MAU)
                    # select SIPM ZS and create S2Si
                    SIPM = cpf.select_sipm(sipmzs)
                    S2Si = pmp.sipm_s2_dict(SIPM,
                                            S2,
                                            thr = self.thr_sipm_s2)

                    if not self.pmap_file:
                        raise IOError('must set PMAPS before storing')
                    self.store_pmaps(n_events_tot, evt, S1, S2, S2Si)

                    n_events_tot += 1
                    if n_events_tot%self.nprint == 0:
                        print('event in file = {}, total = {}'
                              .format(evt, n_events_tot))

                    if n_events_tot >= nmax and nmax > -1:
                        print('reached max nof of events (={})'
                              .format(nmax))
                        break


        if self.pmap_file:
            if self.run_number > 0:
                row = self.runInfot.row
                row['run_number'] = self.run_number
                row.append()
                self.runInfot.flush()
            self.pmap_file.close()

        if print_empty:
            print('Energy plane empty events (skipped) = {}'.format(
                   self.empty_events))
        return n_events_tot



def IRENE(argv = sys.argv):
    """IRENE DRIVER"""
    CFP = configure(argv)

    fpp = Irene(run_number=CFP['RUN_NUMBER'])

    files_in = glob(CFP['FILE_IN'])
    files_in.sort()
    fpp.set_input_files(files_in)
    fpp.set_pmap_store(CFP['FILE_OUT'],
                       compression = CFP['COMPRESSION'])
    fpp.set_print(nprint=CFP['NPRINT'])

    fpp.set_BLR(n_baseline  = CFP['NBASELINE'],
                thr_trigger = CFP['THR_TRIGGER'] * units.adc)

    fpp.set_MAU(  n_MAU = CFP['NMAU'],
                thr_MAU = CFP['THR_MAU'] * units.adc)

    fpp.set_CSUM(thr_csum = CFP['THR_CSUM'] * units.pes)

    fpp.set_s12(s1 = S12P(tmin   = CFP['S1_TMIN'] * units.mus,
                          tmax   = CFP['S1_TMAX'] * units.mus,
                          stride = CFP['S1_STRIDE'],
                          lmin   = CFP['S1_LMIN'],
                          lmax   = CFP['S1_LMAX']),
                s2 = S12P(tmin   = CFP['S2_TMIN'] * units.mus,
                          tmax   = CFP['S2_TMAX'] * units.mus,
                          stride = CFP['S2_STRIDE'],
                          lmin   = CFP['S2_LMIN'],
                          lmax   = CFP['S2_LMAX']))

    fpp.set_sipm(thr_zs=CFP['THR_ZS'] * units.pes,
                 thr_sipm_s2=CFP['THR_SIPM_S2'] * units.pes)

    t0 = time()
    nevts = CFP['NEVENTS'] if not CFP['RUN_ALL'] else -1
    nevt = fpp.run(nmax=nevts, print_empty=CFP['PRINT_EMPTY_EVENTS'])
    t1 = time()
    dt = t1 - t0

    print("run {} evts in {} s, time/event = {}".format(nevt, dt, dt/nevt))


if __name__ == "__main__":
    IRENE(sys.argv)
