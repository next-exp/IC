from __future__ import print_function
import sys

from glob import glob
from time import time
import numpy as np
import tables as tb
import matplotlib.pyplot as plt

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

units = SystemOfUnits()


class Irene:
    """
    The city of IRENE performs a fast processing directly
    from raw data (pmtrwf and sipmrwf) to PMAPS.
    It is optimized for speed (use of CYTHON functions) and intended
    for fast processing of data.
    """
    def __init__(self, run_number=0):
        """
        Init the machine with the run number.
        Load the data base to access calibration and geometry.
        Sets all switches to default value.
        """
        # NB: following JCK-1 convention
        self.run_number = run_number

        # calibration an geometry constants from DB
        # PEP-8
        DataPMT = load_db.DataPMT(run_number)
        DataSiPM = load_db.DataSiPM(run_number)

        # This is JCK-1: text reveals symmetry!
        self.xs              = DataSiPM.X.values
        self.ys              = DataSiPM.Y.values
        self.adc_to_pes      = abs(DataPMT.adc_to_pes.values).astype(np.double)
        self.sipm_adc_to_pes = DataSiPM.adc_to_pes.values    .astype(np.double)
        self.coeff_c         = DataPMT.coeff_c.values        .astype(np.double)
        self.coeff_blr       = DataPMT.coeff_blr.values      .astype(np.double)
        self.xs              = DataSiPM.X.values
        self.ys              = DataSiPM.Y.values

        # BLR default values (override with set_BLR)
        self.n_baseline   = 28000
        self.thr_trigger  = 5 * units.adc

        # MAU default values (override with set_MAU)
        self.n_MAU        = 100
        self.thr_MAU      = thr_MAU = 3 * units.adc

        # CSUM default values (override with set_CSUM)
        self.thr_csum     = 1 * units.pes

        # set switches
        self.setFiles     = False
        self.setPmapStore = False
        self.setS1        = False
        self.setS2        = False
        self.setSiPM      = False
        self.plot_csum    = False
        self.plot_csum_S1 = False
        self.plot_s1      = False
        self.plot_s2      = False
        self.plot_sipm    = False
        self.plot_sipmzs  = False
        self.plot_simap   = False

        # defaul parameters
        self.nprint       = 1000000
        self.signal_start =       0
        self.signal_end   =    1200
        self.ymax         =     200
        self.S1_start     =     100
        self.S1_end       =     100.5
        self.S1_ymax      =       5

        # counters
        self.empty_events = 0 # counts empty events in the energy plane
        self.print_empty = False

    def set_plot(self,
                 plot_csum    = False,
                 plot_csum_S1 = False,
                 plot_s1      = False,
                 plot_s2      = False,
                 plot_sipm    = False,
                 plot_sipmzs  = False,
                 plot_simap   = False,
                 signal_start =    0,
                 signal_end   = 1200,
                 ymax         =  200,
                 S1_start     =  100,
                 S1_end       =  100.5,
                 S1_ymax      =    5):
        """Decide what to plot."""

        self.plot_csum    = plot_csum
        self.plot_csum_S1 = plot_csum_S1
        self.plot_s1      = plot_s1
        self.plot_s2      = plot_s2
        self.plot_sipm    = plot_sipm
        self.plot_sipmzs  = plot_sipmzs
        self.plot_simap   = plot_simap
        self.signal_start = signal_start
        self.signal_end   = signal_end
        self.ymax         = ymax
        self.S1_start     = S1_start
        self.S1_end       = S1_end
        self.S1_ymax      = S1_ymax

    def set_print(self, nprint=10, print_empty_events=False):
        """Print frequency."""
        self.nprint = nprint
        self.print_empty = print_empty_events

    def set_input_files(self, input_files):
        """Set the input files."""
        self.input_files = input_files
        self.setFiles    = True

    def set_pmap_store(self, pmap_file, compression='ZLIB4'):
        """Set the input files."""
        # open pmap store
        self.pmapFile = tb.open_file(
            pmap_file, "w", filters=tbl.filters(compression))

        # create a group
        pmapsgroup = self.pmapFile.create_group(
            self.pmapFile.root, "PMAPS")

        # create tables to store pmaps
        self.s1t  = self.pmapFile.create_table(
            pmapsgroup, "S1", S12, "S1 Table",
            tbl.filters(compression))

        self.s2t  = self.pmapFile.create_table(
            pmapsgroup, "S2", S12, "S2 Table",
            tbl.filters(compression))

        self.s2sit = self.pmapFile.create_table(
            pmapsgroup, "S2Si", S2Si, "S2Si Table",
            tbl.filters(compression))

        self.s1t  .cols.event.create_index()
        self.s2t  .cols.event.create_index()
        self.s2sit.cols.event.create_index()

        # Create group for run info
        if self.run_number >0:
            rungroup     = self.pmapFile.create_group(
               self.pmapFile.root, "Run")

            self.runInfo = self.pmapFile.create_table(
                rungroup, "runInfo", RunInfo, "runInfo",
                tbl.filters(compression))

            self.evtInfot = self.pmapFile.create_table(
                rungroup, "events", EventInfo, "events",
                tbl.filters(compression))

        self.setPmapStore = True

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

    def set_S1(self, tmin=0 * units.mus, tmax=590 * units.mus,
               stride=4, lmin=4, lmax=20):
        """Parameters for S1 search."""
        self.  tmin_s1 = tmin
        self.  tmax_s1 = tmax
        self.stride_s1 = stride
        self.  lmin_s1 = lmin
        self.  lmax_s1 = lmax
        self.setS1     = True

    def set_S2(self, tmin=590 * units.mus, tmax=620 * units.mus,
               stride=40, lmin=100, lmax=1000000):
        """Parameters for S2 search."""
        self.  tmin_s2 = tmin
        self.  tmax_s2 = tmax
        self.stride_s2 = stride
        self.  lmin_s2 = lmin
        self.  lmax_s2 = lmax
        self.setS2     = True

    def set_SiPM(self, thr_zs=5 * units.pes, thr_sipm_s2=50 * units.pes):
        """Parameters for SiPM analysis."""
        self.thr_zs      = thr_zs
        self.thr_sipm_s2 = thr_sipm_s2
        self.setSiPM     = True

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

        #S1
        row = self.s1t.row
        for i in S1:
            time = S1[i][0]
            ene  = S1[i][1]
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
        self.s1t.flush()

        #S2
        row = self.s2t.row
        for i in S2.keys():
            time = S2[i][0]
            ene  = S2[i][1]
            assert len(time) == len(ene)
            for j in range(len(time)):
                row["event"] = event
                if self.run_number > 0:
                    evtInfo = self.eventsInfo[evt]
                    row["evtDaq"] = evtInfo[0]
                else:
                    row["evtDaq"] = event
                row["peak"] =      i
                row["time"] = time[j]
                row["ene"]  =  ene[j]
                row.append()
        self.s2t.flush()

        # S2Si
        row = self.s2sit.row
        for i in S2Si.keys():
            sipml = S2Si[i]
            for sipm in sipml:
                nsipm = sipm[0]
                ene   = sipm[1]
                for j in range(len(ene)):
                    if ene[j] > 0:
                        row["event"] = event
                        if self.run_number >0:
                            evtInfo = self.eventsInfo[evt]
                            row["evtDaq"] = evtInfo[0]
                        else:
                            row["evtDaq"] = event
                        row["peak"]    = i
                        row["nsipm"]   = nsipm
                        row["nsample"] = j
                        row["ene"]     = ene[j]
                        row.append()
        self.s2t.flush()

    def plot_ene_sipm(self, S2Si, radius=3):
        """
        plots the reconstructed energy of the SiPMs
        input: sipm dictionary
        """
        r = np.ones(len(self.xs)) * radius
        col = np.zeros(len(self.xs))
        for i in S2Si.keys():
            sipml = S2Si[i]
            for sipm in sipml:
                sipm_n  = sipm[0]
                sipm_wf = sipm[1]
                col[sipm_n] = np.sum(sipm_wf)

        plt.figure(figsize=(10, 10))
        plt.subplot(aspect="equal")
        mpl.circles(self.xs, self.ys, r, c=col, alpha=0.5, ec="none")
        plt.colorbar()
        plt.xlim(-198, 198)
        plt.ylim(-198, 198)

    def run(self, nmax, store_pmaps=False):
        """
        Run the machine
        nmax is the max number of events to run
        store_pmaps decides whether to store pmaps or not
        """
        n_events_tot = 0
        # check the state of the machine
        if not self.input_files:
            raise IOError('input file list is empty')

        if not self.setFiles:
            raise IOError('must set files before running')
        if not self.setS1:
            raise IOError('must set S1 parameters before running')
        if not self.setS2:
            raise IOError('must set S2 parameters before running')
        if not self.setSiPM:
            raise IOError('must set Sipm parameters before running')

        print("""
                 IRENE will run a max of {} events
                 Storing PMAPS (1=yes/0=no)
                 Input Files ={}
                          """.format(nmax, store_pmaps, self.input_files))

        print("""
                 S1 parameters
                 tmin = {} mus tmax = {} mus stride = {}
                 lmin = {} lmax = {}
                          """.format(self.  tmin_s1 / units.mus,
                                     self.  tmax_s1 / units.mus,
                                     self.stride_s1,
                                     self.  lmin_s1,
                                     self.  lmax_s1))
        print("""
                 S2 parameters
                 tmin = {} mus tmax = {} mus stride = {}
                 lmin = {} lmax = {}
                          """.format(self.  tmin_s2 / units.mus,
                                     self.  tmax_s2 / units.mus,
                                     self.stride_s2,
                                     self.  lmin_s2,
                                     self.  lmax_s2))
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
            try:
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
                        # plots
                        if self.plot_csum:
                            mpl.plot_signal(
                                self.signal_t / units.mus,
                                csum,
                                title        = "calibrated sum, ZS",
                                signal_start = self.signal_start,
                                signal_end   = self.signal_end,
                                ymax         = self.ymax,
                                t_units      = 'mus',
                                units        = 'pes')

                        if self.plot_csum_S1:
                            mpl.plot_signal(
                                self.signal_t / units.mus,
                                csum,
                                title        = "calibrated sum, S1",
                                signal_start = self.S1_start,
                                signal_end   = self.S1_end,
                                ymax         = self.S1_ymax,
                                t_units      = 'mus',
                                units        = "pes")
                            plt.show()
                            raw_input('->')
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
                                          tmin   = self.  tmin_s1,
                                          tmax   = self.  tmax_s1,
                                          lmin   = self.  lmin_s1,
                                          lmax   = self.  lmax_s1,
                                          stride = self.stride_s1,
                                          rebin  = False)
                        # find S2
                        S2 = cpf.find_S12(wfzs_ene,
                                          wfzs_indx,
                                          tmin         = self.  tmin_s2,
                                          tmax         = self.  tmax_s2,
                                          lmin         = self.  lmin_s2,
                                          lmax         = self.  lmax_s2,
                                          stride       = self.stride_s2,
                                          rebin        = True,
                                          rebin_stride = self.stride_s2)
                        #plot S1 & S2
                        if self.plot_s1: pmp.scan_s12(S1)
                        if self.plot_s2: pmp.scan_s12(S2)
                        # plot sipms
                        if self.plot_sipm:
                            mpl.plot_sipm(sipmrwf[evt],
                                          nmin = 0,
                                          nmax = 16,
                                          x = 4,
                                          y = 4)
                            plt.show()
                            raw_input('->')
                        # SiPMs zero suppression
                        sipmzs = cpf.signal_sipm(
                            sipmrwf[evt],
                            self.sipm_adc_to_pes,
                            thr   = self.thr_zs,
                            n_MAU = self.n_MAU)
                        # plot
                        if self.plot_sipmzs:
                            mpl.plot_sipm(sipmzs,
                                          nmin = 0,
                                          nmax = 16,
                                          x = 4,
                                          y = 4)
                            plt.show()
                            raw_input('->')
                        # select SIPM ZS and create S2Si
                        SIPM = cpf.select_sipm(sipmzs)
                        S2Si = pmp.sipm_s2_dict(SIPM,
                                                S2,
                                                thr = self.thr_sipm_s2)

                        if store_pmaps == True:
                            if self.setPmapStore == False:
                                raise IOError(
                                  'must set PMAPS before storing')
                            self.store_pmaps(n_events_tot, evt, S1, S2, S2Si)

                        if self.plot_simap:
                            self.plot_ene_sipm(S2Si)
                            plt.show()
                            raw_input('->')

                        n_events_tot +=1
                        if n_events_tot%self.nprint == 0:
                            print('event in file = {}, total = {}'
                                  .format(evt, n_events_tot))

                        if n_events_tot >= nmax and nmax > -1:
                            print('reached max nof of events (={})'
                                  .format(nmax))
                            break


            except Exception:
                print('error')
                raise

        if store_pmaps == True:
            if self.run_number > 0:
                row = self.runInfot.row
                row['run_number'] = self.run_number
                row.append()
                self.runInfot.flush()
            self.pmapFile.close()

        if self.print_empty:
            print('Energy plane empty events (skipped) = {}'.format(
                   self.empty_events))
        return n_events_tot



def IRENE(argv = sys.argv):
    """IRENE DRIVER"""
    CFP = configure(argv)

    fpp = Irene(run_number = CFP['RUN_NUMBER'])

    files_in = glob(CFP['FILE_IN'])
    files_in.sort()
    fpp.set_input_files(files_in)
    fpp.set_pmap_store(CFP['FILE_OUT'],
                       compression = CFP['COMPRESSION'])
    fpp.set_print(nprint = CFP['NPRINT'],
                  print_empty_events=CFP['PRINT_EMPTY_EVENTS'])

    fpp.set_BLR(n_baseline  = CFP['NBASELINE'],
                thr_trigger = CFP['THR_TRIGGER'] * units.adc)
    fpp.set_MAU(  n_MAU = CFP['NMAU'],
                thr_MAU = CFP['THR_MAU'] * units.adc)
    fpp.set_CSUM(thr_csum = CFP['THR_CSUM'] * units.pes)
    fpp.set_S1(tmin   = CFP['S1_TMIN'] * units.mus,
               tmax   = CFP['S1_TMAX'] * units.mus,
               lmax   = CFP['S1_LMAX'],
               lmin   = CFP['S1_LMIN'],
               stride = CFP['S1_STRIDE'])
    fpp.set_S2(tmin   = CFP['S2_TMIN'] * units.mus,
               tmax   = CFP['S2_TMAX'] * units.mus,
               stride = CFP['S2_STRIDE'],
               lmin   = CFP['S2_LMIN'],
               lmax   = CFP['S2_LMAX'])
    fpp.set_SiPM(thr_zs      = CFP['THR_ZS']      * units.pes,
                 thr_sipm_s2 = CFP['THR_SIPM_S2'] * units.pes)

    t0 = time()
    nevts = CFP['NEVENTS'] if not CFP['RUN_ALL'] else -1
    nevt = fpp.run(nmax=nevts, store_pmaps=True)
    t1 = time()
    dt = t1 - t0

    print("run {} evts in {} s, time/event = {}".format(nevt, dt, dt/nevt))


if __name__ == "__main__":
    IRENE(sys.argv)
