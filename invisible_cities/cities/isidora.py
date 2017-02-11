from __future__ import print_function
import sys

from glob import glob
from time import time
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
from   invisible_cities.core.exceptions import NoInputFiles

units = SystemOfUnits()


class Isidora:
    """
    The city of ISIDORA performs a fast processing from raw data
    (pmtrwf and sipmrwf) to BLR wavefunctions.
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

        DataPMT  = load_db.DataPMT (run_number)
        DataSiPM = load_db.DataSiPM(run_number)

        # This is JCK-1: text reveals symmetry!
        self.xs              = DataSiPM.X.values
        self.ys              = DataSiPM.Y.values
        self.adc_to_pes      = abs(DataPMT.adc_to_pes.values).astype(np.double)
        self.sipm_adc_to_pes = DataSiPM.adc_to_pes.values    .astype(np.double)
        self.coeff_c         = DataPMT.coeff_c.values        .astype(np.double)
        self.coeff_blr       = DataPMT.coeff_blr.values      .astype(np.double)

        # BLR default values (override with set_BLR)
        self.n_baseline   = 28000
        self.thr_trigger  = 5 * units.adc

        # MAU default values (override with set_MAU)
        self.  n_MAU      = 100
        self.thr_MAU      = thr_MAU = 3 * units.adc

        # set switches
        self.setFiles     = False
        self.setSiPM      = False
        self.setCwfStore  = False

        # defaul parameters
        self.nprint       = 1000000
        self.signal_start =       0
        self.signal_end   =    1200

    def set_print(self, nprint=10):
        """Print frequency."""
        self.nprint = nprint

    def set_input_files(self, input_files):
        """Set the input files."""
        self.input_files = input_files
        self.setFiles    = True

    def set_BLR(self, n_baseline=38000, thr_trigger=5 * units.adc):
        """Parameters of the BLR."""
        self.n_baseline  = n_baseline
        self.thr_trigger = thr_trigger

    def set_MAU(self, n_MAU=100, thr_MAU=3 * units.adc):
        """Parameters of the MAU used to remove low frequency noise."""
        self.  n_MAU =   n_MAU
        self.thr_MAU = thr_MAU

    def set_cwf_store(self, cwf_file, compression='ZLIB4'):
        """Set the input files."""
        # open cwf store
        self.cwfFile = tb.open_file(
            cwf_file, "w", filters=tbl.filters(compression))

        # create a group
        self.cwf_group = self.cwfFile.create_group(self.cwfFile.root, "BLR")

        self.setCwfStore = True
        self.compression = compression

    def store_cwf(self, cwf):
        """Store CWF."""
        NPMT, PMTWL = cwf.shape
        if "pmtcwf" not in self.cwf_group:
            # create earray to store cwf
            self.pmtcwf = self.cwfFile.create_earray(self.cwf_group, "pmtcwf",
                                                atom    = tb.Float32Atom(),
                                                shape   = (0, NPMT, PMTWL),
                                                filters = tbl.filters(self.compression))
        self.pmtcwf.append(cwf.reshape(1, NPMT, PMTWL))


    def run(self, nmax):
        """
        Run the machine
        nmax is the max number of events to run
        """
        n_events_tot = 0
        # check the state of the machine
        if not self.setFiles:
            raise NoInputFiles('Input file list is empty, set it before running')

        print("""
                 ISIDORA will run a max of {} events
                 Input Files ={}
                          """.format(nmax, self.input_files))

        # loop over input files
        first = False
        for ffile in self.input_files:
            print("Opening", ffile, end="... ")
            filename = ffile
            with tb.open_file(filename, "r") as h5in:
                # access RWF
                pmtrwf  = h5in.root.RD.pmtrwf
                sipmrwf = h5in.root.RD.sipmrwf

                # Copy sensor table if exists (needed for GATE)
                if self.setCwfStore:
                    if 'Sensors' in h5in.root:
                        self.sensors_group = self.cwfFile.create_group(
                            self.cwfFile.root, "Sensors")
                        datapmt = h5in.root.Sensors.DataPMT
                        datapmt.copy(newparent=self.sensors_group)

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

                    self.signal_t = np.arange(0, PMTWL * 25, 25)
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

                    if self.setCwfStore:
                        self.store_cwf(CWF)

                    n_events_tot += 1
                    if n_events_tot % self.nprint == 0:
                        print('event in file = {}, total = {}'
                              .format(evt, n_events_tot))

                    if n_events_tot >= nmax > -1:
                        print('reached max nof of events (= {})'
                              .format(nmax))
                        break

        if self.setCwfStore:
            self.cwfFile.close()

        return n_events_tot


def ISIDORA(argv = sys.argv):
    """ISIDORA DRIVER"""
    CFP = configure(argv)

    fpp = Isidora(run_number = CFP['RUN_NUMBER'])

    files_in = glob(CFP['FILE_IN'])
    files_in.sort()
    fpp.set_input_files(files_in)
    fpp.set_cwf_store(CFP['FILE_OUT'],
                       compression = CFP['COMPRESSION'])
    fpp.set_print(nprint = CFP['NPRINT'])

    fpp.set_BLR(n_baseline  = CFP['NBASELINE'],
                thr_trigger = CFP['THR_TRIGGER'] * units.adc)
    fpp.set_MAU(  n_MAU = CFP['NMAU'],
                thr_MAU = CFP['THR_MAU'] * units.adc)

    t0 = time()
    nevts = CFP['NEVENTS'] if not CFP['RUN_ALL'] else -1
    nevt = fpp.run(nmax=nevts)
    t1 = time()
    dt = t1 - t0

    print("run {} evts in {} s, time/event = {}".format(nevt, dt, dt / nevt))


if __name__ == "__main__":
    ISIDORA(sys.argv)
