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

from   invisible_cities.cities.base_cities import DeconvolutionCity
from   invisible_cities.cities.base_cities import DeconvolutionCity, SensorParam

units = SystemOfUnits()

class Isidora(DeconvolutionCity):
    """
    The city of ISIDORA performs a fast processing from raw data
    (pmtrwf and sipmrwf) to BLR wavefunctions.
    It is optimized for speed (use of CYTHON functions) and intended
    for fast processing of data.
    """

    def __init__(self,
                 run_number  = 0,
                 files_in    = None,
                 file_out    = None,
                 nprint      = 10000,
                 n_baseline  = 28000,
                 thr_trigger = 5 * units.adc):
        """
        Init the machine with the run number.
        Load the data base to access calibration and geometry.
        Sets all switches to default value.
        """

        DeconvolutionCity.__init__(self,
                                   run_number  = run_number,
                                   files_in    = files_in,
                                   file_out    = file_out,
                                   nprint      = nprint,
                                   n_baseline  = n_baseline,
                                   thr_trigger = thr_trigger)

    def _store_cwf(self, cwf, cwf_file, cwf_group):
        """Store CWF."""
        NPMT, PMTWL = cwf.shape
        if "pmtcwf" not in cwf_group:
            # create earray to store cwf
            self.pmtcwf = cwf_file.create_earray(cwf_group, "pmtcwf",
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

        self.check_files()
        self.display_IO_info(nmax)

        # loop over input files
        with tb.open_file(self.output_file, "w",
                          filters = tbl.filters(self.compression)) as cwf_file:
            cwf_group = cwf_file.create_group(cwf_file.root, "BLR")
            first = False
            for ffile in self.input_files:
                print("Opening", ffile, end="... ")
                filename = ffile
                with tb.open_file(filename, "r") as h5in:
                    # access RWF
                    pmtrwf  = h5in.root.RD.pmtrwf
                    sipmrwf = h5in.root.RD.sipmrwf

                    # Copy sensor table if exists (needed for GATE)
                    if 'Sensors' in h5in.root:
                        self.sensors_group = cwf_file.create_group(
                            cwf_file.root, "Sensors")
                        datapmt = h5in.root.Sensors.DataPMT
                        datapmt.copy(newparent=self.sensors_group)

                    if not self.monte_carlo:
                        self.eventsInfo = h5in.root.Run.events

                    NEVT, NPMT,   PMTWL = pmtrwf .shape
                    NEVT, NSIPM, SIPMWL = sipmrwf.shape
                    print("Events in file = {}".format(NEVT))
                    sensor_param = SensorParam(NPMT   = NPMT,
                                               PMTWL  = PMTWL,
                                               NSIPM  = NSIPM,
                                               SIPMWL = SIPMWL)


                    print("Events in file = {}".format(NEVT))

                    if not first:
                        self.print_configuration(sensor_param)
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

                        self._store_cwf(CWF, cwf_file, cwf_group)

                        n_events_tot += 1
                        if n_events_tot % self.nprint == 0:
                            print('event in file = {}, total = {}'
                                  .format(evt, n_events_tot))

                        if n_events_tot >= nmax > -1:
                            print('reached max nof of events (= {})'
                                  .format(nmax))
                            break

        return n_events_tot


def ISIDORA(argv = sys.argv):
    """ISIDORA DRIVER"""
    CFP = configure(argv)

    files_in    = glob(CFP['FILE_IN'])
    files_in.sort()


    fpp = Isidora(run_number  = CFP['RUN_NUMBER'],
                  files_in    = files_in,
                  n_baseline  = CFP['NBASELINE'],
                  thr_trigger = CFP['THR_TRIGGER'] * units.adc)

    #fpp.set_input_files(files_in)
    fpp.set_output_file(CFP['FILE_OUT'])
    fpp.set_compression(CFP['COMPRESSION'])
    fpp.set_print(nprint = CFP['NPRINT'])

    t0 = time()
    nevts = CFP['NEVENTS'] if not CFP['RUN_ALL'] else -1
    nevt = fpp.run(nmax=nevts)
    t1 = time()
    dt = t1 - t0

    print("run {} evts in {} s, time/event = {}".format(nevt, dt, dt / nevt))


if __name__ == "__main__":
    ISIDORA(sys.argv)
