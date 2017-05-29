import sys

from glob import glob
from time import time

import numpy  as np
import tables as tb

from .. core.configure         import configure
from .. core.system_of_units_c import units

from .. reco        import tbl_functions as tbl
from .. reco.params import SensorParams

from .  base_cities import DeconvolutionCity

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

        self.wf_tables = {}
        DeconvolutionCity.__init__(self,
                                   run_number  = run_number,
                                   files_in    = files_in,
                                   file_out    = file_out,
                                   nprint      = nprint,
                                   n_baseline  = n_baseline,
                                   thr_trigger = thr_trigger)


    def _store_wf(self, wf, wf_table, wf_file, wf_group):
        """Store WF."""
        n_sensors, wl = wf.shape
        if wf_table not in wf_group:
            # create earray to store cwf
            self.wf_tables[wf_table] = wf_file.create_earray(wf_group, wf_table,
                                                 atom    = tb.Float32Atom(),
                                                 shape   = (0, n_sensors, wl),
                                                 filters = tbl.filters(self.compression))
        self.wf_tables[wf_table].append(wf.reshape(1, n_sensors, wl))


    def run(self, nmax):
        """
        Run the machine
        nmax is the max number of events to run
        """
        self.check_files()
        self.display_IO_info(nmax)

        # loop over input files
        with tb.open_file(self.output_file, "w",
                          filters = tbl.filters(self.compression)) as cwf_file:
            cwf_group = cwf_file.create_group(cwf_file.root, "BLR")
            n_events_tot = self.FACTOR1(cwf_file, cwf_group, nmax)

        return n_events_tot


    def FACTOR1(self, cwf_file, cwf_group, nmax):
        n_events_tot = 0
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
                    datasipm = h5in.root.Sensors.DataSiPM
                    datasipm.copy(newparent=self.sensors_group)

                if not self.monte_carlo:
                    self.eventsInfo = h5in.root.Run.events

                NEVT, NPMT,   PMTWL = pmtrwf .shape
                NEVT, NSIPM, SIPMWL = sipmrwf.shape
                sensor_param = SensorParams(NPMT   = NPMT,
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
                    CWF = self.deconv_pmt(pmtrwf[evt])
                    self._store_wf(CWF, 'pmtcwf', cwf_file, cwf_group)
                    #Copy sipm waveform
                    self._store_wf(sipmrwf[evt], 'sipmrwf', cwf_file, cwf_group)

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

    files_in    = glob(CFP.FILE_IN)
    files_in.sort()


    fpp = Isidora(run_number  = CFP.RUN_NUMBER,
                  files_in    = files_in,
                  n_baseline  = CFP.NBASELINE,
                  thr_trigger = CFP.THR_TRIGGER * units.adc)

    #fpp.set_input_files(files_in)
    fpp.set_output_file(CFP.FILE_OUT)
    fpp.set_compression(CFP.COMPRESSION)
    fpp.set_print(nprint = CFP.NPRINT)

    t0 = time()
    nevts = CFP.NEVENTS if not CFP.RUN_ALL else -1
    nevt = fpp.run(nmax=nevts)
    t1 = time()
    dt = t1 - t0

    print("run {} evts in {} s, time/event = {}".format(nevt, dt, dt / nevt))

    return nevts, nevt

if __name__ == "__main__":
    ISIDORA(sys.argv)
