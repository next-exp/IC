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
                 compression = 'ZLIB4',
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
                                   compression = compression,
                                   nprint      = nprint,
                                   n_baseline  = n_baseline,
                                   thr_trigger = thr_trigger)

        self.check_files()


    def run(self, nmax : 'max number of events to run'):
        self.display_IO_info(nmax)
        sensor_params = self.get_sensor_params(self.input_files[0])
        self.print_configuration(sensor_params)

        with tb.open_file(self.output_file, "w",
                          filters = tbl.filters(self.compression)) as h5out:
            pmt_writer  = rwf_writer(h5out,
                                     group_name      = 'BLR',
                                     table_name      = 'pmtcwf',
                                     n_sensors       = sensor_params.NPMT,
                                     waveform_length = sensor_params.PMTWL)
            sipm_writer = rwf_writer(h5out,
                                     group_name      = 'BLR',
                                     table_name      = 'sipmrwf',
                                     n_sensors       = sensor_params.NSIPM,
                                     waveform_length = sensor_params.SIPMWL)

            n_events_tot = self._main_loop(pmt_writer, sipm_writer, nmax)

        return n_events_tot


    def _main_loop(self, pmt_writer, sipm_writer, nmax):
        n_events_tot = 0
        for filename in self.input_files:
            print("Opening", filename, end="... ")
            with tb.open_file(filename, "r") as h5in:

                self._copy_sensor_table(h5in)

                # access RWF
                NEVT, pmtrwf, sipmrwf = self.get_rwf_vectors(h5in)
                # loop over all events in file unless reach nmax
                n_events_tot = self._event_loop(NEVT, pmtrwf, sipmrwf, pmt_writer, sipm_writer, nmax, n_events_tot)

        return n_events_tot


    def _event_loop(self, NEVT, pmtrwf, sipmrwf, write_pmt, write_sipm, nmax, n_events_tot):
        for evt in range(NEVT):
            # Item 1: Deconvolve
            CWF = self.deconv_pmt(pmtrwf[evt])
            # Item 2: Store
            write_pmt (CWF)
            write_sipm(sipmrwf[evt])

            n_events_tot += 1
            self.conditional_print(evt, n_events_tot)

            if self.max_events_reached(nmax, n_events_tot):
                break

        return n_events_tot


    def _copy_sensor_table(self, h5in):
        # Copy sensor table if exists (needed for GATE)
        if 'Sensors' in h5in.root:
            self.sensors_group = cwf_file.create_group(
                cwf_file.root, "Sensors")
            datapmt = h5in.root.Sensors.DataPMT
            datapmt.copy(newparent=self.sensors_group)
            datasipm = h5in.root.Sensors.DataSiPM
            datasipm.copy(newparent=self.sensors_group)



def ISIDORA(argv = sys.argv):
    """ISIDORA DRIVER"""
    CFP = configure(argv)

    files_in    = glob(CFP.FILE_IN)
    files_in.sort()


    fpp = Isidora(run_number  = CFP.RUN_NUMBER,
                  nprint      = CFP.NPRINT,
                  compression = CFP.COMPRESSION,
                  files_in    = files_in,
                  file_out    = CFP.FILE_OUT,
                  n_baseline  = CFP.NBASELINE,
                  thr_trigger = CFP.THR_TRIGGER * units.adc)

    t0 = time()
    nevts = CFP.NEVENTS if not CFP.RUN_ALL else -1
    nevt = fpp.run(nmax=nevts)
    t1 = time()
    dt = t1 - t0

    print("run {} evts in {} s, time/event = {}".format(nevt, dt, dt / nevt))

    return nevts, nevt


class rwf_writer:

    def __init__(self,
                 file,
                 *,
                 group_name      : 'options: RD, BLR',
                 table_name      : 'options: pmtrwf, pmtcwf, sipmrwf',
                 compression     = 'ZLIB4',
                 n_sensors       : 'number of pmts or sipms',
                 waveform_length : 'length of pmt or sipm waveform_length'):

        #rwf_file = tb.open_file(filename, 'w', filters=tbl.filters(compression))
        try:                       rwf_group = getattr          (file.root, group_name)
        except tb.NoSuchNodeError: rwf_group = file.create_group(file.root, group_name)

        self.rwf_table = file.create_earray(rwf_group,
                                                table_name,
                                                atom    = tb.Float32Atom(),
                                                shape   = (0, n_sensors, waveform_length),
                                                filters = tbl.filters(compression))
        self._hdf5_file = file # for close
        self._n_sensors = n_sensors
        self._waveform_length = waveform_length

    def __call__(self, waveform : 'np.array: RWF, CWF, SiPM'):
        self.rwf_table.append(waveform.reshape(1, self._n_sensors, self._waveform_length))

    def close(self):
        self._hdf5_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


    # def _store_wf(self, wf, wf_table, wf_file, wf_group):
    #     """Store WF."""
    #     n_sensors, wl = wf.shape
    #     if wf_table not in wf_group:
    #         # create earray to store cwf
    #         self.wf_tables[wf_table] = wf_file.create_earray(wf_group, wf_table,
    #                                                          atom    = tb.Float32Atom(),
    #                                                          shape   = (0, n_sensors, wl),
    #                                                          filters = tbl.filters(self.compression))

    #     self.wf_tables[wf_table].append(wf.reshape(1, n_sensors, wl))



if __name__ == "__main__":
    ISIDORA(sys.argv)
