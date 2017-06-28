import sys

from argparse import Namespace

from glob import glob
from time import time

import tables as tb

from .. core.configure         import configure
from .. core.system_of_units_c import units

from .. reco        import tbl_functions as tbl
from .. io.rwf_io   import rwf_writer
from .  base_cities import DeconvolutionCity

from .  diomira     import Diomira


class Isidora(DeconvolutionCity):
    """
    The city of ISIDORA performs a fast processing from raw data
    (pmtrwf and sipmrwf) to BLR wavefunctions.
    It is optimized for speed (use of CYTHON functions) and intended
    for fast processing of data.
    """

    go = Diomira.go

    def run(self):
        self.display_IO_info()
        sensor_params = self.get_sensor_params(self.input_files[0])
        print(sensor_params)
        
        with tb.open_file(self.output_file, "w",
                          filters = tbl.filters(self.compression)) as h5out:

            writers = Namespace(
                pmt  = rwf_writer(h5out,
                                  group_name      = 'BLR',
                                  table_name      = 'pmtcwf',
                                  n_sensors       = sensor_params.NPMT,
                                  waveform_length = sensor_params.PMTWL),
                sipm = rwf_writer(h5out,
                                  group_name      = 'BLR',
                                  table_name      = 'sipmrwf',
                                  n_sensors       = sensor_params.NSIPM,
                                  waveform_length = sensor_params.SIPMWL)
            )

            n_events_tot = self._file_loop(writers)

        return n_events_tot


    def _file_loop(self, writers):
        n_events_tot = 0
        for filename in self.input_files:
            print("Opening", filename, end="... ")
            with tb.open_file(filename, "r") as h5in:

                self._copy_sensor_table(h5in)

                # access RWF
                NEVT, pmtrwf, sipmrwf, _ = self.get_rwf_vectors(h5in)
                # loop over all events in file unless reach nmax
                n_events_tot = self._event_loop(NEVT, pmtrwf, sipmrwf, writers, n_events_tot)

        return n_events_tot


    def _event_loop(self, NEVT, pmtrwf, sipmrwf, write, n_events_tot):
        for evt in range(NEVT):
            # Item 1: Deconvolve
            CWF = self.deconv_pmt(pmtrwf[evt])
            # Item 2: Store
            write.pmt (CWF)
            write.sipm(sipmrwf[evt])

            n_events_tot += 1
            self.conditional_print(evt, n_events_tot)

            if self.max_events_reached(n_events_tot):
                break

        return n_events_tot

    def _copy_sensor_table(self, h5in):
        # Copy sensor table if exists (needed for GATE)
        if 'Sensors' in h5in.root:
            self.sensors_group = self.output_file.create_group(
                self.output_file.root, "Sensors")
            datapmt = h5in.root.Sensors.DataPMT
            datapmt.copy(newparent=self.sensors_group)
            datasipm = h5in.root.Sensors.DataSiPM
            datasipm.copy(newparent=self.sensors_group)
