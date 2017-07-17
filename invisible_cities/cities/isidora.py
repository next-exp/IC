"""
code: isidora.py
description: performs a fast processing from raw data
(pmtrwf and sipmrwf) to BLR wavefunctions.
credits: see ic_authors_and_legal.rst in /doc

last revised: JJGC,July-2017
"""

import sys

from argparse import Namespace

from glob import glob
from time import time
from functools import partial

import tables as tb

from .. core.configure         import configure
from .. core.system_of_units_c import units
from .. reco        import tbl_functions as tbl

from .. io.run_and_event_io    import run_and_event_writer
from .. io.rwf_io   import rwf_writer
from .  base_cities import DeconvolutionCity


class Isidora(DeconvolutionCity):
    """
    The city of ISIDORA performs a fast processing from raw data
    (pmtrwf and sipmrwf) to BLR wavefunctions.

    """
    def __init__(self, **kwds):
        """Isidora Init:
        1. inits base city
        2. inits counters
        3. gets sensor parameters

        """

        super().__init__(**kwds)
        self.cnt.set_name('isidora')
        self.cnt.set_counter('nmax', value=self.conf.nmax)

        self.sp = self.get_sensor_params(self.input_files[0])

    # def run(self):
    #     self.display_IO_info()
    #     sensor_params = self.get_sensor_params(self.input_files[0])
    #     print(sensor_params)
    #
    #     with tb.open_file(self.output_file, "w",
    #                       filters = tbl.filters(self.compression)) as h5out:
    #
    #         writers = Namespace(
    #             pmt  = rwf_writer(h5out,
    #                               group_name      = 'BLR',
    #                               table_name      = 'pmtcwf',
    #                               n_sensors       = sensor_params.NPMT,
    #                               waveform_length = sensor_params.PMTWL),
    #             sipm = rwf_writer(h5out,
    #                               group_name      = 'BLR',
    #                               table_name      = 'sipmrwf',
    #                               n_sensors       = sensor_params.NSIPM,
    #                               waveform_length = sensor_params.SIPMWL)
    #         )
    #
    #         n_events_tot = self._file_loop(writers)
    #
    #     return n_events_tot


    def file_loop(self):
        """
        actions:
        1. inite counters
        2. access RWF vectors for PMT and SiPMs
        3. access run and event info
        4. call event_loop
        """
        self.cnt.init_counter('n_events_tot')
        for filename in self.input_files:
            print("Opening", filename, end="... ")
            with tb.open_file(filename, "r") as h5in:

                self._copy_sensor_table(h5in)

                NEVT, pmtrwf, sipmrwf, _ = self.get_rwf_vectors(h5in)
                events_info              = self.get_run_and_event_info(h5in)

                self.event_loop(NEVT, pmtrwf, sipmrwf, events_info)


    def event_loop(self, NEVT, pmtrwf, sipmrwf, events_info):
        """actions:
        1. loops over all the events in each file.
        2. write event/run to file
        3. compute deconvoluted functions and write them to file
        """
        write = self.writers
        for evt in range(NEVT):
            CWF = self.deconv_pmt(pmtrwf[evt])
            event, timestamp = self.event_and_timestamp(evt, events_info)
            write.run_and_event(self.run_number, event, timestamp)
            write.pmt (CWF)
            write.sipm(sipmrwf[evt])

            self.conditional_print(evt, self.cnt.counter_value('n_events_tot'))
            if self.max_events_reached(self.cnt.counter_value('n_events_tot')):
                break
            else:
                self.cnt.increment_counter('n_events_tot')


    def get_writers(self, h5out):
        """Get the writers needed by Isidora"""

        RWF = partial(rwf_writer,  h5out,   group_name='BLR')
        writers = Namespace(
            run_and_event = run_and_event_writer(h5out),
            pmt  = RWF(table_name='pmtcwf' , n_sensors=self.sp.NPMT , waveform_length=self.sp.PMTWL),
            sipm  = RWF(table_name='sipmrwf' , n_sensors=self.sp.NSIPM , waveform_length=self.sp.SIPMWL),
            )
        #     pmt  = rwf_writer(h5out,
        #                       group_name      = 'BLR',
        #                       table_name      = 'pmtcwf',
        #                       n_sensors       = sensor_params.NPMT,
        #                       waveform_length = sensor_params.PMTWL),
        #     sipm = rwf_writer(h5out,
        #                       group_name      = 'BLR',
        #                       table_name      = 'sipmrwf',
        #                       n_sensors       = sensor_params.NSIPM,
        #                       waveform_length = sensor_params.SIPMWL)
        # )


        return writers

    def display_IO_info(self):
        """display info"""
        print(self.sp)


    def _copy_sensor_table(self, h5in):
        # Copy sensor table if exists (needed for GATE)
        if 'Sensors' in h5in.root:
            self.sensors_group = self.output_file.create_group(
                self.output_file.root, "Sensors")
            datapmt = h5in.root.Sensors.DataPMT
            datapmt.copy(newparent=self.sensors_group)
            datasipm = h5in.root.Sensors.DataSiPM
            datasipm.copy(newparent=self.sensors_group)
