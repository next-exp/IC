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

from .. io.mc_io               import mc_track_writer
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
        self.cnt.init_counter('n_events_tot')
        self.sp = self.get_sensor_params(self.input_files[0])

    def event_loop(self, NEVT, pmtrwf, sipmrwf, mc_tracks, events_info):
        """actions:
        1. loops over all the events in each file.
        2. write event/run to file
        3. compute deconvoluted functions and write them to file
        """
        write = self.writers
        for evt in range(NEVT):
            CWF = self.deconv_pmt(pmtrwf[evt])  # deconvolution

            # write stuff
            event, timestamp = self.event_and_timestamp(evt, events_info)
            write.run_and_event(self.run_number, event, timestamp)
            write.pmt (CWF)
            write.sipm(sipmrwf[evt])
            if self.monte_carlo:
                write.mc(mc_tracks, self.cnt.counter_value('n_events_tot'))

            # conditional print and exit of loop condition
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
            mc            =      mc_track_writer(h5out) if self.monte_carlo else None,
            )

        return writers

    def display_IO_info(self):
        """display info"""
        super().display_IO_info()
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
