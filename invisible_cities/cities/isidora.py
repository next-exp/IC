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
from .. reco                   import tbl_functions as tbl

from .. io.       mcinfo_io    import       mc_info_writer
from .. io.          rwf_io    import           rwf_writer
from .. io.run_and_event_io    import run_and_event_writer

from .  base_cities import DeconvolutionCity
from .  base_cities import EventLoop


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

        self.cnt.init(n_events_tot = 0)
        self.sp = self.get_sensor_params(self.input_files[0])

    def event_loop(self, NEVT, dataVectors):
        """actions:
        1. loops over all the events in each file.
        2. write event/run to file
        3. compute deconvoluted functions and write them to file
        """
        write = self.writers
        pmtrwf       = dataVectors.pmt
        sipmrwf      = dataVectors.sipm
        mc_info      = dataVectors.mc
        events_info  = dataVectors.events

        for evt in range(NEVT):
            self.conditional_print(evt, self.cnt.n_events_tot)

            what_next = self.event_range_step()
            if what_next is EventLoop.skip_this_event: continue
            if what_next is EventLoop.terminate_loop : break
            self.cnt.n_events_tot += 1

            CWF = self.deconv_pmt(pmtrwf[evt])  # deconvolution

            # write stuff
            event, timestamp = self.event_and_timestamp(evt, events_info)
            write.run_and_event(self.run_number, event, timestamp)
            write.pmt (CWF)
            write.sipm(sipmrwf[evt])
            if self.monte_carlo:
                write.mc(mc_info, self.cnt.n_events_tot)

    def get_writers(self, h5out):
        RWF = partial(rwf_writer,  h5out,   group_name='BLR')
        writers = Namespace(
            run_and_event = run_and_event_writer(h5out),
            pmt           = RWF(table_name='pmtcwf' , n_sensors=self.sp.NPMT , waveform_length=self.sp.PMTWL),
            sipm          = RWF(table_name='sipmrwf' , n_sensors=self.sp.NSIPM , waveform_length=self.sp.SIPMWL),
            mc            = mc_info_writer(h5out) if self.monte_carlo else None,
            )

        return writers

    def write_parameters(self, h5out):
        pass

    def display_IO_info(self):
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
