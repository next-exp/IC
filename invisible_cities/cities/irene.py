"""
code: irene.py
description: perform fast processing from raw data to pmaps.
credits: see ic_authors_and_legal.rst in /doc

last revised: JJGC, 12-July-2017
"""
from argparse import Namespace

import numpy  as np
import tables as tb
import warnings

from os.path                import expandvars

from .. io.mcinfo_io        import mc_info_writer
from .. io.pmaps_io         import pmap_writer
from .. io.run_and_event_io import run_and_event_writer
from .. io.trigger_io       import trigger_writer

from .. reco                import tbl_functions as tbl

from .  base_cities  import PmapCity
from .  base_cities  import EventLoop


class Irene(PmapCity):
    """Perform fast processing from raw data to pmaps."""

    parameters = tuple("""trg1_code trg2_code split_triggers
                       file_out2""".split())

    def __init__(self, **kwds):
        """actions:
        1. inits base city
        2. inits counters
        3. gets sensor parameters

        """
        super().__init__(**kwds)

        self.split_triggers = self.conf.split_triggers
        if self.conf.split_triggers:
            self.output_file2 = expandvars(self.conf.file_out2)
            self.trg1_code      = self.conf.trg1_code
            self.trg2_code      = self.conf.trg2_code

        self.cnt.init(n_events_tot                 = 0,
                      n_empty_events               = 0,
                      n_empty_events_s2_ene_eq_0   = 0,
                      n_empty_events_s1_indx_empty = 0,
                      n_empty_events_s2_indx_empty = 0,
                      n_empty_pmaps                = 0)

        self.sp = self.get_sensor_params(self.input_files[0])

    def run(self):
        """The (base) run method of a city does the following chores:
        1. Calls a display_IO_info() function (to be provided by the concrete cities)
        2. open the output file
        3. Writes any desired parameters to output file (must be implemented by cities)
        4. gets the writers for the specific city.
        5. Pass the writers to the file_loop() method.
        6. returns the counter dictionary.
        """
        self.display_IO_info()

        with tb.open_file(self.output_file, "w",
                          filters = tbl.filters(self.compression)) as h5out1:
            self.write_parameters(h5out1)
            self.writers1 = self.get_writers(h5out1)
            self.writers2 = None

            if self.split_triggers:
                with tb.open_file(self.output_file2, "w",
                          filters = tbl.filters(self.compression)) as h5out2:
                    self.write_parameters(h5out2)
                    self.writers2 = self.get_writers(h5out2)
                    self.file_loop()
            else:
                self.file_loop()

    def event_loop(self, NEVT, dataVectors):
        """actions:
        1. loops over all the events in each file.
        2. write MC tracks on file
        3. write event/run to file
        3. compute PMAPS and write them to file
        """

        write1       = self.writers1
        write2       = self.writers2
        pmtrwf       = dataVectors.pmt
        sipmrwf      = dataVectors.sipm
        mc_info      = dataVectors.mc
        events_info  = dataVectors.events
        trg_types    = dataVectors.trg_type
        trg_channels = dataVectors.trg_channels

        for evt in range(NEVT):
            self.conditional_print(evt, self.cnt.n_events_tot)

            what_next = self.event_range_step()
            if what_next is EventLoop.skip_this_event: continue
            if what_next is EventLoop.terminate_loop : break
            self.cnt.n_events_tot += 1

            # calibrated sum in PMTs
            pmt_wfs = self.mask_pmts(pmtrwf[evt])
            s12sum, cal_cwf, _ = self.pmt_transformation(pmt_wfs)

            if not self.check_s12(s12sum): # ocasional but rare empty events
                self.cnt.n_empty_events += 1
                continue

            # calibrated sum in SiPMs
            sipm_wfs = self.mask_sipms(sipmrwf[evt])
            sipmzs = self.calibrate_sipms(sipm_wfs)

            # pmaps
            pmap = self.pmaps(s12sum.s1_indx, s12sum.s2_indx,
                              cal_cwf.ccwf, sipmzs)

            if not pmap.s1s and not pmap.s2s:
                self.cnt.n_empty_pmaps += 1
                continue

            # write stuff
            event, timestamp = self.event_and_timestamp(evt, events_info)
            trg_type         = self.trigger_type    (evt, trg_types)
            trg_channel      = self.trigger_channels(evt, trg_channels)

            # If there is trigger information and split is true => write 2 files
            if self.split_triggers and trg_types:
                if trg_type == self.trg1_code:
                    self.write_event(write1, pmap, mc_info, event, timestamp,
                                     trg_type, trg_channel)
                elif trg_type == self.trg2_code:
                    self.write_event(write2, pmap, mc_info, event, timestamp,
                                     trg_type, trg_channel)
                else:
                    message = f"Event {event} has an unknown trigger type ({trg_type})"
                    warnings.warn(message)
            else:
                self.write_event(write1, pmap, mc_info, event, timestamp,
                                 trg_type, trg_channel)


    def write_event(self, writers, pmap, mc_info, event, timestamp,
                    trg_type, trg_channel):
        writers.pmap         (pmap, event)
        writers.run_and_event(self.run_number, event, timestamp)
        if self.monte_carlo:
            writers.mc(mc_info, event)
        writers.trigger(trg_type, trg_channel)


    def check_s12(self, s12sum):
        """Checks for ocassional empty events, characterized by null s2_energy
        or empty index list for s1/s2

        """
        if  np.sum(s12sum.s2_ene) == 0:
            self.cnt.n_empty_events_s2_ene_eq_0 += 1
            return False
        elif np.sum(s12sum.s1_indx) == 0:
            self.cnt.n_empty_events_s1_indx_empty += 1
            return False
        elif np.sum(s12sum.s2_indx) == 0:
            self.cnt.n_empty_events_s2_indx_empty += 1
            return False
        else:
            return True

    def write_parameters(self, h5out):
        """Write deconvolution parameters to output file"""
        self.write_deconv_params(h5out)

    def get_writers(self, h5out):
        writers = Namespace(
        run_and_event = run_and_event_writer(h5out),
        mc            = mc_info_writer(h5out) if self.monte_carlo else None,
        pmap          = pmap_writer(h5out),
        trigger       = trigger_writer(h5out, len(self.pmt_active)))
        return writers

    def display_IO_info(self):
        super().display_IO_info()
        print(self.sp)
        print("""
                 S1 parameters {}""" .format(self.s1_params))
        print("""
                 S2 parameters {}""" .format(self.s2_params))
        print("""
                 S2Si parameters
                 threshold min charge per SiPM = {s.thr_sipm} pes
                 threshold min charge in  S2   = {s.thr_sipm_s2} pes
                          """.format(s=self))
