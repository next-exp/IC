"""
code: irene.py
description: perform fast processing from raw data to pmaps.
credits: see ic_authors_and_legal.rst in /doc

last revised: JJGC, 12-July-2017
"""
from argparse import Namespace

import numpy  as np

from .. io.mc_io            import mc_track_writer
from .. io.pmaps_io         import pmap_writer
from .. io.run_and_event_io import run_and_event_writer

from .  base_cities  import PmapCity
from .  base_cities  import EventLoop


class Irene(PmapCity):
    """Perform fast processing from raw data to pmaps."""

    def __init__(self, **kwds):
        """actions:
        1. inits base city
        2. inits counters
        3. gets sensor parameters

        """
        super().__init__(**kwds)

        self.cnt.init(n_events_tot                 = 0,
                      n_empty_events               = 0,
                      n_empty_events_s2_ene_eq_0   = 0,
                      n_empty_events_s1_indx_empty = 0,
                      n_empty_events_s2_indx_empty = 0,
                      n_empty_pmaps                = 0)

        self.sp = self.get_sensor_params(self.input_files[0])

    def event_loop(self, NEVT, dataVectors):
        """actions:
        1. loops over all the events in each file.
        2. write MC tracks on file
        3. write event/run to file
        3. compute PMAPS and write them to file
        """

        write       = self.writers
        pmtrwf      = dataVectors.pmt
        sipmrwf     = dataVectors.sipm
        mc_tracks   = dataVectors.mc
        events_info = dataVectors.events

        for evt in range(NEVT):
            self.conditional_print(evt, self.cnt.n_events_tot)

            what_next = self.event_range_step()
            if what_next is EventLoop.skip_this_event: continue
            if what_next is EventLoop.terminate_loop : break
            self.cnt.n_events_tot += 1

            # calibrated sum in PMTs
            s12sum, cal_cwf, _ = self.pmt_transformation(pmtrwf[evt])

            if not self.check_s12(s12sum): # ocasional but rare empty events
                self.cnt.n_empty_events += 1
                continue

            # calibrated sum in SiPMs
            sipmzs = self.calibrate_sipms(sipmrwf[evt])

            # pmaps
            pmap = self.pmaps(s12sum.s1_indx, s12sum.s2_indx,
                              cal_cwf.ccwf, sipmzs)

            if not pmap.s1s and not pmap.s2s:
                self.cnt.n_empty_pmaps += 1
                continue

            # write stuff
            event, timestamp = self.event_and_timestamp(evt, events_info)
            write.pmap         (pmap, event)
            write.run_and_event(self.run_number, event, timestamp)
            if self.monte_carlo:
                write.mc(mc_tracks, event)

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
        mc            =      mc_track_writer(h5out) if self.monte_carlo else None,
        pmap          =          pmap_writer(h5out))
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
