"""
code: irene.py
description: perform fast processing from raw data to pmaps.
credits: see ic_authors_and_legal.rst in /doc

last revised: JJGC, 12-July-2017
"""
import sys

from argparse import Namespace

import numpy  as np
import tables as tb

from .. core.configure         import configure
from .. core.system_of_units_c import units

from .. io.mc_io               import mc_track_writer
from .. io.pmap_io             import pmap_writer
from .. io.run_and_event_io    import run_and_event_writer
from .. reco                   import tbl_functions as tbl
from .. reco.params            import S12Params as S12P
from .. types.ic_types         import minmax


from .  base_cities  import PmapCity


class Irene(PmapCity):
    """Perform fast processing from raw data to pmaps."""

    def __init__(self, **kwds):
        """actions:
        1. inits base city
        2. inits counters
        3. gets sensor parameters

        """
        super().__init__(**kwds)
        self.cnt.set_name('irene')
        self.cnt.set_counter('nmax', value=self.conf.nmax)
        self.cnt.init_counters(('n_events_tot', 'n_empty_events'))

        self.sp = self.get_sensor_params(self.input_files[0])

    # def run(self):
    #     self.display_IO_info()
    #
    #
    #     with tb.open_file(self.output_file, "w",
    #                       filters = tbl.filters(self.compression)) as h5out:
    #
    #         self.write_deconv_params(h5out)
    #         writers = self._get_writers()
    #         self._file_loop(writers)
    #
    #
    #     return self.cnt

    # def file_loop(self):
    #     """
    #     actions:
    #     1. inite counters
    #     2. access RWF vectors for PMT and SiPMs
    #     3. access run and event info
    #     4. access MC track info
    #     5. call event_loop
    #     """
    #     # import pdb; pdb.set_trace()
    #
    #
    #     for filename in self.input_files:
    #         print("Opening", filename, end="... ")
    #         with tb.open_file(filename, "r") as h5in:
    #
    #             NEVT, pmtrwf, sipmrwf, _ = self.get_rwf_vectors(h5in)
    #             events_info              = self.get_run_and_event_info(h5in)
    #             mc_tracks                = self.get_mc_tracks(h5in)
    #
    #             self.event_loop(NEVT, pmtrwf, sipmrwf, mc_tracks, events_info)
        # return self.cnt

    def event_loop(self, NEVT, pmtrwf, sipmrwf, mc_tracks, events_info):
        """actions:
        1. loops over all the events in each file.
        2. write MC tracks on file
        3. write event/run to file
        3. compute PMAPS and write them to file
        """

        write = self.writers
        for evt in range(NEVT):
            # calibrated sum in PMTs
            s12sum, calsum = self.pmt_transformation(pmtrwf[evt])

            if np.sum(s12sum.s2_ene) == 0: # ocasional but rare empty events
                self.cnt.increment_counter('n_empty_events')
                continue
            # calibrated sum in SiPMs
            sipmzs = self.calibrated_signal_sipm(sipmrwf[evt])
            # pmaps
            s1, s2, s2si = self.pmaps(s12sum.s1_indx,
                                      s12sum.s2_indx,
                                      calsum.csum,
                                      sipmzs)

            # write stuff
            event, timestamp = self.event_and_timestamp(evt, events_info)
            write.pmap         (event, s1, s2, s2si)
            write.run_and_event(self.run_number, event, timestamp)
            if self.monte_carlo:
                write.mc(mc_tracks, self.cnt.counter_value('n_events_tot'))

            self.conditional_print(evt, self.cnt.counter_value('n_events_tot'))
            if self.max_events_reached(self.cnt.counter_value('n_events_tot')):
                break
            else:
                self.cnt.increment_counter('n_events_tot')

    def write_parameters(self, h5out):
        """Write deconvolution parameters to output file"""
        self.write_deconv_params(h5out)

    def get_writers(self, h5out):
        """Get the writers needed by Irene"""
        writers = Namespace(
        pmap          =          pmap_writer(h5out),
        run_and_event = run_and_event_writer(h5out),
        mc            =      mc_track_writer(h5out) if self.monte_carlo else None,
        )
        return writers

    def display_IO_info(self):
        """display info"""
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
