import sys

from argparse import Namespace

import numpy  as np
import tables as tb

from .. core.configure         import configure
from .. core.system_of_units_c import units

from .. io.mc_io              import mc_track_writer
from .. io.pmap_io            import pmap_writer
from .. io.run_and_event_io   import run_and_event_writer
from .. reco                  import tbl_functions as tbl
from .. reco.params           import S12Params as S12P
from .. core.ic_types         import minmax

from .  base_cities  import PmapCity


class Irene(PmapCity):
    """Perform fast processing from raw data to pmaps.

    Raw data pmtrwf and sipmrwf.
    It is optimized for speed (use of CYTHON functions) and intended
    for fast processing of data.
    """
    def __init__(self, **kwds):
        PmapCity.__init__(self, **kwds)
        self.check_s1s2_params()

    def run(self, print_empty=True):
        self.display_IO_info()
        sensor_params = self.get_sensor_params(self.input_files[0])
        print(sensor_params)
        
        with tb.open_file(self.output_file, "w",
                          filters = tbl.filters(self.compression)) as h5out:
            writers = Namespace(
                pmap          =          pmap_writer(h5out),
                run_and_event = run_and_event_writer(h5out),
                mc            =      mc_track_writer(h5out) if self.monte_carlo else None,
            )
            self.write_deconv_params(h5out)
            n_events_tot, n_empty_events = self._file_loop(writers)
        if print_empty:
            print('Energy plane empty events (skipped) = {}'.format(
                   n_empty_events))
        return n_events_tot, n_empty_events

    def _file_loop(self, writers):
        n_events_tot, n_empty_events = 0, 0
        for filename in self.input_files:
            print("Opening", filename, end="... ")
            with tb.open_file(filename, "r") as h5in:
                # access RWF
                NEVT, pmtrwf, sipmrwf, _ = self.get_rwf_vectors(h5in)
                events_info = self.get_run_and_event_info(h5in)
                # loop over all events in file unless reach nmax
                (n_events_tot,
                 n_empty_events) = self._event_loop(NEVT, pmtrwf, sipmrwf, events_info,
                                                    writers,
                                                    n_events_tot, n_empty_events, h5in)
        return n_events_tot, n_empty_events

    def _event_loop(self, NEVT, pmtrwf, sipmrwf, events_info,
                    write,
                    n_events_tot, n_empty_events, h5in):
        for evt in range(NEVT):
            if self.monte_carlo:
                write.mc(h5in.root.MC.MCTracks, n_events_tot)

            s1_ene, s1_indx, s2_ene, s2_indx, csum = self.pmt_transformation(pmtrwf[evt])

            # In a few rare cases s2_ene is empty
            # this is due to empty energy plane events
            # a protection is set to avoid a crash
            if np.sum(s2_ene) == 0:
                n_empty_events += 1
                continue

            sipmzs = self.calibrated_signal_sipm(sipmrwf[evt])
            S1, S2, Si = self.pmaps(s1_ene, s1_indx, s2_ene, s2_indx, csum, sipmzs)

            event, timestamp = self.event_and_timestamp(evt, events_info)
            # write to file
            write.pmap         (event, S1, S2, Si)
            write.run_and_event(self.run_number, event, timestamp)
            n_events_tot += 1
            self.conditional_print(evt, n_events_tot)
            if self.max_events_reached(n_events_tot):
                break
        return n_events_tot, n_empty_events

    def pmt_transformation(self, RWF):
            # deconvolve
            CWF = self.deconv_pmt(RWF)
            # calibrated PMT sum
            csum, csum_mau = self.calibrated_pmt_sum(CWF)
            #ZS sum for S1 and S2
            s1_ene, s1_indx = self.csum_zs(csum_mau, threshold =
                                           self.thr_csum_s1)
            s2_ene, s2_indx = self.csum_zs(csum,     threshold =
                                           self.thr_csum_s2)
            return s1_ene, s1_indx, s2_ene, s2_indx, csum

    def pmaps(self, s1_ene, s1_indx, s2_ene, s2_indx, csum, sipmzs):
        S1, S2 = self.find_S12(s1_ene, s1_indx,   s2_ene, s2_indx)
        S1     = self.correct_S1_ene(S1, csum)
        Si     = self.find_S2Si(S2, sipmzs)
        return S1, S2, Si

    def display_IO_info(self):
        PmapCity.display_IO_info(self)
        print("""
                 S1 parameters {}""" .format(self.s1_params))
        print("""
                 S2 parameters {}""" .format(self.s2_params))
        print("""
                 S2Si parameters
                 threshold min charge per SiPM = {s.thr_sipm} pes
                 threshold min charge in  S2   = {s.thr_sipm_s2} pes
                          """.format(s=self))
