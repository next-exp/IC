import sys
import glob
import time
import textwrap

import numpy  as np
import tables as tb

from ..core.configure         import configure
from ..core.system_of_units_c import units
from ..core.ic_types          import xy
from ..io.dst_io              import hits_writer
from ..cities.base_cities     import City
from ..reco.event_model       import PersistentHitCollection
from ..reco.event_model       import Cluster
from ..reco.event_model       import Hit
from ..cities.base_cities     import HitCollectionCity
from ..reco                   import tbl_functions as tbl
from ..reco.tbl_functions     import get_event_numbers_and_timestamps_from_file_name
from ..reco.pmaps_functions   import load_pmaps
from ..filters.s1s2_filter    import s1s2_filter
from ..filters.s1s2_filter    import s2si_filter
from ..filters.s1s2_filter    import S12Selector

class Penthesilea(HitCollectionCity):
    def __init__(self, **kwds):
        super().__init__(**kwds)
        conf = self.conf

        self.drift_v        = conf.drift_v
        self._s1s2_selector = S12Selector(S1_Nmin     = conf.s1_nmin,
                                          S1_Nmax     = conf.s1_nmax,
                                          S1_Emin     = conf.s1_emin,
                                          S1_Emax     = conf.s1_emax,
                                          S1_Lmin     = conf.s1_lmin,
                                          S1_Lmax     = conf.s1_lmax,
                                          S1_Hmin     = conf.s1_hmin,
                                          S1_Hmax     = conf.s1_hmax,
                                          S1_Ethr     = conf.s1_ethr,

                                          S2_Nmin     = conf.s2_nmin,
                                          S2_Nmax     = conf.s2_nmax,
                                          S2_Emin     = conf.s2_emin,
                                          S2_Emax     = conf.s2_emax,
                                          S2_Lmin     = conf.s2_lmin,
                                          S2_Lmax     = conf.s2_lmax,
                                          S2_Hmin     = conf.s2_hmin,
                                          S2_Hmax     = conf.s2_hmax,
                                          S2_NSIPMmin = conf.s2_nsipmmin,
                                          S2_NSIPMmax = conf.s2_nsipmmax,
                                          S2_Ethr     = conf.s2_ethr)

    def run(self):
        self.display_IO_info()

        # with HitCollection_writer(self.output_file, "DST", "w",
        #                           self.compression, "Tracks") as write:
        with tb.open_file(self.output_file, "w",
                          filters = tbl.filters(self.compression)) as h5out:
            write_hits = hits_writer(h5out)
            nevt_in, nevt_out = self._file_loop(write_hits)
        print(textwrap.dedent("""
                              Number of events in : {}
                              Number of events out: {}
                              Ratio               : {}
                              """.format(nevt_in, nevt_out, nevt_out / nevt_in)))
        return nevt_in, nevt_out

    def _file_loop(self, write_hits):
        nevt_in = nevt_out = 0

        for filename in self.input_files:
            print("Opening {filename}".format(**locals()), end="... ")

            try:
                S1s, S2s, S2Sis = load_pmaps(filename)
            except (ValueError, tb.exceptions.NoSuchNodeError):
                print("Empty file. Skipping.")
                continue

            event_numbers, timestamps = get_event_numbers_and_timestamps_from_file_name(filename)

            nevt_in, nevt_out, max_events_reached = self._event_loop(
                event_numbers, timestamps, nevt_in, nevt_out, write_hits, S1s, S2s, S2Sis)

            if max_events_reached:
                print('Max events reached')
                break
            else:
                print("OK")

        return nevt_in, nevt_out

    def _event_loop(self, event_numbers, timestamps,
                    nevt_in, nevt_out, write_hits, S1s, S2s, S2Sis):
        max_events_reached = False
        for evt_number, evt_time in zip(event_numbers, timestamps):
            nevt_in += 1
            if self.max_events_reached(nevt_in):
                max_events_reached = True
                break
            S1   = S1s  .get(evt_number, {})
            S2   = S2s  .get(evt_number, {})
            S2Si = S2Sis.get(evt_number, {})

            f1 = s1s2_filter(self._s1s2_selector, S1, S2, S2Si)
            f2 = s2si_filter(S2Si)
            if not f1 or not f2:
                continue
            nevt_out += 1

            evt = self._create_hits_event(evt_number, evt_time, S1, S2, S2Si)
            write_hits(evt)

            self.conditional_print(evt, nevt_in)

        return nevt_in, nevt_out, max_events_reached

    def _create_hits_event(self, evt_number, evt_time, S1, S2, Si):

        hitc = PersistentHitCollection(evt_number, evt_time * 1e-3)
        #hitc.evt   = evt_number
        #hitc.time  = evt_time * 1e-3 # s

        t, e = next(iter(S1.values()))
        S1t  = t[np.argmax(e)]

        S2, Si = self.rebin_s2(S2, Si)

        npeak = 0
        for peak_no, (t_peak, e_peak) in sorted(S2.items()):
            si = Si[peak_no]
            for slice_no, (t_slice, e_slice) in enumerate(zip(t_peak, e_peak)):
                clusters = self.compute_xy_position(si, slice_no)
                es       = self.split_energy(e_slice, clusters)
                z        = (t_slice - S1t) * units.ns * self.drift_v
                for c, e in zip(clusters, es):
                    hit       = Hit(npeak, c, z, e)
                    hitc.hits.append(hit)
            npeak += 1

        return hitc
