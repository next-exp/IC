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
from ..io.pmap.io             import load_pmaps
from ..filters.s1s2_filter    import s1s2_filter
from ..filters.s1s2_filter    import s2si_filter
from ..filters.s1s2_filter    import S12Selector

class Penthesilea(HitCollectionCity):
    def __init__(self, **kwds):
        super().__init__(**kwds)
        conf = self.conf

        self.drift_v        = conf.drift_v
        self._s1s2_selector = S12Selector(**kwds)

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
                S1_dict, S2_dict, S2Si_dict = load_pmaps(filename)
            except (ValueError, tb.exceptions.NoSuchNodeError):
                print("Empty file. Skipping.")
                continue

            event_numbers, timestamps = get_event_numbers_and_timestamps_from_file_name(filename)

            nevt_in, nevt_out, max_events_reached = self._event_loop(
                event_numbers, timestamps, nevt_in, nevt_out, write_hits, S1_dict, S2_dict, S2Si_dict)

            if max_events_reached:
                print('Max events reached')
                break
            else:
                print("OK")

        return nevt_in, nevt_out

    def _event_loop(self, event_numbers, timestamps,
                    nevt_in, nevt_out, write_hits, S1_dict, S2_dict, S2Si_dict):
        max_events_reached = False
        for evt_number, evt_time in zip(event_numbers, timestamps):
            nevt_in += 1
            if self.max_events_reached(nevt_in):
                max_events_reached = True
                break
            s1   = S1_dict  .get(evt_number, {})
            s2   = S2_dict  .get(evt_number, {})
            s2si = S2Si_dict.get(evt_number, {})

            f1 = s1s2_filter(self._s1s2_selector, s1, s2, s2si)
            f2 = s2si_filter(s2si)
            if not f1 or not f2:
                continue
            nevt_out += 1

            evt = self._create_hits_event(evt_number, evt_time, s1, s2, s2si)
            write_hits(evt)

            self.conditional_print(evt, nevt_in)

        return nevt_in, nevt_out, max_events_reached

    def _create_hits_event(self, evt_number, evt_time,
                           s1, s2, s2si):

        hitc = PersistentHitCollection(evt_number, evt_time * 1e-3)
        #hitc.evt   = evt_number
        #hitc.time  = evt_time * 1e-3 # s

        # in order to compute z one needs to define one S1
        # for time reference. By default the filter will only
        # take events with exactly one s1. Otherwise, the
        # convention is to take the first peak in the S1 object
        # as reference.

        # peak  = S1.peak_waveform(0)
        # t, e = next(iter(S1.values()))
        # S1t  = t[np.argmax(e)]

        s1_t = s1.peak_waveform(0).tpeak

        #S2, Si = self.rebin_s2(S2, Si) TODO REVISE

        npeak = 0
        for peak_no, (t_peak, e_peak) in sorted(s2.s2d.items()):
            si = s2si.s2sid[peak_no]
            for slice_no, (t_slice, e_slice) in enumerate(zip(t_peak, e_peak)):
                clusters = self.compute_xy_position(si, slice_no)
                es       = self.split_energy(e_slice, clusters)
                z        = (t_slice - s1_t) * units.ns * self.drift_v
                for c, e in zip(clusters, es):
                    hit       = Hit(npeak, c, z, e)
                    hitc.hits.append(hit)
            npeak += 1

        return hitc
