"""
code: penthesila.py
description: Read PMAPS and produces hits and beyond
credits: see ic_authors_and_legal.rst in /doc

last revised: JJGC, July-2017
"""

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
from ..cities.base_cities     import HitCity
from ..reco                   import tbl_functions as tbl

from ..filters.s1s2_filter    import s1s2_filter
from ..filters.s1s2_filter    import s2si_filter
from ..filters.s1s2_filter    import S12Selector

class Penthesilea(HitCity):
    """Read PMAPS and produces hits and beyond"""
    def __init__(self, **kwds):
        """actions:
        1. inits base city
        2. inits counters
        3. inits s1s2 selector

        """
        super().__init__(**kwds)
        conf = self.conf
        self.cnt.set_name('penthesilea')
        self.cnt.set_counter('nmax', value=conf.nmax)
        self.cnt.init_counters(('n_events_tot', 'nevt_out'))
        self.drift_v        = conf.drift_v
        self._s1s2_selector = S12Selector(**kwds)


    def event_loop(self, event_numbers, timestamps, s1_dict, s2_dict, s2si_dict):
        """actions:
        1. loops over all PMAPS
        2. filter pmaps
        3. write hit_event
        """

        write_hits = self.writers
        for evt_number, evt_time in zip(event_numbers, timestamps):
            # Count events in and break if necessary before filtering
            if self.max_events_reached(self.cnt.counter_value('n_events_tot')):
                break
            else:
                self.cnt.increment_counter('n_events_tot')

            # get pmaps
            s1, s2, s2si = self. get_pmaps_from_dicts(s1_dict,
                                                      s2_dict,
                                                      s2si_dict,
                                                      evt_number)
            # filtering
            # loop event away if any signal (s1, s2 or s2si) not present
            if s1 == None or s2 == None or s2si == None:
                continue
            # filters in s12 and s2si
            f1 = s1s2_filter(self._s1s2_selector, s1, s2, s2si)
            f2 = s2si_filter(s2si)
            if not f1 or not f2:
                continue
            # event passed selection: increment counter and write
            self.cnt.increment_counter('nevt_out')

            evt = self._create_hits_event(evt_number, evt_time, s1, s2, s2si)
            write_hits(evt)
            self.conditional_print(self.cnt.counter_value('n_events_tot'),
            self.cnt.counter_value('nevt_out'))


    def _create_hits_event(self, evt_number, evt_time,
                           s1, s2, s2si):

        hitc = PersistentHitCollection(evt_number, evt_time * 1e-3)

        # in order to compute z one needs to define one S1
        # for time reference. By default the filter will only
        # take events with exactly one s1. Otherwise, the
        # convention is to take the first peak in the S1 object
        # as reference.

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
