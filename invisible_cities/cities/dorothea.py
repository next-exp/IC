"""
code: dorothea.py
description: create a lightweight DST.
credits: see ic_authors_and_legal.rst in /doc

last revised: JJGC, July-2017
"""

import numpy  as np
import tables as tb

from .. core.configure         import configure
from .. core.system_of_units_c import units

from .. io.kdst_io              import kr_writer
from .. evm.ic_containers       import PmapVectors

from .. reco                   import tbl_functions   as tbl
from .. reco                   import pmaps_functions_c as pmp

from .. filters.s1s2_filter    import s1s2_filter
from .. filters.s1s2_filter    import S12Selector

from .  base_cities            import KrCity
from .  base_cities            import EventLoop


class Dorothea(KrCity):
    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.cnt.set_name('dorothea')
        self.cnt.init_counters(('n_events_tot',
                                'nevt_out',
                                'n_events_not_s1',
                                'n_events_not_s2',
                                'n_events_not_s2si',
                                'n_events_rejected_filter',
                                'n_events_more_than_1_cluster'
                                ))

        self.drift_v = self.conf.drift_v
        self._s1s2_selector = S12Selector(**kwds)


    def event_loop(self, pmapVectors):
        """actions:
        1. loops over all PMAPS
        2. filter pmaps
        3. write kr_event
        """

        write_kr = self.writers
        event_numbers= pmapVectors.events
        timestamps = pmapVectors.timestamps
        s1_dict = pmapVectors.s1
        s2_dict = pmapVectors.s2
        s2si_dict = pmapVectors.s2si

        for evt_number, evt_time in zip(event_numbers, timestamps):
            # Count events in and break if necessary before filtering
            what_next = self.event_range_step()
            if what_next is EventLoop.skip_this_event: continue
            if what_next is EventLoop.terminate_loop : break
            self.cnt.increment_counter('n_events_tot')
            # get pmaps
            s1, s2, s2si = self. get_pmaps_from_dicts(s1_dict,
                                                      s2_dict,
                                                      s2si_dict,
                                                      evt_number)
            # filtering
            # loop event away if any signal (s1, s2 or s2si) not present
            if s1 == None:
                self.cnt.increment_counter('n_events_not_s1')
                continue
            if s2 == None:
                self.cnt.increment_counter('n_events_not_s2')
                continue
            if s2si == None:
                self.cnt.increment_counter('n_events_not_s2si')
                continue
            # loop event away if filter fails
            if not s1s2_filter(self._s1s2_selector, s1, s2, s2si):
                self.cnt.increment_counter('n_events_more_than_1_cluster')
                continue
            # event passed selection:
            self.cnt.increment_counter('nevt_out') # increment counter
            # create Kr event & write to file
            pmapVectors = PmapVectors(s1=s1, s2=s2, s2si=s2si,
                                      events=evt_number,
                                      timestamps=evt_time)
            evt = self.create_kr_event(pmapVectors)
            write_kr(evt)

            self.conditional_print(self.cnt.counter_value('n_events_tot'),
            self.cnt.counter_value('nevt_out'))

    def get_writers(self, h5out):
        """Get the writers needed by dorothea"""
        return  kr_writer(h5out)
