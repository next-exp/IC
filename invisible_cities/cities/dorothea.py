"""
code: dorothea.py
description: create a lightweight DST.
credits: see ic_authors_and_legal.rst in /doc

last revised: JJGC, July-2017
"""
import sys
import glob
import time
import textwrap

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


class Dorothea(KrCity):
    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.cnt.set_name('dorothea')
        self.cnt.set_counter('nmax', value=self.conf.nmax)
        self.cnt.init_counters(('n_events_tot', 'nevt_out'))

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
            # loop event away if filter fails
            if not s1s2_filter(self._s1s2_selector, s1, s2, s2si):
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
