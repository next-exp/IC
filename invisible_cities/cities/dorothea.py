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
from .. evm.event_model         import PersistentKrEvent

from .. reco                   import tbl_functions   as tbl
from .. reco                   import pmaps_functions_c as pmp

from .. filters.s1s2_filter    import s1s2_filter
from .. filters.s1s2_filter    import S12Selector

from .  base_cities            import HitCity


class Dorothea(HitCity):
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
            # event passed selection: increment counter and write
            self.cnt.increment_counter('nevt_out')
            evt = self._create_kr_event(evt_number, evt_time, s1, s2, s2si)
            write_kr(evt)

            self.conditional_print(self.cnt.counter_value('n_events_tot'),
            self.cnt.counter_value('nevt_out'))


    @staticmethod
    def print_stats(nevt_in, nevt_out):
        print(textwrap.dedent("""
                              Number of events in : {}
                              Number of events out: {}
                              Ratio               : {}
                              """.format(nevt_in, nevt_out, nevt_out / nevt_in)))

    def get_writers(self, h5out):
        """Get the writers needed by dorothea"""
        return  kr_writer(h5out)


    def _create_kr_event(self, evt_number, evt_time, s1, s2, s2si):
        evt       = PersistentKrEvent(evt_number, evt_time * 1e-3)

        evt.nS1 = s1.number_of_peaks
        for peak_no in s1.peak_collection():
            peak = s1.peak_waveform(peak_no)
            evt.S1w.append(peak.width)
            evt.S1h.append(peak.height)
            evt.S1e.append(peak.total_energy)
            evt.S1t.append(peak.tpeak)

        evt.nS2 = s2.number_of_peaks
        for peak_no in s2.peak_collection():
            peak = s2.peak_waveform(peak_no)
            evt.S2w.append(peak.width/units.mus)
            evt.S2h.append(peak.height)
            evt.S2e.append(peak.total_energy)
            evt.S2t.append(peak.tpeak)

            IDs, Qs = pmp.integrate_sipm_charges_in_peak(s2si, peak_no)
            xsipms  = self.xs[IDs]
            ysipms  = self.ys[IDs]
            x       = np.average(xsipms, weights=Qs)
            y       = np.average(ysipms, weights=Qs)
            q       = np.sum    (Qs)

            evt.Nsipm.append(len(IDs))
            evt.S2q  .append(q)

            evt.X    .append(x)
            evt.Y    .append(y)

            evt.Xrms .append((np.sum(Qs * (xsipms-x)**2) / (q - 1))**0.5)
            evt.Yrms .append((np.sum(Qs * (ysipms-y)**2) / (q - 1))**0.5)

            evt.R    .append((x**2 + y**2)**0.5)
            evt.Phi  .append(np.arctan2(y, x))

            dt  = evt.S2t[peak_no] - evt.S1t[0]
            dt  *= units.ns / units.mus
            evt.DT   .append(dt)
            evt.Z    .append(dt * units.mus * self.drift_v)

        return evt
