import sys
import glob
import time
import textwrap

import numpy  as np
import tables as tb

from .. core.configure         import configure
from .. core.system_of_units_c import units

from .. io.kdst_io              import kr_writer
from .. reco.event_model        import PersistentKrEvent

from .. reco                   import tbl_functions   as tbl
from .. reco                   import pmaps_functions as pmp
from .. reco.pmaps_functions   import load_pmaps
from .. reco.tbl_functions     import get_event_numbers_and_timestamps_from_file_name

from .. filters.s1s2_filter    import s1s2_filter
from .. filters.s1s2_filter    import S12Selector

from .  base_cities            import City


class Dorothea(City):
    def __init__(self, **kwds):
        super().__init__(**kwds)
        conf = self.conf

        self.drift_v = conf.drift_v
        self._s1s2_selector = S12Selector(S1_Nmin     = 1,
                                          S1_Nmax     = 1,
                                          S1_Emin     = conf.s1_emin,
                                          S1_Emax     = conf.s1_emax,
                                          S1_Lmin     = conf.s1_lmin,
                                          S1_Lmax     = conf.s1_lmax,
                                          S1_Hmin     = conf.s1_hmin,
                                          S1_Hmax     = conf.s1_hmax,
                                          S1_Ethr     = conf.s1_ethr,

                                          S2_Nmin     = 1,
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
        with tb.open_file(self.output_file, "w",
                          filters = tbl.filters(self.compression)) as h5out:

            write_kr = kr_writer(h5out)

            nevt_in, nevt_out = self._file_loop(write_kr)
        print(textwrap.dedent("""
                              Number of events in : {}
                              Number of events out: {}
                              Ratio               : {}
                              """.format(nevt_in, nevt_out, nevt_out / nevt_in)))
        return nevt_in, nevt_out

    def _file_loop(self, write_kr):
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
                event_numbers, timestamps, nevt_in, nevt_out, write_kr, S1s, S2s, S2Sis)

            if max_events_reached:
                print('Max events reached')
                break
            else:
                print("OK")

        return nevt_in, nevt_out

    def _event_loop(self, event_numbers, timestamps, nevt_in, nevt_out, write_kr, S1s, S2s, S2Sis):
        max_events_reached = False
        for evt_number, evt_time in zip(event_numbers, timestamps):
            nevt_in += 1
            if self.max_events_reached(nevt_in):
                max_events_reached = True
                break
            S1 = S1s  .get(evt_number, {})
            S2 = S2s  .get(evt_number, {})
            Si = S2Sis.get(evt_number, {})

            if not s1s2_filter(self._s1s2_selector, S1, S2, Si):
                continue
            nevt_out += 1

            evt = self._create_kr_event(evt_number, evt_time, S1, S2, Si)
            write_kr(evt)

            self.conditional_print(evt, nevt_in)

        return nevt_in, nevt_out, max_events_reached

    def _create_kr_event(self, evt_number, evt_time, S1, S2, Si):
        evt       = PersistentKrEvent(evt_number, evt_time * 1e-3)
        #evt.event =
        #evt.time  = evt_time * 1e-3 # s

        evt.nS1 = len(S1)
        for peak_no, (t, e) in sorted(S1.items()):
            evt.S1w.append(pmp.width(t))
            evt.S1h.append(np.max(e))
            evt.S1e.append(np.sum(e))
            evt.S1t.append(t[np.argmax(e)])

        evt.nS2 = len(S2)
        for peak_no, (t, e) in sorted(S2.items()):
            s2time  = t[np.argmax(e)]

            evt.S2w.append(pmp.width(t, to_mus=True))
            evt.S2h.append(np.max(e))
            evt.S2e.append(np.sum(e))
            evt.S2t.append(s2time)

            IDs, Qs = pmp.integrate_sipm_charges_in_peak(Si[peak_no])
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

            dt  = s2time - evt.S1t[0] if len(evt.S1t) > 0 else -1e3
            dt *= units.ns / units.mus
            evt.DT   .append(dt)
            evt.Z    .append(dt * units.mus * self.drift_v)

        return evt
