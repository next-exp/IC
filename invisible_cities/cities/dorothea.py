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
from .. reco                   import pmaps_functions_c as pmp

from .. filters.s1s2_filter    import s1s2_filter
from .. filters.s1s2_filter    import S12Selector

from .  base_cities            import City


class Dorothea(City):
    def __init__(self, **kwds):
        super().__init__(**kwds)
        conf = self.conf

        self.drift_v = conf.drift_v
        self._s1s2_selector = S12Selector(**kwds)

    def run(self):
        self.display_IO_info()
        with tb.open_file(self.output_file, "w",
                          filters = tbl.filters(self.compression)) as h5out:

            write_kr = kr_writer(h5out)

            nevt_in, nevt_out = self._file_loop(write_kr)
            self.print_stats(nevt_in, nevt_out)

        return nevt_in, nevt_out

    def _file_loop(self, write_kr):
        nevt_in = nevt_out = 0

        for filename in self.input_files:
            print("Opening {filename}".format(**locals()), end="... ")

            try:
                s1_dict, s2_dict, s2si_dict = self.get_pmaps_dicts(filename)

            except (ValueError, tb.exceptions.NoSuchNodeError):
                print("Empty file. Skipping.")
                continue

            event_numbers, timestamps = self.event_numbers_and_timestamps_from_file_name(filename)

            nevt_in, nevt_out, max_events_reached = self._event_loop(
                event_numbers, timestamps, nevt_in, nevt_out, write_kr, s1_dict, s2_dict, s2si_dict)

            if max_events_reached:
                print('Max events reached')
                break
            else:
                print("OK")

        return nevt_in, nevt_out

    def _event_loop(self, event_numbers, timestamps, nevt_in, nevt_out, write_kr, s1_dict, s2_dict, s2si_dict):
        max_events_reached = False
        for evt_number, evt_time in zip(event_numbers, timestamps):
            nevt_in += 1
            if self.max_events_reached(nevt_in):
                max_events_reached = True
                break
            s1, s2, s2si = self. get_pmaps_from_dicts(s1_dict,
                                                      s2_dict,
                                                      s2si_dict,
                                                      evt_number)
            # loop event away if any signal (s1, s2 or s2si) not present
            if s1 == None or s2 == None or s2si == None:
                continue
            # loop event away if filter fails
            if not s1s2_filter(self._s1s2_selector, s1, s2, s2si):
                continue
            nevt_out += 1

            evt = self._create_kr_event(evt_number, evt_time, s1, s2, s2si)
            write_kr(evt)

            self.conditional_print(evt, nevt_in)

        return nevt_in, nevt_out, max_events_reached


    @staticmethod
    def print_stats(nevt_in, nevt_out):
        print(textwrap.dedent("""
                              Number of events in : {}
                              Number of events out: {}
                              Ratio               : {}
                              """.format(nevt_in, nevt_out, nevt_out / nevt_in)))

    def _create_kr_event(self, evt_number, evt_time, s1, s2, s2si):
        evt       = PersistentKrEvent(evt_number, evt_time * 1e-3)
        #evt.event =
        #evt.time  = evt_time * 1e-3 # s

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

            # IDs, Qs = pmp.integrate_sipm_charges_in_peak(si.s2sid[peak_no])

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

