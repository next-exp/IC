import sys
import glob
import time
import textwrap

import numpy  as np
import tables as tb

from ..core.configure         import configure
from ..core.system_of_units_c import units
from ..io.dst_io              import hits_writer
from ..cities.base_cities     import City
from ..io.dst_io              import PersistentHitCollection
from ..io.dst_io              import Hit
from ..cities.base_cities     import HitCollectionCity
from ..reco                   import tbl_functions as tbl
from ..reco.tbl_functions     import get_event_numbers_and_timestamps_from_file_name
from ..reco.pmaps_functions   import load_pmaps
from ..reco.xy_algorithms     import barycenter
from ..reco.xy_algorithms     import find_algorithm
from ..filters.s1s2_filter    import s1s2_filter
from ..filters.s1s2_filter    import S12Selector
from ..reco.pmaps_functions   import integrate_S2Si_charge


class Penthesilea(HitCollectionCity):
    def __init__(self,
                 run_number       = 0,
                 files_in         = None,
                 file_out         = None,
                 compression      = "ZLIB4",
                 nprint           = 10000,

                 drift_v          = 1 * units.mm / units.mus,

                 S1_Emin          = 0,
                 S1_Emax          = np.inf,
                 S1_Lmin          = 0,
                 S1_Lmax          = np.inf,
                 S1_Hmin          = 0,
                 S1_Hmax          = np.inf,
                 S1_Ethr          = 0,

                 S2_Nmin          = 1,
                 S2_Nmax          = 1000,
                 S2_Emin          = 0,
                 S2_Emax          = np.inf,
                 S2_Lmin          = 0,
                 S2_Lmax          = np.inf,
                 S2_Hmin          = 0,
                 S2_Hmax          = np.inf,
                 S2_NSIPMmin      = 1,
                 S2_NSIPMmax      = np.inf,
                 S2_Ethr          = 0,

                 rebin            = 1,
                 reco_algorithm   = barycenter):

        HitCollectionCity.__init__(self,
                         run_number       = run_number,
                         files_in         = files_in,
                         file_out         = file_out,
                         compression      = compression,
                         nprint           = nprint,
                         rebin            = rebin,
                         reco_algorithm   = reco_algorithm)

        self.drift_v        = drift_v
        self._s1s2_selector = S12Selector(S1_Nmin     = 1,
                                          S1_Nmax     = 1,
                                          S1_Emin     = S1_Emin,
                                          S1_Emax     = S1_Emax,
                                          S1_Lmin     = S1_Lmin,
                                          S1_Lmax     = S1_Lmax,
                                          S1_Hmin     = S1_Hmin,
                                          S1_Hmax     = S1_Hmax,
                                          S1_Ethr     = S1_Ethr,

                                          S2_Nmin     = 1,
                                          S2_Nmax     = S2_Nmax,
                                          S2_Emin     = S2_Emin,
                                          S2_Emax     = S2_Emax,
                                          S2_Lmin     = S2_Lmin,
                                          S2_Lmax     = S2_Lmax,
                                          S2_Hmin     = S2_Hmin,
                                          S2_Hmax     = S2_Hmax,
                                          S2_NSIPMmin = S2_NSIPMmin,
                                          S2_NSIPMmax = S2_NSIPMmax,
                                          S2_Ethr     = S2_Ethr)

    def run(self, nmax):
        self.display_IO_info(nmax)

        # with HitCollection_writer(self.output_file, "DST", "w",
        #                           self.compression, "Tracks") as write:
        with tb.open_file(self.output_file, "w",
                          filters = tbl.filters(self.compression)) as h5out:
            write_hits = hits_writer(h5out)
            nevt_in, nevt_out = self._file_loop(write_hits, nmax)
        print(textwrap.dedent("""
                              Number of events in : {}
                              Number of events out: {}
                              Ratio               : {}
                              """.format(nevt_in, nevt_out, nevt_out / nevt_in)))
        return nevt_in, nevt_out

    def _file_loop(self, write_hits, nmax):
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
                event_numbers, timestamps, nmax, nevt_in, nevt_out, write_hits, S1s, S2s, S2Sis)

            if max_events_reached:
                print('Max events reached')
                break
            else:
                print("OK")

        return nevt_in, nevt_out

    def _event_loop(self, event_numbers, timestamps, nmax, nevt_in, nevt_out, write_hits, S1s, S2s, S2Sis):
        max_events_reached = False
        for evt_number, evt_time in zip(event_numbers, timestamps):
            nevt_in += 1
            if self.max_events_reached(nmax, nevt_in):
                max_events_reached = True
                break
            S1   = S1s  .get(evt_number, {})
            S2   = S2s  .get(evt_number, {})
            S2Si = S2Sis.get(evt_number, {})

            if not s1s2_filter(self._s1s2_selector, S1, S2, S2Si):
                continue
            iS2Si = integrate_S2Si_charge(S2Si)
            # All peaks must contain at least one non-zero charged sipm
            def at_least_one_sipm_with_Q_gt_0(Si):
                return any(q > 0 for q in Si.values())
            def all_peaks_contain_at_least_one_non_zero_charged_sipm(iS2Si):
               return all(at_least_one_sipm_with_Q_gt_0(Si)
                          for Si in iS2Si.values())
            if not all_peaks_contain_at_least_one_non_zero_charged_sipm(iS2Si):
                continue
            nevt_out += 1

            evt = self._create_hits_event(evt_number, evt_time, S1, S2, S2Si)
            write_hits(evt)

            self.conditional_print(evt, nevt_in)

        return nevt_in, nevt_out, max_events_reached

    def _create_hits_event(self, evt_number, evt_time, S1, S2, Si):
        hitc = PersistentHitCollection()
        hitc.evt   = evt_number
        hitc.time  = evt_time * 1e-3 # s

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
                    hit       = Hit()
                    hit.npeak = npeak
                    hit.nsipm = c.nsipm
                    hit.X     = c.pos.X
                    hit.Y     = c.pos.Y
                    hit.Z     = z
                    hit.Q     = c.Q
                    hit.E     = e
                    hitc.hits.append(hit)
            npeak += 1

        return hitc



def PENTHESILEA(argv = sys.argv):
    """Penthesilea DRIVER"""

    # get parameters dictionary
    CFP = configure(argv)

    #class instance
    penthesilea = Penthesilea(run_number       = CFP.RUN_NUMBER,
                              files_in         = sorted(glob.glob(CFP.FILE_IN)),
                              file_out         = CFP.FILE_OUT,
                              compression      = CFP.COMPRESSION,
                              nprint           = CFP.NPRINT,

                              drift_v          = CFP.DRIFT_V * units.mm/units.mus,

                              S1_Emin          = CFP.S1_EMIN * units.pes,
                              S1_Emax          = CFP.S1_EMAX * units.pes,
                              S1_Lmin          = CFP.S1_LMIN,
                              S1_Lmax          = CFP.S1_LMAX,
                              S1_Hmin          = CFP.S1_HMIN * units.pes,
                              S1_Hmax          = CFP.S1_HMAX * units.pes,
                              S1_Ethr          = CFP.S1_ETHR * units.pes,

                              S2_Nmin          = CFP.S2_NMIN,
                              S2_Nmax          = CFP.S2_NMAX,
                              S2_Emin          = CFP.S2_EMIN * units.pes,
                              S2_Emax          = CFP.S2_EMAX * units.pes,
                              S2_Lmin          = CFP.S2_LMIN,
                              S2_Lmax          = CFP.S2_LMAX,
                              S2_Hmin          = CFP.S2_HMIN * units.pes,
                              S2_Hmax          = CFP.S2_HMAX * units.pes,
                              S2_NSIPMmin      = CFP.S2_NSIPMMIN,
                              S2_NSIPMmax      = CFP.S2_NSIPMMAX,
                              S2_Ethr          = CFP.S2_ETHR * units.pes,

                              rebin            = CFP.REBIN if "REBIN" in CFP else 1,
                              reco_algorithm   = find_algorithm(CFP.RECO_ALGORITHM))

    t0 = time.time()
    nevts = CFP.NEVENTS if not CFP.RUN_ALL else -1

    nevt_in, nevt_out, ratio = penthesilea.run(nmax = nevts)
    t1 = time.time()
    dt = t1 - t0

    print("run {} evts in {} s, time/event = {}".format(nevt_in, dt, dt/nevt_in))

    return nevts, nevt_in, nevt_out

if __name__ == "__main__":
    PENTHESILEA(sys.argv)
