from __future__ import print_function

import sys
import glob
import time
import textwrap

import numpy  as np
import tables as tb

from invisible_cities.core.configure         import configure
from invisible_cities.core.system_of_units_c import units
from invisible_cities.reco.dst_io            import HitCollection_writer
from invisible_cities.cities.base_cities     import City, HitCollectionCity, merge_two_dicts
from invisible_cities.reco.tbl_functions     import get_event_numbers_and_timestamps
from invisible_cities.reco.pmaps_functions   import load_pmaps
from invisible_cities.reco.xy_algorithms     import barycenter, find_algorithm

class Penthesilea(City, HitCollectionCity):
    def __init__(self,
                 run_number       = 0,
                 files_in         = None,
                 file_out         = None,
                 compression      = "ZLIB4",
                 nprint           = 10000,

                 drift_v          = 1 * units.mm/units.mus,

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
                  z_corr_filename = None,
                 xy_corr_filename = None,
                 lifetime         = None,
                 reco_algorithm   = barycenter):


        City             .__init__(self,
                                   run_number ,
                                   files_in   ,
                                   file_out   ,
                                   compression,
                                   nprint     )

        HitCollectionCity.__init__(self,
                                   drift_v          = drift_v,

                                   S1_Emin          = S1_Emin,
                                   S1_Emax          = S1_Emax,
                                   S1_Lmin          = S1_Lmin,
                                   S1_Lmax          = S1_Lmax,
                                   S1_Hmin          = S1_Hmin,
                                   S1_Hmax          = S1_Hmax,
                                   S1_Ethr          = S1_Ethr,

                                   S2_Nmin          = S2_Nmin,
                                   S2_Nmax          = S2_Nmax,
                                   S2_Emin          = S2_Emin,
                                   S2_Emax          = S2_Emax,
                                   S2_Lmin          = S2_Lmin,
                                   S2_Lmax          = S2_Lmax,
                                   S2_Hmin          = S2_Hmin,
                                   S2_Hmax          = S2_Hmax,
                                   S2_NSIPMmin      = S2_NSIPMmin,
                                   S2_NSIPMmax      = S2_NSIPMmax,
                                   S2_Ethr          = S2_Ethr,

                                   rebin            = rebin,
                                    z_corr_filename =  z_corr_filename,
                                   xy_corr_filename = xy_corr_filename,
                                   lifetime         = lifetime,
                                   reco_algorithm   = reco_algorithm)

    config_file_format = City.config_file_format + """

    RUN_NUMBER       {RUN_NUMBER}

    # set_print
    NPRINT           {NPRINT}

    DRIFT_V          {DRIFT_V}

    S1_EMIN          {S1_EMIN}
    S1_EMAX          {S1_EMAX}
    S1_LMIN          {S1_LMIN}
    S1_LMAX          {S1_LMAX}
    S1_HMIN          {S1_HMIN}
    S1_HMAX          {S1_HMAX}
    S1_ETHR          {S1_ETHR}

    S2_NMIN          {S2_NMIN}
    S2_NMAX          {S2_NMAX}
    S2_EMIN          {S2_EMIN}
    S2_EMAX          {S2_EMAX}
    S2_LMIN          {S2_LMIN}
    S2_LMAX          {S2_LMAX}
    S2_HMIN          {S2_HMIN}
    S2_HMAX          {S2_HMAX}
    S2_ETHR          {S2_ETHR}
    S2_NSIPMMIN      {S2_NSIPMMIN}
    S2_NSIPMMAX      {S2_NSIPMMAX}

    REBIN            {REBIN}
    Z_CORR_FILENAME  {Z_CORR_FILENAME}
    XY_CORR_FILENAME {XY_CORR_FILENAME}
    RECO_ALGORITHM   {RECO_ALGORITHM}

    # run
    NEVENTS          {NEVENTS}
    RUN_ALL          {RUN_ALL}
    """

    config_file_format = textwrap.dedent(config_file_format)

    default_config = merge_two_dicts(
        City.default_config,
        dict(RUN_NUMBER        =     0,
             NPRINT            =     1,

             DRIFT_V           =   1.0,

             S1_EMIN           =     0,
             S1_EMAX           =    30,
             S1_LMIN           =     4,
             S1_LMAX           =    20,
             S1_HMIN           =   0.5,
             S1_HMAX           =    10,
             S1_ETHR           =   0.5,

             S2_NMIN           =     1,
             S2_NMAX           =     5,
             S2_EMIN           =   1e3,
             S2_EMAX           =   1e8,
             S2_LMIN           =     1,
             S2_LMAX           =    20,
             S2_HMIN           =   500,
             S2_HMAX           =   1e5,
             S2_ETHR           =     1,
             S2_NSIPMMIN       =     2,
             S2_NSIPMMAX       =  1000,

             REBIN             = 1,
             Z_CORR_FILENAME   = "",
             XY_CORR_FILENAME  = "",
             RECO_ALGORITHM    = "barycenter",

             NEVENTS           =     3,
             RUN_ALL           = False))

    def run(self, max_evt = -1):
        nevt_in = nevt_out = 0

        with HitCollection_writer(self.output_file, "DST", "w",
                                  self.compression, "Tracks") as write:

            exit_file_loop = False
            for filename in self.input_files:
                print("Reading {}".format(filename), end="... ")

                try:
                    S1s, S2s, S2Sis = load_pmaps(filename)
                except (ValueError, tb.exceptions.NoSuchNodeError):
                    print("Empty file. Skipping.")
                    continue

                event_numbers, timestamps = get_event_numbers_and_timestamps(filename)
                for evt_number, evt_time in zip(event_numbers, timestamps):
                    nevt_in +=1

                    S1 = S1s  .get(evt_number, {})
                    S2 = S2s  .get(evt_number, {})
                    Si = S2Sis.get(evt_number, {})

                    evt = self.select_event(evt_number, evt_time, S1, S2, Si)

                    if evt:
                        nevt_out += 1
                        write(evt)

                    if not nevt_in % self.nprint:
                        print("{} evts analyzed".format(nevt_in))

                    if -1 < max_evt < nevt_in:
                        exit_file_loop = True
                        break

                print("OK")
                if exit_file_loop:
                    break


        print(textwrap.dedent("""
                              Number of events in : {}
                              Number of events out: {}
                              Ratio               : {}
                              """.format(nevt_in, nevt_out, nevt_out/nevt_in)))
        return nevt_in, nevt_out, nevt_in/nevt_out


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
                               z_corr_filename = CFP. Z_CORR_FILENAME if "Z_CORR_FILENAME" in CFP else None,
                              xy_corr_filename = CFP.XY_CORR_FILENAME,
                              lifetime         = CFP.LIFETIME if "LIFETIME" in CFP else None,
                              reco_algorithm   = find_algorithm(CFP.RECO_ALGORITHM))

    t0 = time.time()
    nevts = CFP.NEVENTS if not CFP.RUN_ALL else -1
    # run
    nevt_in, nevt_out, ratio = penthesilea.run(max_evt = nevts)
    t1 = time.time()
    dt = t1 - t0

    print("run {} evts in {} s, time/event = {}".format(nevt_in, dt, dt/nevt_in))

    return nevts, nevt_in, nevt_out

if __name__ == "__main__":
    PENTHESILEA(sys.argv)
