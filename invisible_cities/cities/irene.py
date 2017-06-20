import sys

from glob     import glob
from time     import time
from argparse import Namespace

import numpy  as np
import tables as tb

from .. core.configure         import configure
from .. core.system_of_units_c import units

from .. io.mc_io              import mc_track_writer
from .. io.pmap_io            import pmap_writer
from .. io.run_and_event_io   import run_and_event_writer
from .. reco                  import tbl_functions as tbl
from .. reco.params           import S12Params as S12P
from .. core.ic_types         import minmax

from .  base_cities  import PmapCity


class Irene(PmapCity):
    """Perform fast processing from raw data to pmaps.

    Raw data pmtrwf and sipmrwf.
    It is optimized for speed (use of CYTHON functions) and intended
    for fast processing of data.
    """
    def __init__(self,
                 run_number            = 0,
                 nprint                = 10000,
                 files_in              = None,
                 file_out              = None,
                 compression           = 'ZLIB4',
                 n_baseline            = 28000,
                 thr_trigger           =   5.0 * units.adc,
                 n_MAU                 = 100,
                 thr_MAU               =   3.0 * units.adc,
                 thr_csum_s1           =   0.2 * units.pes,
                 thr_csum_s2           =   1.0 * units.pes,
                 n_MAU_sipm            = 100   * units.adc,
                 thr_sipm              =   5.0 * units.pes,
                 s1_params             = None,
                 s2_params             = None,
                 thr_sipm_s2           =  30 * units.pes,
                 # Almost a hard-wired parameter. Not settable via
                 # driver or config file.
                 acum_discharge_length = 5000):

        PmapCity.__init__(self,
                          run_number            = run_number,
                          files_in              = files_in,
                          file_out              = file_out,
                          compression           = compression,
                          nprint                = nprint,
                          n_baseline            = n_baseline,
                          thr_trigger           = thr_trigger,
                          acum_discharge_length = acum_discharge_length,
                          n_MAU                 = n_MAU,
                          thr_MAU               = thr_MAU,
                          thr_csum_s1           = thr_csum_s1,
                          thr_csum_s2           = thr_csum_s2,
                          n_MAU_sipm            = n_MAU_sipm,
                          thr_sipm              = thr_sipm,
                          s1_params             = s1_params,
                          s2_params             = s2_params,
                          thr_sipm_s2           = thr_sipm_s2)

        self.check_files()
        self.check_s1s2_params()


    def run(self, nmax, print_empty=True):
        self.display_IO_info(nmax)
        sensor_params = self.get_sensor_params(self.input_files[0])
        print(sensor_params)
        
        with tb.open_file(self.output_file, "w",
                          filters = tbl.filters(self.compression)) as h5out:
            writers = Namespace(
                pmap          =          pmap_writer(h5out),
                run_and_event = run_and_event_writer(h5out),
                mc            =      mc_track_writer(h5out) if self.monte_carlo else None,
            )
            self.write_deconv_params(h5out)
            n_events_tot, n_empty_events = self._file_loop(writers, nmax)
        if print_empty:
            print('Energy plane empty events (skipped) = {}'.format(
                   n_empty_events))
        return n_events_tot, n_empty_events

    def _file_loop(self, writers, nmax):
        n_events_tot, n_empty_events = 0, 0
        for filename in self.input_files:
            print("Opening", filename, end="... ")
            with tb.open_file(filename, "r") as h5in:
                # access RWF
                NEVT, pmtrwf, sipmrwf, _ = self.get_rwf_vectors(h5in)
                events_info = self.get_run_and_event_info(h5in)
                # loop over all events in file unless reach nmax
                (n_events_tot,
                 n_empty_events) = self._event_loop(NEVT, pmtrwf, sipmrwf, events_info,
                                                    writers,
                                                    nmax, n_events_tot, n_empty_events, h5in)
        return n_events_tot, n_empty_events

    def _event_loop(self, NEVT, pmtrwf, sipmrwf, events_info,
                    write,
                    nmax, n_events_tot, n_empty_events, h5in):
        for evt in range(NEVT):
            if self.monte_carlo:
                write.mc(h5in.root.MC.MCTracks, n_events_tot)

            s1_ene, s1_indx, s2_ene, s2_indx, csum = self.pmt_transformation(pmtrwf[evt])

            # In a few rare cases s2_ene is empty
            # this is due to empty energy plane events
            # a protection is set to avoid a crash
            if np.sum(s2_ene) == 0:
                n_empty_events += 1
                continue

            sipmzs = self.calibrated_signal_sipm(sipmrwf[evt])
            S1, S2, Si = self.pmaps(s1_ene, s1_indx, s2_ene, s2_indx, csum, sipmzs)

            event, timestamp = self.event_and_timestamp(evt, events_info)
            # write to file
            write.pmap         (event, S1, S2, Si)
            write.run_and_event(self.run_number, event, timestamp)
            n_events_tot += 1
            self.conditional_print(evt, n_events_tot)
            if self.max_events_reached(nmax, n_events_tot):
                break
        return n_events_tot, n_empty_events

    def pmt_transformation(self, RWF):
            # deconvolve
            CWF = self.deconv_pmt(RWF)
            # calibrated PMT sum
            csum, csum_mau = self.calibrated_pmt_sum(CWF)
            #ZS sum for S1 and S2
            s1_ene, s1_indx = self.csum_zs(csum_mau, threshold =
                                           self.thr_csum_s1)
            s2_ene, s2_indx = self.csum_zs(csum,     threshold =
                                           self.thr_csum_s2)
            return s1_ene, s1_indx, s2_ene, s2_indx, csum

    def pmaps(self, s1_ene, s1_indx, s2_ene, s2_indx, csum, sipmzs):
        S1, S2 = self.find_S12(s1_ene, s1_indx,   s2_ene, s2_indx)
        S1     = self.correct_S1_ene(S1, csum)
        Si     = self.find_S2Si(S2, sipmzs)
        return S1, S2, Si

    def display_IO_info(self, nmax):
        PmapCity.display_IO_info(self, nmax)
        print("""
                 S1 parameters {}""" .format(self.s1_params))
        print("""
                 S2 parameters {}""" .format(self.s2_params))
        print("""
                 S2Si parameters
                 threshold min charge per SiPM = {s.thr_sipm} pes
                 threshold min charge in  S2   = {s.thr_sipm_s2} pes
                          """.format(s=self))


def IRENE(argv = sys.argv):
    """IRENE DRIVER"""

    # get parameters dictionary
    CFP = configure(argv)

    # parameters for s1 searches
    s1par = S12P(time = minmax(min   = CFP.S1_TMIN * units.mus,
                               max   = CFP.S1_TMAX * units.mus),
                 stride              = CFP.S1_STRIDE,
                 length = minmax(min = CFP.S1_LMIN,
                                 max = CFP.S1_LMAX),
                 rebin               = False)

    # parameters for s2 searches
    s2par = S12P(time = minmax(min   = CFP.S2_TMIN * units.mus,
                               max   = CFP.S2_TMAX * units.mus),
                 stride              = CFP.S2_STRIDE,
                 length = minmax(min = CFP.S2_LMIN,
                                 max = CFP.S2_LMAX),
                 rebin               = True)

    # input files
    # TODO detect non existing files and raise sensible message
    files_in = glob(CFP.FILE_IN)
    files_in.sort()
    irene = Irene(run_number  = CFP.RUN_NUMBER,
                  nprint      = CFP.NPRINT,
                  files_in    = files_in,
                  file_out    = CFP.FILE_OUT,
                  compression = CFP.COMPRESSION,
                  n_baseline  = CFP.NBASELINE,
                  thr_trigger = CFP.THR_TRIGGER * units.adc,
                  n_MAU       = CFP.NMAU,
                  thr_MAU     = CFP.THR_MAU     * units.adc,
                  thr_csum_s1 = CFP.THR_CSUM_S1 * units.pes,
                  thr_csum_s2 = CFP.THR_CSUM_S2 * units.pes,
                  n_MAU_sipm  = CFP.NMAU_SIPM   * units.adc,
                  thr_sipm    = CFP.THR_SIPM    * units.pes,
                  s1_params   = s1par,
                  s2_params   = s2par,
                  thr_sipm_s2 = CFP.THR_SIPM_S2)

    t0 = time()
    nevts = CFP.NEVENTS if not CFP.RUN_ALL else -1
    # run
    nevt, n_empty_events = irene.run(nmax=nevts, print_empty=CFP.PRINT_EMPTY_EVENTS)
    t1 = time()
    dt = t1 - t0

    if nevt > 0:
        print("run {} evts in {} s, time/event = {}".format(nevt, dt, dt/nevt))
