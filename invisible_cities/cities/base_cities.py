"""
This module defines base classes for the IC cities. The classes are:
City: Handles input and output files, compression, and access to data base
DeconvolutionCity: A City that performs deconvolution of the PMT RWFs
CalibratedCity: A DeconvolutionCity that perform the calibrated sum of the
                PMTs and computes the calibrated signals in the SiPM plane.
PmapCity: A CalibratedCity that computes S1, S2 and S2Si that togehter
          constitute a PMAP.
SensorResponseCity: A city that describes sensor response

Authors: J.J. Gomez-Cadenas and J. Generowicz.
Feburary, 2017.
"""

import sys
from textwrap import dedent

import numpy as np

from .. core.configure         import print_configuration
from .. core.exceptions        import NoInputFiles
from .. core.exceptions        import NoOutputFile
from .. core.system_of_units_c import units
from .. core                   import fit_functions        as fitf

from .. database import load_db

from ..reco             import peak_functions_c as cpf
from ..reco             import peak_functions   as pf
from ..reco             import pmaps_functions  as pmp
from ..reco             import pmap_io          as pio
from ..reco             import tbl_functions    as tbf
from ..reco             import wfm_functions    as wfm
from ..reco.dst_io      import PointLikeEvent
from ..reco.nh5         import DECONV_PARAM
from ..reco.corrections import Correction
from ..reco.corrections import Fcorrection

from ..sierpe import blr
from ..sierpe import fee as FE

if sys.version_info >= (3,5):
    # Exec to avoid syntax errors in older Pythons
    exec("""def merge_two_dicts(a,b):
               return {**a, **b}""")
else:
    def merge_two_dicts(a,b):
        c = a.copy()
        c.update(b)
        return c


class City:
    """Base class for all cities.
       An IC city consumes data stored in the input_files and produce new data
       which is stored in the output_file. In addition to setting input and
       output files, the base class sets the print frequency and accesses
       the data base, storing as attributed several calibration coefficients

     """

    def __init__(self,
                 run_number  = 0,
                 files_in    = None,
                 file_out    = None,
                 compression = 'ZLIB4',
                 nprint      = 10000):

        self.run_number     = run_number
        self.nprint         = nprint  # default print frequency
        self.input_files    = files_in
        self.output_file    = file_out
        self.compression    = compression
        # access data base
        DataPMT             = load_db.DataPMT (run_number)
        DataSiPM            = load_db.DataSiPM(run_number)
        self.det_geo        = load_db.DetectorGeo()

        # This is JCK-1: text reveals symmetry!
        self.xs              = DataSiPM.X.values
        self.ys              = DataSiPM.Y.values
        self.pmt_active      = np.nonzero(DataPMT.Active.values)[0].tolist()
        self.adc_to_pes      = abs(DataPMT.adc_to_pes.values).astype(np.double)
        self.sipm_adc_to_pes = DataSiPM.adc_to_pes.values    .astype(np.double)
        self.coeff_c         = DataPMT.coeff_c.values        .astype(np.double)
        self.coeff_blr       = DataPMT.coeff_blr.values      .astype(np.double)
        self.noise_rms       = DataPMT.noise_rms.values      .astype(np.double)

    @property
    def monte_carlo(self):
        return self.run_number <= 0

    def check_files(self):
        if not self.input_files:
            raise NoInputFiles('input file list is empty, must set before running')
        if not self.output_file:
            raise NoOutputFile('must set output file before running')

    def conditional_print(self, evt, n_events_tot):
        if n_events_tot % self.nprint == 0:
            print('event in file = {}, total = {}'
                  .format(evt, n_events_tot))

    def max_events_reached(self, nmax, n_events_tot):
        if n_events_tot >= nmax > -1:
            print('reached max nof of events (= {})'
                  .format(nmax))
            return True
        return False


    def display_IO_info(self, nmax):
        print("""
                 {} will run a max of {} events
                 Input Files = {}
                 Output File = {}
                          """.format(self.__class__.__name__,
                                     nmax, self.input_files, self.output_file))

    def print_configuration(self, sp):
        print_configuration({"# PMT"        : sp.NPMT,
                             "PMT WL"       : sp.PMTWL,
                             "# SiPM"       : sp.NSIPM,
                             "SIPM WL"      : sp.SIPMWL})

    def _get_rwf(self, h5in):
        return (h5in.root.RD.pmtrwf,
                h5in.root.RD.sipmrwf)

    def _get_run_info(self, h5in):
        return h5in.root.Run.events


class SensorResponseCity(City):
    """A SensorResponseCity city extends the City base class adding the
       response (Monte Carlo simulation) of the energy plane and
       tracking plane sensors (PMTs and SiPMs).
    """

    def __init__(self,
                 run_number  = 0,
                 files_in    = None,
                 file_out    = None,
                 compression = 'ZLIB4',
                 nprint      = 10000,
                 # Parameters added at this level
                 sipm_noise_cut = 3 * units.pes,
                 first_evt = 0):

        City.__init__(self,
                      run_number  = run_number,
                      files_in    = files_in,
                      file_out    = file_out,
                      compression = compression,
                      nprint      = nprint)

        self.sipm_noise_cut = sipm_noise_cut
        self.first_evt      = first_evt

    def simulate_sipm_response(self, event, sipmrd,
                               sipms_noise_sampler):
        """Add noise with the NoiseSampler class and return
        the noisy waveform (in pes)."""
        # add noise (in PES) to true waveform
        dataSiPM = sipmrd[event] + sipms_noise_sampler.Sample()
        # return total signal in adc counts
        return wfm.to_adc(dataSiPM, self.sipm_adc_to_pes)

    def simulate_pmt_response(self, event, pmtrd):
        """ Full simulation of the energy plane response
        Input:
         1) extensible array pmtrd
         2) event_number

        returns:
        array of raw waveforms (RWF) obtained by convoluting pmtrd with the PMT
        front end electronics (LPF, HPF filters)
        array of BLR waveforms (only decimation)
        """
        # Single Photoelectron class
        spe = FE.SPE()
        # FEE, with noise PMT
        fee  = FE.FEE(noise_FEEPMB_rms=FE.NOISE_I, noise_DAQ_rms=FE.NOISE_DAQ)
        NPMT = pmtrd.shape[1]
        RWF  = []
        BLRX = []

        for pmt in range(NPMT):
            # normalize calibration constants from DB to MC value
            cc = self.adc_to_pes[pmt] / FE.ADC_TO_PES
            # signal_i in current units
            signal_i = FE.spe_pulse_from_vector(spe, pmtrd[event, pmt])
            # Decimate (DAQ decimation)
            signal_d = FE.daq_decimator(FE.f_mc, FE.f_sample, signal_i)
            # Effect of FEE and transform to adc counts
            signal_fee = FE.signal_v_fee(fee, signal_d, pmt) * FE.v_to_adc()
            # add noise daq
            signal_daq = cc * FE.noise_adc(fee, signal_fee)
            # signal blr is just pure MC decimated by adc in adc counts
            signal_blr = cc * FE.signal_v_lpf(fee, signal_d) * FE.v_to_adc()
            # raw waveform stored with negative sign and offset
            RWF.append(FE.OFFSET - signal_daq)
            # blr waveform stored with positive sign and no offset
            BLRX.append(signal_blr)
        return np.array(RWF), np.array(BLRX)


    def store_FEE_table(self):
        """Store the parameters of the EP FEE simulation."""
        row = self.fee_table.row
        row["OFFSET"]        = FE.OFFSET
        row["CEILING"]       = FE.CEILING
        row["PMT_GAIN"]      = FE.PMT_GAIN
        row["FEE_GAIN"]      = FE.FEE_GAIN
        row["R1"]            = FE.R1
        row["C1"]            = FE.C1
        row["C2"]            = FE.C2
        row["ZIN"]           = FE.Zin
        row["DAQ_GAIN"]      = FE.DAQ_GAIN
        row["NBITS"]         = FE.NBITS
        row["LSB"]           = FE.LSB
        row["NOISE_I"]       = FE.NOISE_I
        row["NOISE_DAQ"]     = FE.NOISE_DAQ
        row["t_sample"]      = FE.t_sample
        row["f_sample"]      = FE.f_sample
        row["f_mc"]          = FE.f_mc
        row["f_LPF1"]        = FE.f_LPF1
        row["f_LPF2"]        = FE.f_LPF2
        row["coeff_c"]       = self.coeff_c
        row["coeff_blr"]     = self.coeff_blr
        row["adc_to_pes"]    = self.adc_to_pes
        row["pmt_noise_rms"] = self.noise_rms
        row.append()
        self.fee_table.flush()

    @property
    def FE_t_sample(self):
        return FE.t_sample


class DeconvolutionCity(City):
    """A Deconvolution city extends the City base class adding the
       deconvolution step, which transforms RWF into CWF.
       The parameters of the deconvolution are the number of samples
       used to compute the baseline (n_baseline) and the threshold to
       thr_trigger in the rising signal (thr_trigger)
    """

    def __init__(self,
                 run_number            = 0,
                 files_in              = None,
                 file_out              = None,
                 compression           = 'ZLIB4',
                 nprint                = 10000,
                 # Parameters added at this level
                 n_baseline            = 28000,
                 thr_trigger           = 5 * units.adc,
                 acum_discharge_length = 5000):

        City.__init__(self,
                      run_number  = run_number,
                      files_in    = files_in,
                      file_out    = file_out,
                      compression = compression,
                      nprint      = nprint)

        # BLR parameters
        self.n_baseline            = n_baseline
        self.thr_trigger           = thr_trigger
        self.acum_discharge_length = acum_discharge_length

    def write_deconv_params(self, ofile):
        group = ofile.create_group(ofile.root, "DeconvParams")

        table = ofile.create_table(group,
                                   "DeconvParams",
                                   DECONV_PARAM,
                                   "deconvolution parameters",
                                   tbf.filters(self.compression))

        row = table.row
        row["N_BASELINE"]            = self.n_baseline
        row["THR_TRIGGER"]           = self.thr_trigger
        row["ACUM_DISCHARGE_LENGTH"] = self.acum_discharge_length
        table.flush()

    def deconv_pmt(self, RWF):
        """Deconvolve the RWF of the PMTs"""
        return blr.deconv_pmt(RWF,
                              self.coeff_c,
                              self.coeff_blr,
                              pmt_active            = self.pmt_active,
                              n_baseline            = self.n_baseline,
                              thr_trigger           = self.thr_trigger,
                              acum_discharge_length = self.acum_discharge_length)


class CalibratedCity(DeconvolutionCity):
    """A calibrated city extends a DeconvCity, performing two actions.
       1. Compute the calibrated sum of PMTs, in two flavours:
          a) csum: PMTs waveforms are equalized to photoelectrons (pes) and
             added
          b) csum_mau: waveforms are equalized to photoelectrons;
             compute a MAU that follows baseline and add PMT samples above
             MAU + threshold
       2. Compute the calibrated signal in the SiPMs:
          a) equalize to pes;
          b) compute a MAU that follows baseline and keep samples above
             MAU + threshold.
       """

    def __init__(self,
                 run_number            = 0,
                 files_in              = None,
                 file_out              = None,
                 compression           = 'ZLIB4',
                 nprint                = 10000,
                 n_baseline            = 28000,
                 thr_trigger           = 5 * units.adc,
                 acum_discharge_length = 5000,
                 # Parameters added at this level
                 n_MAU                 = 100,
                 thr_MAU               = 3.0 * units.adc,
                 thr_csum_s1           = 0.2 * units.pes,
                 thr_csum_s2           = 1.0 * units.pes,
                 n_MAU_sipm            = 100,
                   thr_sipm            = 5.0 * units.pes):

        DeconvolutionCity.__init__(self,
                                   run_number            = run_number,
                                   files_in              = files_in,
                                   file_out              = file_out,
                                   compression           = compression,
                                   nprint                = nprint,
                                   n_baseline            = n_baseline,
                                   thr_trigger           = thr_trigger,
                                   acum_discharge_length = acum_discharge_length)

        # Parameters of the PMT csum.
        self.n_MAU       = n_MAU
        self.thr_MAU     = thr_MAU
        self.thr_csum_s1 = thr_csum_s1
        self.thr_csum_s2 = thr_csum_s2

        # Parameters of the SiPM signal
        self.n_MAU_sipm = n_MAU_sipm
        self.  thr_sipm =   thr_sipm

    def calibrated_pmt_sum(self, CWF):
        """Return the csum and csum_mau calibrated sums."""
        return cpf.calibrated_pmt_sum(CWF,
                                      self.adc_to_pes,
                                      pmt_active = self.pmt_active,
                                           n_MAU = self.  n_MAU   ,
                                         thr_MAU = self.thr_MAU   )

    def csum_zs(self, csum, threshold):
        """Zero Suppression over csum"""
        return cpf.wfzs(csum, threshold=threshold)

    def calibrated_signal_sipm(self, SiRWF):
        """Return the calibrated signal in the SiPMs."""
        return cpf.signal_sipm(SiRWF,
                               self.sipm_adc_to_pes,
                               thr   = self.  thr_sipm,
                               n_MAU = self.n_MAU_sipm)


class PmapCity(CalibratedCity):
    """A PMAP city extends a CalibratedCity, computing the S1, S2 and S2Si
       objects that togehter constitute a PMAP.

    """

    def __init__(self,
                 run_number            = 0,
                 files_in              = None,
                 file_out              = None,
                 compression           = 'ZLIB4',
                 nprint                = 10000,
                 n_baseline            = 28000,
                 thr_trigger           = 5 * units.adc,
                 acum_discharge_length = 5000,
                 n_MAU                 = 100,
                 thr_MAU               = 3.0 * units.adc,
                 thr_csum_s1           = 0.2 * units.adc,
                 thr_csum_s2           = 1.0 * units.adc,
                 n_MAU_sipm            = 100,
                 thr_sipm              = 5.0 * units.pes,
                 # Parameters added at this level
                 s1_params             = None,
                 s2_params             = None,
                 thr_sipm_s2           = 30 * units.pes):

        CalibratedCity.__init__(self,
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
                                  thr_sipm            =   thr_sipm)

        self.s1_params   = s1_params
        self.s2_params   = s2_params
        self.thr_sipm_s2 = thr_sipm_s2

    def find_S12(self, s1_ene, s1_indx, s2_ene, s2_indx):
        """Return S1 and S2."""
        S1 = cpf.find_S12(s1_ene,
                          s1_indx,
                          **self.s1_params._asdict())

        S2 = cpf.find_S12(s2_ene,
                          s2_indx,
                          **self.s2_params._asdict())
        return S1, S2

    def correct_S1_ene(self, S1, csum):
        return cpf.correct_S1_ene(S1, csum)

    def find_S2Si(self, S2, sipmzs):
        """Return S2Si."""
        SIPM = cpf.select_sipm(sipmzs)
        S2Si = pf.sipm_s2_dict(SIPM, S2, thr = self.thr_sipm_s2)
        return pio.S2Si(S2Si)


class S12SelectorCity:
    def __init__(self,
                 drift_v     = 1 * units.mm/units.mus,

                 S1_Nmin     = 0,
                 S1_Nmax     = 1000,
                 S1_Emin     = 0,
                 S1_Emax     = np.inf,
                 S1_Lmin     = 0,
                 S1_Lmax     = np.inf,
                 S1_Hmin     = 0,
                 S1_Hmax     = np.inf,
                 S1_Ethr     = 0,

                 S2_Nmin     = 0,
                 S2_Nmax     = 1000,
                 S2_Emin     = 0,
                 S2_Emax     = np.inf,
                 S2_Lmin     = 0,
                 S2_Lmax     = np.inf,
                 S2_Hmin     = 0,
                 S2_Hmax     = np.inf,
                 S2_NSIPMmin = 1,
                 S2_NSIPMmax = np.inf,
                 S2_Ethr     = 0):

        self.drift_v     = drift_v

        self.S1_Nmin     = S1_Nmin
        self.S1_Nmax     = S1_Nmax
        self.S1_Emin     = S1_Emin
        self.S1_Emax     = S1_Emax
        self.S1_Lmin     = S1_Lmin
        self.S1_Lmax     = S1_Lmax
        self.S1_Hmin     = S1_Hmin
        self.S1_Hmax     = S1_Hmax
        self.S1_Ethr     = S1_Ethr

        self.S2_Nmin     = S2_Nmin
        self.S2_Nmax     = S2_Nmax
        self.S2_Emin     = S2_Emin
        self.S2_Emax     = S2_Emax
        self.S2_Lmin     = S2_Lmin
        self.S2_Lmax     = S2_Lmax
        self.S2_Hmin     = S2_Hmin
        self.S2_Hmax     = S2_Hmax
        self.S2_NSIPMmin = S2_NSIPMmin
        self.S2_NSIPMmax = S2_NSIPMmax
        self.S2_Ethr     = S2_Ethr

    def select_S1(self, s1s):
        return pf.select_peaks(s1s,
                               self.S1_Emin, self.S1_Emax,
                               self.S1_Lmin, self.S1_Lmax,
                               self.S1_Hmin, self.S1_Hmax,
                               self.S1_Ethr)

    def select_S2(self, s2s, sis):
        s2s = pf.select_peaks(s2s,
                              self.S2_Emin, self.S2_Emax,
                              self.S2_Lmin, self.S2_Lmax,
                              self.S2_Hmin, self.S2_Hmax,
                              self.S2_Ethr)
        sis = pf.select_Si(sis,
                           self.S2_NSIPMmin, self.S2_NSIPMmax)

        valid_peaks = set(s2s) & set(sis)
        s2s = {peak_no: peak for peak_no, peak in s2s.items() if peak_no in valid_peaks}
        sis = {peak_no: peak for peak_no, peak in sis.items() if peak_no in valid_peaks}
        return s2s, sis

    def select_event(self, evt_number, evt_time, S1, S2, Si):
        evt       = PointLikeEvent()
        evt.event = evt_number
        evt.time  = evt_time * 1e-3 # s

        S1     = self.select_S1(S1)
        S2, Si = self.select_S2(S2, Si)

        if (not self.S1_Nmin <= len(S1) <= self.S1_Nmax or
            not self.S2_Nmin <= len(S2) <= self.S2_Nmax):
            return None

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

            IDs, Qs = pmp.integrate_charge(Si[peak_no])
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
            dt *= units.ns  / units.mus
            evt.DT   .append(dt)
            evt.Z    .append(dt * units.mus * self.drift_v)

        return evt


class MapCity(City):
    def __init__(self,
                 lifetime           ,

                 xbins        =  100,
                 xmin         = None,
                 xmax         = None,

                 ybins        =  100,
                 ymin         = None,
                 ymax         = None):

        self._lifetimes = [lifetime] if not np.shape(lifetime) else lifetime
        self._lifetime_corrections = tuple(map(self._create_fcorrection, self._lifetimes))

        xmin = self.det_geo.XMIN[0] if xmin is None else xmin
        xmax = self.det_geo.XMAX[0] if xmax is None else xmax
        ymin = self.det_geo.YMIN[0] if ymin is None else ymin
        ymax = self.det_geo.YMAX[0] if ymax is None else ymax

        self._xbins  = xbins
        self._ybins  = ybins
        self._xrange = xmin, xmax
        self._yrange = ymin, ymax

    def xy_correction(self, X, Y, E):
        xs, ys, es, us = \
        fitf.profileXY(X, Y, E, self._xbins, self._ybins, self._xrange, self._yrange)

        norm_index = xs.size//2, ys.size//2
        return Correction((xs, ys), es, us, norm_strategy="index", index=norm_index)

    def xy_statistics(self, X, Y):
        return np.histogram2d(X, Y, (self._xbins, self._ybins), (self._xrange, self._yrange))

    def _create_fcorrection(self, LT):
        return Fcorrection(lambda x, lt:             fitf.expo(x, 1, -lt),
                           lambda x, lt: x / LT**2 * fitf.expo(x, 1, -lt),
                           (LT,))
