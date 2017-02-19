"""
This module defines base classes for the IC cities. The classes are:
City: Handles input and output files, compression, and access to data base
DeconvolutionCity: A City that performs deconvolution of the PMT RWFs
CalibratedCity: A DeconvolutionCity that perform the calibrated sum of the
                PMTs and computes the calibrated signals in the SiPM plane.
PmapCity: A CalibratedCity that computes S1, S2 and S2Si that togehter
          constitute a PMAP.

Authors: J.J. Gomez-Cadenas and J. Generowicz.
Feburary, 2017.
"""

from collections import namedtuple
import numpy as np

from   invisible_cities.database import load_db
from   invisible_cities.core.system_of_units_c import SystemOfUnits
import invisible_cities.sierpe.blr as blr
import invisible_cities.core.peak_functions_c as cpf
import invisible_cities.core.pmaps_functions as pmp
from   invisible_cities.core.exceptions import NoInputFiles, NoOutputFile
import invisible_cities.sierpe.fee as FE
import invisible_cities.core.wfm_functions as wfm
from   invisible_cities.core.random_sampling \
         import NoiseSampler as SiPMsNoiseSampler
from   invisible_cities.core.configure import print_configuration

units = SystemOfUnits()

S12Params   = namedtuple('S12Params', 'tmin tmax stride lmin lmax rebin')
SensorParam = namedtuple('SensorParam', 'NPMT PMTWL NSIPM SIPMWL')


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

        # This is JCK-1: text reveals symmetry!
        self.xs              = DataSiPM.X.values
        self.ys              = DataSiPM.Y.values
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

    def set_print(self, nprint=1000):
        """Print frequency."""
        self.nprint = nprint

    def set_input_files(self, input_files):
        """Set the input files."""
        self.input_files = input_files

    def set_output_file(self, output_file):
        """Set the input files."""
        self.output_file = output_file

    def set_compression(self, compression):
        """Set the input files."""
        self.compression = compression

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
                 sipm_noise_cut = 3 * units.pes):

        City.__init__(self,
                      run_number  = run_number,
                      files_in    = files_in,
                      file_out    = file_out,
                      compression = compression,
                      nprint      = nprint)

        self.sipm_noise_cut = sipm_noise_cut

    def set_sipm_noise_cut(self, noise_cut=3.0):
        """Sets the SiPM noise cut (in PES)"""
        self.sipm_noise_cut = noise_cut

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
                 run_number  = 0,
                 files_in    = None,
                 file_out    = None,
                 compression = 'ZLIB4',
                 nprint      = 10000,
                 # Parameters added at this level
                 n_baseline  = 28000,
                 thr_trigger = 5 * units.adc):

        City.__init__(self,
                      run_number  = run_number,
                      files_in    = files_in,
                      file_out    = file_out,
                      compression = compression,
                      nprint      = nprint)
        # BLR parameters
        self.n_baseline  = n_baseline
        self.thr_trigger = thr_trigger

    def set_blr(self, n_baseline, thr_trigger):
        """Set the parameters of the Base Line Restoration (BLR)"""
        self.n_baseline  = n_baseline
        self.thr_trigger = thr_trigger

    def deconv_pmt(self, RWF):
        """Deconvolve the RWF of the PMTs"""
        return blr.deconv_pmt(RWF,
                              self.coeff_c,
                              self.coeff_blr,
                              n_baseline  = self.n_baseline,
                              thr_trigger = self.thr_trigger)


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
                 run_number  = 0,
                 files_in    = None,
                 file_out    = None,
                 compression = 'ZLIB4',
                 nprint      = 10000,
                 n_baseline  = 28000,
                 thr_trigger = 5 * units.adc,
                 # Parameters added at this level
                 n_MAU       = 100,
                 thr_MAU     = 3.0 * units.adc,
                 thr_csum_s1 = 0.2 * units.pes,
                 thr_csum_s2 = 1.0 * units.pes,
                 n_MAU_sipm  = 100,
                   thr_sipm  = 5.0 * units.pes):

        DeconvolutionCity.__init__(self,
                                   run_number  = run_number,
                                   files_in    = files_in,
                                   file_out    = file_out,
                                   compression = compression,
                                   nprint      = nprint,
                                   n_baseline  = n_baseline,
                                   thr_trigger = thr_trigger)

        # Parameters of the PMT csum.
        self.n_MAU       = n_MAU
        self.thr_MAU     = thr_MAU
        self.thr_csum_s1 = thr_csum_s1
        self.thr_csum_s2 = thr_csum_s2

        # Parameters of the SiPM signal
        self.n_MAU_sipm = n_MAU_sipm
        self.  thr_sipm =   thr_sipm

    def set_csum(self,
                   n_MAU     = 100,
                 thr_MAU     = 3   * units.adc,
                 thr_csum_s1 = 0.2 * units.pes,
                 thr_csum_s2 = 1.0 * units.pes):
        """Set CSUM parameters."""
        self.  n_MAU     =   n_MAU
        self.thr_MAU     = thr_MAU
        self.thr_csum_s1 = thr_csum_s1
        self.thr_csum_s2 = thr_csum_s2

    def set_sipm(self, n_MAU_sipm=100, thr_sipm=5*units.pes):
        """Cutoff for SiPMs."""
        self.  thr_sipm =   thr_sipm
        self.n_MAU_sipm = n_MAU_sipm

    def calibrated_pmt_sum(self, CWF):
        """Return the csum and csum_mau calibrated sums."""
        return cpf.calibrated_pmt_sum(CWF,
                                      self.adc_to_pes,
                                        n_MAU =   self.n_MAU,
                                      thr_MAU = self.thr_MAU)

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
                 run_number  = 0,
                 files_in    = None,
                 file_out    = None,
                 compression = 'ZLIB4',
                 nprint      = 10000,
                 n_baseline  = 28000,
                 thr_trigger = 5 * units.adc,
                 n_MAU       = 100,
                 thr_MAU     = 3.0 * units.adc,
                 thr_csum_s1 = 0.2 * units.adc,
                 thr_csum_s2 = 1.0 * units.adc,
                 n_MAU_sipm  = 100,
                 thr_sipm    = 5.0 * units.pes,
                 # Parameters added at this level
                 s1_params   = None,
                 s2_params   = None,
                 thr_sipm_s2 = 30 * units.pes):

        CalibratedCity.__init__(self,
                                run_number  = run_number,
                                files_in    = files_in,
                                file_out    = file_out,
                                compression = compression,
                                nprint      = nprint,
                                n_baseline  = n_baseline,
                                thr_trigger = thr_trigger,
                                n_MAU       = n_MAU,
                                thr_MAU     = thr_MAU,
                                thr_csum_s1 = thr_csum_s1,
                                thr_csum_s2 = thr_csum_s2,
                                n_MAU_sipm  = n_MAU_sipm,
                                  thr_sipm  =   thr_sipm)

        self.s1_params   = s1_params
        self.s2_params   = s2_params
        self.thr_sipm_s2 = thr_sipm_s2

    def set_pmap_params(self,
                        s1_params,
                        s2_params,
                        thr_sipm_s2 = 30 * units.pes):
        """Parameters for PMAP searches."""
        self.s1_params = s1_params
        self.s2_params = s2_params
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

    def find_S2Si(self, S2, sipmzs):
        """Return S2Si."""
        SIPM = cpf.select_sipm(sipmzs)
        S2Si = pmp.sipm_s2_dict(SIPM, S2, thr = self.thr_sipm_s2)
        return S2Si
