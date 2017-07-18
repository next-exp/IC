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
from argparse import Namespace
from operator import attrgetter
from glob     import glob
from time     import time
from os.path  import expandvars

import numpy  as np
import tables as tb

from .. core.configure         import configure
from .. core.exceptions        import NoInputFiles
from .. core.exceptions        import NoOutputFile
from .. types.ic_types         import minmax
from .. core.system_of_units_c import units
from .. core                   import fit_functions        as fitf

from .. database import load_db

from ..io                 import pmap_io          as pio

from ..reco               import peak_functions_c as cpf
from ..reco               import sensor_functions as sf
from ..reco               import peak_functions   as pf
from ..reco               import pmaps_functions  as pmp
from ..reco               import dst_functions    as dstf
from ..reco               import wfm_functions    as wfm
from ..reco               import tbl_functions    as tbl
from ..io                 import pmap_io          as pio
from .. io.fee_io         import write_FEE_table

from .. evm.ic_containers      import S12Params
from .. evm.ic_containers      import S12Sum
from .. evm.ic_containers      import CSum
from .. evm.ic_containers      import DataVectors
from .. evm.ic_containers      import PmapVectors
from ..evm.event_model         import SensorParams
from ..evm.nh5                 import DECONV_PARAM
from ..reco.corrections        import Correction
from ..reco.corrections        import Fcorrection
from ..reco.corrections        import LifetimeCorrection
from ..reco.xy_algorithms      import find_algorithm

from ..sierpe                   import blr
from ..sierpe                   import fee as FE
from .. types.ic_types          import Counter


def merge_two_dicts(a,b):
    return {**a, **b}


class City:
    """Base class for all cities.
       An IC city consumes data stored in the input_files and produce new data
       which is stored in the output_file. In addition to setting input and
       output files, the base class sets the print frequency and accesses
       the data base, storing as attributed several calibration coefficients

     """

    def __init__(self, **kwds):
        """The init method of a city handles:
        1. provides access to an instance of counters (cnt) to be used by derived cities.
        2. provides access to the conf namespace
        3. provides access to input/output files.
        4. provides access to the data base.
        """

        self.cnt = Counter()
        conf = Namespace(**kwds)
        self.conf = conf

        if not hasattr(conf, 'files_in'):
            raise NoInputFiles

        if not hasattr(conf, 'file_out'):
            raise NoOutputFile


        self.input_files = sorted(glob(expandvars(conf.files_in)))
        self.output_file =             expandvars(conf.file_out)
        self.compression = conf.compression
        self.run_number  = conf.run_number
        self.nprint      = conf.nprint  # default print frequency
        self.nmax        = conf.nmax

        self.set_up_database()

    def run(self):
        """The (base) run method of a city does the following chores:
        1. Calls a display_IO_info() function (to be provided by the concrete cities)
        2. open the output file
        3. Writes any desired parameters to output file (must be implemented by cities)
        4. gets the writers for the specific city.
        5. Pass the writers to the file_loop() method.
        6. returns the counter dictionary.
        """
        self.display_IO_info()


        with tb.open_file(self.output_file, "w",
                          filters = tbl.filters(self.compression)) as h5out:

            self.write_parameters(h5out)
            self.writers = self.get_writers(h5out)
            self.file_loop()

        return self.cnt

    def display_IO_info(self):
        print("""
                 {} will run a max of {} events
                 Input Files = {}
                 Output File = {}
                          """.format(self.__class__.__name__, self.nmax, self.input_files, self.output_file))

    def file_loop(self):
        """Must be implemented by concrete cities"""
        pass

    def event_loop(self):
        """Must be implemented by concrete cities"""
        pass

    def write_parameters(self, h5out):
        """Must be implemented by concrete cities"""
        pass

    def get_writers(self, h5out):
        """Must be implemented by concrete cities"""
        pass

    @classmethod
    def drive(cls, argv):
        conf = configure(argv)
        opts = conf.as_namespace
        if not opts.hide_config:
            conf.display()
        if opts.print_config_only:
            return
        instance = cls(**conf.as_dict)
        instance.go()

    def go(self):
        t0 = time()
        cnt = self.run()
        t1 = time()
        dt = t1 - t0
        n_events = cnt.counter_value('n_events_tot')
        print("run {} evts in {} s, time/event = {}".format(n_events, dt, dt/n_events))
        print(cnt)

    def set_up_database(self):
        DataPMT       = load_db.DataPMT (self.run_number)
        DataSiPM      = load_db.DataSiPM(self.run_number)
        self.det_geo  = load_db.DetectorGeo()
        self.DataPMT  = DataPMT
        self.DataSiPM = DataSiPM

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

    def conditional_print(self, evt, n_events_tot):
        if n_events_tot % self.nprint == 0:
            print('event in file = {}, total = {}'
                  .format(evt, n_events_tot))

    def max_events_reached(self, n_events_in):
        if self.nmax < 0:
            return False
        if n_events_in == self.nmax:
            print('reached max nof of events (= {})'
                  .format(self.nmax))
            return True
        return False

    def get_mc_tracks(self, h5in):
        "Return RWF vectors and sensor data."
        if self.monte_carlo:
            return tbl.get_mc_tracks(h5in)
        else:
            return None

    @staticmethod
    def get_rwf_vectors(h5in):
        "Return RWF vectors and sensor data."
        return tbl.get_rwf_vectors(h5in)

    @staticmethod
    def get_rd_vectors(h5in):
        "Return MC RD vectors and sensor data."
        return tbl.get_rd_vectors(h5in)

    def get_sensor_rd_params(self, filename):
        """Return MCRD sensors.
           pmtrd.shape returns the length of the RD PMT vector
           (1 ns bins). PMTWL_FEE is the length of the RWF vector
           obtained by divinding the RD PMT vector and the sample
           time of the electronics (25 ns). """
        with tb.open_file(filename, "r") as h5in:
            #pmtrd, sipmrd = self._get_rd(h5in)
            _, pmtrd, sipmrd = tbl.get_rd_vectors(h5in)
            _, NPMT,   PMTWL = pmtrd .shape
            _, NSIPM, SIPMWL = sipmrd.shape
            PMTWL_FEE = int(PMTWL // self.FE_t_sample)
            return SensorParams(NPMT, PMTWL_FEE, NSIPM, SIPMWL)

    @staticmethod
    def get_sensor_params(filename):
        return tbl.get_sensor_params(filename)

    @staticmethod
    def get_run_and_event_info(h5in):
        return h5in.root.Run.events


    @staticmethod
    def event_and_timestamp(evt, events_info):
        return events_info[evt]


    @staticmethod
    def event_number_from_input_file_name(filename):
        return tbl.event_number_from_input_file_name(filename)

    @staticmethod
    def event_numbers_and_timestamps_from_file_name(filename):
        return tbl.get_event_numbers_and_timestamps_from_file_name(filename)


    def _get_rwf(self, h5in):
        "Return raw waveforms for SIPM and PMT data"
        return (h5in.root.RD.pmtrwf,
                h5in.root.RD.sipmrwf,
                h5in.root.RD.pmtblr)

    def _get_rd(self, h5in):
        "Return (MC) raw data waveforms for SIPM and PMT data"
        return (h5in.root.pmtrd,
                h5in.root.sipmrd)

    @staticmethod
    def get_pmaps_dicts(filename):
        return pio.load_pmaps(filename)

    @staticmethod
    def get_pmaps_from_dicts(s1_dict, s2_dict, s2si_dict, evt_number):
        return pio.s1_s2_si_from_pmaps(s1_dict, s2_dict, s2si_dict, evt_number)



class SensorResponseCity(City):
    """A SensorResponseCity city extends the City base class adding the
       response (Monte Carlo simulation) of the energy plane and
       tracking plane sensors (PMTs and SiPMs).
    """

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.sipm_noise_cut = self.conf.sipm_noise_cut

    def simulate_sipm_response(self, event, sipmrd,
                               sipms_noise_sampler):
        """Add noise with the NoiseSampler class and return
        the noisy waveform (in adc counts)."""
        return sf.simulate_sipm_response(event, sipmrd, sipms_noise_sampler, self.sipm_adc_to_pes)

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
        return sf.simulate_pmt_response(event, pmtrd, self.adc_to_pes)

    def file_loop(self):
        """
        The file loop of a SensorResponseCity:
        1. access RD vectors for PMT
        2. access run and event info
        3. access MC track info
        4. calls event_loop
        """

        for filename in self.input_files:
            first_event_no = self.event_number_from_input_file_name(filename)
            print("Opening file {filename} with first event no {first_event_no}"
                  .format(**locals()))
            with tb.open_file(filename, "r") as h5in:
                # NEVT is the total number of events in pmtrd and sipmrd
                # pmtrd = pmrtd[events][NPMT][rd_waveform]
                # sipmrd = sipmrd[events][NPMT][rd_waveform]
                NEVT, pmtrd, sipmrd = self.get_rd_vectors(h5in)
                events_info         = self.get_run_and_event_info(h5in)
                mc_tracks           = self.get_mc_tracks(h5in)
                dataVectors = DataVectors(pmt=pmtrd, sipm=sipmrd,
                                          mc=mc_tracks, events=events_info)
                self.event_loop(NEVT, first_event_no, dataVectors)


    @property
    def FE_t_sample(self):
        return FE.t_sample

    @staticmethod
    def write_simulation_parameters_table(filename):
        write_FEE_table(filename)


class DeconvolutionCity(City):
    """A Deconvolution city extends the City base class adding the
       deconvolution step, which transforms RWF into CWF.
       The parameters of the deconvolution are the number of samples
       used to compute the baseline (n_baseline) and the threshold to
       thr_trigger in the rising signal (thr_trigger)
    """

    def __init__(self, **kwds):
        super().__init__(**kwds)
        conf = self.conf

        # BLR parameters
        self.n_baseline            = conf.n_baseline
        self.thr_trigger           = conf.thr_trigger
        self.acum_discharge_length = conf.acum_discharge_length

    def write_deconv_params(self, ofile):
        group = ofile.create_group(ofile.root, "DeconvParams")

        table = ofile.create_table(group,
                                   "DeconvParams",
                                   DECONV_PARAM,
                                   "deconvolution parameters",
                                   tbl.filters(self.compression))

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

    def file_loop(self):
        """
        The file loop of a deconvolution city:
        1. access RWF vectors for PMT and SiPMs
        2. access run and event info
        3. access MC track info
        4. calls event_loop
        """
        # import pdb; pdb.set_trace()

        for filename in self.input_files:
            print("Opening", filename, end="... ")
            with tb.open_file(filename, "r") as h5in:

                NEVT, pmtrwf, sipmrwf, _ = self.get_rwf_vectors(h5in)
                events_info              = self.get_run_and_event_info(h5in)
                mc_tracks                = self.get_mc_tracks(h5in)
                dataVectors              = DataVectors(pmt=pmtrwf, sipm=sipmrwf,
                                                       mc=mc_tracks,
                                                       events=events_info)
                self.event_loop(NEVT, dataVectors)

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

    def __init__(self, **kwds):

        super().__init__(**kwds)
        conf = self.conf
        # Parameters of the PMT csum.
        self.n_MAU       = conf.n_mau
        self.thr_MAU     = conf.thr_mau
        self.thr_csum_s1 = conf.thr_csum_s1
        self.thr_csum_s2 = conf.thr_csum_s2

        # Parameters of the SiPM signal
        self.n_MAU_sipm = conf.n_mau_sipm
        self.  thr_sipm = conf.  thr_sipm

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

    def __init__(self, **kwds):
        super().__init__(**kwds)
        conf = self.conf
        self.s1_params = S12Params(time = minmax(min   = conf.s1_tmin,
                                                 max   = conf.s1_tmax),
                                   stride              = conf.s1_stride,
                                   length = minmax(min = conf.s1_lmin,
                                                   max = conf.s1_lmax),
                                   rebin               = False)

        self.s2_params = S12Params(time = minmax(min   = conf.s2_tmin,
                                                 max   = conf.s2_tmax),
                                   stride              = conf.s2_stride,
                                   length = minmax(min = conf.s2_lmin,
                                                   max = conf.s2_lmax),
                                   rebin               = True)

        self.thr_sipm_s2 = conf.thr_sipm_s2


    def pmt_transformation(self, RWF):
        """
        Performs the transformations in the PMT plane, namely:
        1. Deconvolve the raw waveforms (RWF) to obtain corrected waveforms (CWF)
        2. Computes the calibrated sum of the PMTs
        3. Finds the zero suppressed waveforms to search for s1 and s2

        """

        # deconvolve
        CWF = self.deconv_pmt(RWF)
        # calibrated PMT sum
        csum, csum_mau = self.calibrated_pmt_sum(CWF)
        #ZS sum for S1 and S2
        s1_ene, s1_indx = self.csum_zs(csum_mau, threshold =
                                           self.thr_csum_s1)
        s2_ene, s2_indx = self.csum_zs(csum,     threshold =
                                           self.thr_csum_s2)
        return (S12Sum(s1_ene  = s1_ene,
                          s1_indx = s1_indx,
                          s2_ene  = s2_ene,
                          s2_indx = s2_indx),
                CSum(csum = csum, csum_mau = csum_mau)
                    )



    def pmaps(self, s1_indx, s2_indx, csum, sipmzs):
        """Computes s1, s2 and s2si objects (PMAPS)"""
        s1 = cpf.find_s1(csum, s1_indx, **self.s1_params._asdict())
        s1 = cpf.correct_s1_ene(s1.s1d, csum)
        s2 = cpf.find_s2(csum, s2_indx, **self.s2_params._asdict())
        s2si = cpf.find_s2si(sipmzs, s2.s2d, thr = self.thr_sipm_s2)
        return s1, s2, s2si


class MapCity(City):
    def __init__(self, **kwds):
        super().__init__(**kwds)

        conf = self.conf
        required_names = 'lifetime u_lifetime xmin xmax ymin ymax xbins ybins'.split()
        lifetime, u_lifetime, xmin, xmax, ymin, ymax, xbins, ybins = attrgetter(*required_names)(conf)

        self.  _lifetimes = [lifetime]   if not np.shape(  lifetime) else   lifetime
        self._u_lifetimes = [u_lifetime] if not np.shape(u_lifetime) else u_lifetime
        self._lifetime_corrections = tuple(map(LifetimeCorrection, self._lifetimes, self._u_lifetimes))

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


class HitCity(City):
    def __init__(self, **kwds):
        super().__init__(**kwds)
        conf  = self.conf
        self.rebin          = conf.rebin
        self.reco_algorithm = find_algorithm(conf.reco_algorithm)

    def rebin_s2si(self, s2, s2si):
        """rebins s2d and sid dictionaries"""
        if self.rebin > 1:
            s2, s2si = pmp.rebin_s2si(s2, s2si, self.rebin)
        return s2, s2si

    def split_energy(self, e, clusters):
        if len(clusters) == 1:
            return [e]
        qs = np.array([c.Q for c in clusters])
        return e * qs / np.sum(qs)

    def compute_xy_position(self, si, slice_no):
        si_slice = pmp.select_si_slice(si, slice_no)
        IDs, Qs  = pmp.integrate_sipm_charges_in_peak(si)
        xs, ys   = self.xs[IDs], self.ys[IDs]
        return self.reco_algorithm(np.stack((xs, ys)).T, Qs)

    def file_loop(self):
        """
        actions:
        1. access pmaps (si_dicts )
        2. access run and event info
        3. call event_loop
        """

        for filename in self.input_files:
            print("Opening {filename}".format(**locals()), end="... ")

            try:
                s1_dict, s2_dict, s2si_dict = self.get_pmaps_dicts(filename)
            except (ValueError, tb.exceptions.NoSuchNodeError):
                print("Empty file. Skipping.")
                continue

            event_numbers, timestamps = self.event_numbers_and_timestamps_from_file_name(filename)
            pmapVectors               = PmapVectors(s1=s1_dict, s2=s2_dict, s2si=s2si_dict,
                                                    events=event_numbers, timestamps=timestamps)

            self.event_loop(pmapVectors)
