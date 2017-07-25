"""
code: base_cities.py
description: defines base classes for the IC cities.
credits: see ic_authors_and_legal.rst in /doc
last revised: JJGC, July-2017

"""

import sys
from argparse import Namespace
from glob     import glob
from time     import time
from os.path  import expandvars

import numpy  as np
import tables as tb

from .. core.core_functions     import merge_two_dicts
from .. core.configure          import configure
from .. core.exceptions         import NoInputFiles
from .. core.exceptions         import NoOutputFile
from .. core.exceptions         import UnknownRWF
from .. core.exceptions         import FileLoopMethodNotSet
from .. core.exceptions         import EventLoopMethodNotSet
from .. core.system_of_units_c  import units
from .. core.exceptions         import SipmEmptyList
from .. core.random_sampling    import NoiseSampler as SiPMsNoiseSampler

from .. database import load_db

from ..io                       import pmap_io          as pio
from ..io                       import pmap_io          as pio
from .. io. dst_io              import load_dst
from .. io.fee_io               import write_FEE_table

from ..reco                     import peak_functions_c as cpf
from ..reco                     import sensor_functions as sf
from ..reco                     import peak_functions   as pf
from ..reco                     import pmaps_functions  as pmp
from ..reco                     import pmaps_functions_c  as cpmp
from ..reco                     import dst_functions    as dstf
from ..reco                     import wfm_functions    as wfm
from ..reco                     import tbl_functions    as tbl
from .. reco.sensor_functions   import convert_channel_id_to_IC_id
from ..reco.corrections         import Correction
from ..reco.corrections         import Fcorrection
from ..reco.xy_algorithms       import find_algorithm

from .. evm.ic_containers       import S12Params
from .. evm.ic_containers       import S12Sum
from .. evm.ic_containers       import CSum
from .. evm.ic_containers       import DataVectors
from .. evm.ic_containers       import PmapVectors
from .. evm.ic_containers       import TriggerParams
from .. evm.event_model         import SensorParams
from .. evm.event_model         import KrEvent
from ..evm.event_model          import HitCollection
from ..evm.event_model          import Hit
from ..evm.nh5                  import DECONV_PARAM

from ..sierpe                   import blr
from ..sierpe                   import fee as FE

from .. types.ic_types          import minmax
from .. types.ic_types          import Counter
from .. types.ic_types          import NN

from .. daemons.idaemon         import invoke_daemon



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

    @classmethod
    def drive(cls, argv):
        """The drive methods allows invocation of the cities and their daemons
        through the command line.
        1. It reads the configuration file and calls the instances of the cities,
           passing a dictionary of arguments.
        2. It instantiates the daemons defined in the city configuration and
           sets them as attributes of the city.
        3. Calls the method "go" to launch execution of the city.
        4. Calls the method "end" for post-processing.

        """
        conf = configure(argv)
        opts = conf.as_namespace
        if not opts.hide_config:
            conf.display()
        if opts.print_config_only:
            return
        instance = cls(**conf.as_dict)

        # set the deamons
        if 'daemons' in conf.as_dict:
            d_list_name = conf.as_dict['daemons']
            instance.daemons = list(map(invoke_daemon, d_list_name))
        instance.go()
        instance.end()

    def go(self):
        """Launch the execution of the city (calling method run)
        and prints execution statistics.

        """
        t0 = time()
        self.run()
        t1 = time()
        dt = t1 - t0
        n_events = self.cnt.counter_value('n_events_tot')
        print("run {} evts in {} s, time/event = {}".format(n_events,
                                                            dt,
                                                            dt/n_events))

    def run(self):
        """The (base) run method of a city does the following chores:
        1. Calls a display_IO_info() function (to be provided by the concrete cities)
        2. open the output file
        3. Writes any desired parameters to output file (must be implemented by cities)
        4. gets the writers for the specific city.
        5. Pass the writers to the file_loop() method.
        6. returns the counter dictionary.

        """
        #import pdb; pdb.set_trace()
        self.display_IO_info()

        with tb.open_file(self.output_file, "w",
                          filters = tbl.filters(self.compression)) as h5out:

            self.write_parameters(h5out)
            self.writers = self.get_writers(h5out)
            self.file_loop()

    def end(self):
        """Postoprocessing after city execution:
        1. calls the end method of the daemons if they have been invoked.
        2. prints the counter dictionary

        """
        if hasattr(self, 'daemons'):
            for deamon in self.daemons:
                deamon.end()

        return self.cnt

    def display_IO_info(self):
        print("""
                 {} will run a max of {} events
                 Input Files = {}
                 Output File = {}
                          """.format(self.__class__.__name__, self.nmax, self.input_files, self.output_file))

    def file_loop(self):
        """Must be implemented by cities"""
        raise FileLoopMethodNotSet

    def event_loop(self):
        """Must be implemented by cities"""
        raise EventLoopMethodNotSet

    def write_parameters(self, h5out):
        """Must be implemented by cities"""
        pass

    def get_writers(self, h5out):
        """Must be implemented by cities"""
        pass

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


class RawCity(City):
    """A Raw city reads Raw Data.
       1. It provides a file loop that access Raw Data. The type of
       Raw Data is controled by the parameter raw_data_type and can be
       RWF (Raw Waveforms) or MCRD (Monte Carlo Raw Waveforms)
       2. It calls the event loop passing the RD, event and run info and (eventually)
       Monte Carlo Track info.

    """
    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.raw_data_type = self.conf.raw_data_type

    def file_loop(self):
        """
        The file loop of a Raw city:
        1. access RWF vectors for PMT and SiPMs
        2. access run and event info
        3. access MC track info
        4. calls event_loop passing a DataVector which holds rwf, mc and event info

        """
        for filename in self.input_files:
            print("Opening", filename, end="... ")
            with tb.open_file(filename, "r") as h5in:

                events_info = self.get_run_and_event_info(h5in)
                mc_tracks   = self.get_mc_tracks(h5in)
                dataVectors = 0
                NEVT        = 0

                if self.raw_data_type == 'RWF':
                    NEVT, pmtrwf, sipmrwf, _ = self.get_rwf_vectors(h5in)
                    dataVectors = DataVectors(pmt=pmtrwf, sipm=sipmrwf,
                                             mc=mc_tracks, events=events_info)

                    self.event_loop(NEVT, dataVectors)
                elif self.raw_data_type == 'MCRD':
                    first_event_no = self.event_number_from_input_file_name(filename)
                    NEVT, pmtrd, sipmrd     = self.get_rd_vectors(h5in)
                    dataVectors = DataVectors(pmt=pmtrd, sipm=sipmrd,
                                             mc=mc_tracks, events=events_info)

                    self.event_loop(NEVT, first_event_no, dataVectors)
                else:
                    raise UnknownRWF


class DeconvolutionCity(RawCity):
    """A Deconvolution city extends the City base class adding the
       deconvolution step, which transforms RWF into CWF.
       The parameters of the deconvolution are the number of samples
       used to compute the baseline (n_baseline) and the threshold to
       thr_trigger in the rising signal (thr_trigger).

       Since a Deconvolution city reads RWF, it is also a RawCity.

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
                                   rebin               = conf.s1_rebin)

        self.s2_params = S12Params(time = minmax(min   = conf.s2_tmin,
                                                 max   = conf.s2_tmax),
                                   stride              = conf.s2_stride,
                                   length = minmax(min = conf.s2_lmin,
                                                   max = conf.s2_lmax),
                                   rebin               = conf.s2_rebin)

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


class DstCity(City):
    """A DstCity reads a list of KDSTs """
    def __init__(self, **kwds):
        super().__init__(**kwds)

        conf = self.conf
        self._dst_group  = conf.dst_group
        self._dst_node   = conf.dst_node

        self.dsts = [load_dst(input_file, self._dst_group, self._dst_node)
                        for input_file in self.input_files]


class PCity(City):
    """A PCity reads PMAPS. Consequently it provides a file loop
       that access and serves to the event_loop the corresponding PMAPS
       vectors.
    """

    def __init__(self, **kwds):
        super().__init__(**kwds)

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


class KrCity(PCity):
    """A city that read pmaps and computes/writes a KrEvent"""

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.reco_algorithm = find_algorithm(self.conf.reco_algorithm)


    def compute_xy_position(self, s2si, peak_no):
        """
        Computes position using the integral of the charge
        in each SiPM.
        """
        IDs, Qs = cpmp.integrate_sipm_charges_in_peak(s2si, peak_no)
        xs, ys   = self.xs[IDs], self.ys[IDs]
        try:
            return self.reco_algorithm(np.stack((xs, ys)).T, Qs)
        except SipmEmptyList:
            return None

    def compute_z_and_dt(self, ts2, ts1):
        """
        Computes dt & z
        dt = ts2 - ts1 (in mus)
        z = dt * v_drift (i natural units)

        """
        dt  = ts2 - ts1
        z = dt * self.drift_v
        dt  *= units.ns / units.mus  #in mus
        return z, dt

    def create_kr_event(self, pmapVectors):
        """Create a Kr event:
        A Kr event treats the data as being produced by a point-like
        (krypton-like) interaction. Thus, the event is assumed to have
        negligible extension in z, and the transverse coordinates are
        computed integrating the temporal dependence of each sipm.
        """
        evt_number = pmapVectors.events
        evt_time   = pmapVectors.timestamps
        s1         = pmapVectors.s1
        s2         = pmapVectors.s2
        s2si       = pmapVectors.s2si

        evt       = KrEvent(evt_number, evt_time * 1e-3)

        evt.nS1 = s1.number_of_peaks
        for peak_no in s1.peak_collection():
            peak = s1.peak_waveform(peak_no)
            evt.S1w.append(peak.width)
            evt.S1h.append(peak.height)
            evt.S1e.append(peak.total_energy)
            evt.S1t.append(peak.tpeak)

        evt.nS2 = s2si.number_of_peaks
        for i, peak_no in enumerate(s2si.peak_collection()):
            peak = s2si.peak_waveform(peak_no)
            evt.S2w.append(peak.width/units.mus)
            evt.S2h.append(peak.height)
            evt.S2e.append(peak.total_energy)
            evt.S2t.append(peak.tpeak)

            clusters = self.compute_xy_position(s2si, peak_no)

            if clusters == None:

                evt.Nsipm.append(NN)
                evt.S2q  .append(NN)
                evt.X    .append(NN)
                evt.Y    .append(NN)
                evt.Xrms .append(NN)
                evt.Yrms .append(NN)
                evt.R    .append(NN)
                evt.Phi  .append(NN)
                evt.DT   .append(NN)
                evt.Z    .append(NN)
            else:
                assert len(clusters) == 1 #only one cluster
                c = clusters[0]
                evt.Nsipm.append(c.nsipm)
                evt.S2q  .append(c.Q)
                evt.X    .append(c.X)
                evt.Y    .append(c.Y)
                evt.Xrms .append(c.Xrms)
                evt.Yrms .append(c.Yrms)
                evt.R    .append(c.R)
                evt.Phi  .append(c.Phi)
                z, dt = self.compute_z_and_dt(evt.S2t[i], evt.S1t[0])
                evt.DT   .append(dt)
                evt.Z    .append(z)

        return evt


class HitCity(KrCity):
    """A city that reads PMPAS and computes/writes a hit event"""
    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.rebin  = self.conf.rebin

    def rebin_s2si(self, s2, s2si, rebin):
        """rebins s2d and sid dictionaries"""
        if rebin > 1:
            s2, s2si = pmp.rebin_s2si(s2, s2si, rebin)
        return s2, s2si

    def split_energy(self, e, clusters):
        if len(clusters) == 1:
            return [e]
        qs = np.array([c.Q for c in clusters])
        return e * qs / np.sum(qs)

    def compute_xy_position(self, s2sid_peak, slice_no):
        """Compute x-y position for each time slice. """
        #import pdb; pdb.set_trace()
        IDs, Qs  = cpmp.sipm_ids_and_charges_in_slice(s2sid_peak, slice_no)
        xs, ys   = self.xs[IDs], self.ys[IDs]
        try:
            return self.reco_algorithm(np.stack((xs, ys)).T, Qs)
        except SipmEmptyList:
            return None

    def create_hits_event(self, pmapVectors):
        """Create a hits_event:
        A hits event treats the data as being produced by a sequence
        of time slices. Thus, the event is assumed to have
        finite extension in z, and the transverse coordinates of the event are
        computed for each time slice in each sipm, creating a hit collection.
        """
        evt_number = pmapVectors.events
        evt_time   = pmapVectors.timestamps
        s1         = pmapVectors.s1
        s2         = pmapVectors.s2
        s2si       = pmapVectors.s2si

        hitc = HitCollection(evt_number, evt_time * 1e-3)

        # in order to compute z one needs to define one S1
        # for time reference. By default the filter will only
        # take events with exactly one s1. Otherwise, the
        # convention is to take the first peak in the S1 object
        # as reference.

        s1_t = s1.peak_waveform(0).tpeak
        # in general one rebins the time slices wrt the time slices
        # produces by pmaps. This is controlled by self.rebin which can
        # be set by parameter to a factor x pmaps-rebin.
        s2, s2si = self.rebin_s2si(s2, s2si, self.rebin)

        npeak = 0
        for peak_no, (t_peak, e_peak) in sorted(s2si.s2d.items()):
            for slice_no, (t_slice, e_slice) in enumerate(zip(t_peak, e_peak)):
                clusters = self.compute_xy_position(s2si.s2sid[peak_no], slice_no)
                if clusters == None:
                    continue
                # create hits only for those slices with OK clusters
                es       = self.split_energy(e_slice, clusters)
                z        = (t_slice - s1_t) * units.ns * self.drift_v
                for c, e in zip(clusters, es):
                    hit       = Hit(npeak, c, z, e)
                    hitc.hits.append(hit)
            npeak += 1

        return hitc

class TriggerEmulationCity(PmapCity):
    """Emulates the trigger in the FPGA.
       1. It is a PmapCity since the FPGA performs deconvolution and PMAP
       searches to set the trigger.

    """

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.trigger_params   = self.trigger_parameters()

    def trigger_parameters(self):
        """Simulate trigger parameters."""
        conf = self.conf
        height = minmax(min = conf.min_height, max = conf.max_height)
        charge = minmax(min = conf.min_charge, max = conf.max_charge)
        width  = minmax(min = conf.min_width , max = conf.max_width )
        return TriggerParams(trigger_channels    = conf.tr_channels,
                             min_number_channels = conf.min_number_channels,
                             charge              = charge * conf.data_mc_ratio,
                             height              = height * conf.data_mc_ratio,
                             width               = width)

    def emulate_trigger(self, RWF):
        """emulates trigger by simulating:
        1. online deconvolution of the waveforms.
        2. peak computation in the FPGA
        """

        CWF = self.deconv_pmt(RWF)
        IC_ids_selection = convert_channel_id_to_IC_id(self.DataPMT,
                                                       self.trigger_params.trigger_channels)

        peak_data = {}
        for pmt_id in IC_ids_selection:
            # Emulate zero suppression in the FPGA
            _, wfm_index = cpf.wfzs(CWF[pmt_id],
                                          threshold = self.trigger_params.height.min)

            # Emulate peak search (s2) in the FPGA
            s2 =  cpf.find_s2(CWF[pmt_id], wfm_index, **self.s2_params._asdict())
            peak_data[pmt_id] = s2

        return peak_data


class MonteCarloCity(TriggerEmulationCity):
    """A MonteCarloCity city:
     1. Simulates the response of sensors (energy plane and tracking plane)
        that transforms MCRD in RWF.
     2. Emulates the trigger prepocessor: the functionality is provided
        by the inheritance from TriggerEmulationCity.

    """

    def __init__(self, **kwds):
        super().__init__(**kwds)
        # Create instance of the noise sampler
        self.sp               = self.get_sensor_rd_params(self.input_files[0])
        self.noise_sampler    = SiPMsNoiseSampler(self.run_number, self.sp.SIPMWL, True)

    @staticmethod
    def simulate_sipm_response(event, sipmrd,
                               sipms_noise_sampler, sipm_adc_to_pes):
        """Add noise with the NoiseSampler class and return
        the noisy waveform (in adc counts)."""
        return sf.simulate_sipm_response(event, sipmrd, sipms_noise_sampler,
                                         sipm_adc_to_pes)

    @staticmethod
    def simulate_pmt_response(event, pmtrd, sipm_adc_to_pes):
        """ Full simulation of the energy plane response
        Input:
         1) extensible array pmtrd
         2) event_number

        returns:
        array of raw waveforms (RWF) obtained by convoluting pmtrd with the PMT
        front end electronics (LPF, HPF filters)
        array of BLR waveforms (only decimation)
        """
        return sf.simulate_pmt_response(event, pmtrd, sipm_adc_to_pes)


    @property
    def FE_t_sample(self):
        return FE.t_sample

    @staticmethod
    def write_simulation_parameters_table(filename):
        write_FEE_table(filename)
