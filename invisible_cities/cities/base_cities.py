"""
code: base_cities.py
description: defines base classes for the IC cities.
credits: see ic_authors_and_legal.rst in /doc
last revised: JJGC, July-2017

"""

from argparse    import Namespace
from glob        import glob
from time        import time
from os.path     import expandvars
from itertools   import chain

import numpy  as np
import tables as tb

from .. core.core_functions     import loc_elem_1d
from .. core.configure          import configure
from .. core.exceptions         import NoInputFiles
from .. core.exceptions         import NoOutputFile
from .. core.exceptions         import UnknownRWF
from .. core.exceptions         import XYRecoFail
from .. core.exceptions         import UnknownParameter
from .. core.system_of_units_c  import units
from .. core.random_sampling    import NoiseSampler as SiPMsNoiseSampler

from .. database import load_db

from .. io                      import pmaps_io    as pio
from .. io.dst_io               import load_dst
from .. io.fee_io               import write_FEE_table
from .. io.mcinfo_io            import mc_info_writer

from .. reco                    import calib_sensors_functions  as csf
from .. reco                    import paolina_functions        as paf
from .. reco                    import sensor_functions         as sf
from .. reco                    import peak_functions           as pkf
from .. reco                    import pmaps_functions          as pmf
from .. reco                    import tbl_functions            as tbl
from .. reco.sensor_functions   import convert_channel_id_to_IC_id
from .. reco.xy_algorithms      import corona

from .. evm.ic_containers       import S12Params
from .. evm.ic_containers       import S12Sum
from .. evm.ic_containers       import CSum
from .. evm.ic_containers       import CCWf
from .. evm.ic_containers       import DataVectors
from .. evm.ic_containers       import PmapVectors
from .. evm.ic_containers       import TriggerParams
from .. evm.event_model         import SensorParams
from .. evm.event_model         import KrEvent
from .. evm.event_model         import HitCollection
from .. evm.event_model         import Hit
from .. evm.event_model         import Cluster
from .. evm.event_model         import Voxel
from .. evm.pmaps               import S2
from .. evm.nh5                 import DECONV_PARAM

from .. sierpe                  import blr
from .. sierpe                  import fee as FE

from .. types.ic_types          import minmax
from .. types.ic_types          import Counters
from .. types.ic_types          import NN
from .. types.ic_types          import xy

from .. filters.s1s2_filter     import pmap_filter
from .. filters.s1s2_filter     import S12Selector
from .. filters.s1s2_filter     import S12SelectorOutput

from .. daemons.idaemon         import summon_daemon

from typing import Sequence


# TODO: move this somewhere else
from enum import Enum

class EventLoop(Enum):
    skip_this_event = 1
    terminate_loop  = 2


class City:
    """Base class for all cities.

       An IC city consumes data stored in the input_files and produce
       new data which is stored in the output_file. In addition to
       setting input and output files, the base class sets the print
       frequency and accesses the data base, storing as attributed
       several calibration coefficients
    """

    parameters = tuple("""print_config_only hide_config no_overrides
        no_files full_files config_file
        files_in file_out compression
        run_number print_mod event_range verbosity print_mod daemons""".split())

    def __init__(self, **kwds):
        """The init method of a city handles:
        1. provides access to an instance of counters (cnt) to be used by derived cities.
        2. provides access to the conf namespace
        3. provides access to input/output files.
        4. provides access to the data base.
        """

        self.detect_unknown_parameters(kwds)

        conf = Namespace(**kwds)
        self.conf = conf

        if not hasattr(conf, 'files_in'):
            raise NoInputFiles

        if not hasattr(conf, 'file_out'):
            raise NoOutputFile

        self.cnt = Counters(n_events_for_range = 0)
        self.input_files  = sorted(glob(expandvars(conf.files_in)))
        self.output_file  =             expandvars(conf.file_out)
        self.compression  = conf.compression
        self.run_number   = conf.run_number
        self.print_mod    = conf.print_mod  # default print frequency
        self.first_event, self.last_event = self._event_range()
        self.set_up_database()
        self.daemons = tuple(map(summon_daemon, kwds.get('daemons', [])))

    def detect_unknown_parameters(self, kwds):
        known = self.allowed_parameters()
        for name in kwds:
            if name not in known:
                raise UnknownParameter('{} does not expect {}.'.format(self.__class__.__name__, name))

    @classmethod
    def allowed_parameters(cls):
        return set(chain.from_iterable(base.parameters for base in cls.__mro__ if hasattr(base, 'parameters')))

    def _event_range(self):
        if not hasattr(self.conf, 'event_range'): return None, 1
        er = self.conf.event_range
        if not isinstance(er, Sequence): er = (er,)
        if len(er) == 1:                          return None, er[0]
        if len(er) == 2:                          return tuple(er)
        if len(er) == 0: ValueError('event_range needs at least one value')
        if len(er) >  2:
            raise ValueError('event_range accepts at most 2 values but was given {}'
                             .format(er))

    @classmethod
    def drive(cls, argv):
        """The drive methods allows invocation of the cities and their daemons
        through the command line.

        1. It reads the configuration files and CLI arguments and
           creates an instance of the city based on that configuration.
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
        instance = cls(**conf)
        instance.go()
        instance.end()
        return conf.as_namespace, instance.cnt

    def go(self):
        """Launch the execution of the city (calling method run)
        and prints execution statistics.
        """
        t0 = time()
        self.run()
        t1 = time()
        dt = t1 - t0
        n_events = self.cnt.n_events_tot
        print("run {} evts in {} s, time/event = {}".format(n_events,
                                                            dt,
                                                            dt / n_events))

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

    def end(self):
        """Postoprocessing after city execution:
        1. calls the end method of the daemons if they have been invoked.
        2. prints the counter dictionary
        """
        self.index_tables() # index cols in tables marked for indexing by city's writers

        for daemon in self.daemons:
            daemon.end()

        print(self.cnt)
        return self.cnt

    def index_tables(self):
        """
        -finds all tables in self.output_file (and in output_file2 if exists)
        -checks if any columns in the tables have been marked to be indexed by writers
        -indexes those columns
        """

        fileout_attrs = ('output_file', 'output_file2')
        for output_file in fileout_attrs:
            if hasattr(self, output_file):
                with tb.open_file(getattr(self, output_file), 'r+') as h5out:
                    # Walk over all tables in h5out
                    for table in h5out.walk_nodes(classname='Table'):
                        # Check for columns to index
                        if 'columns_to_index' not in table.attrs:  continue
                        # Index those columns
                        for colname in table.attrs.columns_to_index:
                            table.colinstances[colname].create_index()


    def display_IO_info(self):
        n_events_max = 'TODO: FIXME'
        print("""
                 {} will run a max of {} events
                 Input Files = {}
                 Output File = {}
                          """.format(self.__class__.__name__, n_events_max, self.input_files, self.output_file))

    def file_loop(self):
        """Must be implemented by cities"""
        raise NotImplementedError("Concrete City must implement `file_loop`")

    def event_loop(self):
        """Must be implemented by cities"""
        raise NotImplementedError("Concrete City must implement `event_loop`")

    def write_parameters(self, h5out):
        """Must be implemented by cities"""
        raise NotImplementedError("Concrete City must implement `write_parameters`")

    def get_writers(self, h5out):
        """Must be implemented by cities"""
        raise NotImplementedError("Concrete City must implement `get_writers`")

    def set_up_database(self):
        DataPMT       = load_db .DataPMT (self.run_number)
        DataSiPM      = load_db .DataSiPM(self.run_number)

        self.det_geo  = load_db.DetectorGeo()
        self.DataPMT  = DataPMT
        self.DataSiPM = DataSiPM

        self.xs                 = DataSiPM.X.values
        self.ys                 = DataSiPM.Y.values
        self.pmt_active         = DataPMT .Active.values
        self.sipm_active        = DataSiPM.Active.values
        self.pmt_active_list    = np.nonzero(self.pmt_active)[0].tolist()
        self.active_pmt_ids     = DataPMT.SensorID[DataPMT.Active == 1].values
        self.all_pmt_adc_to_pes = abs(DataPMT.adc_to_pes.values).astype(np.double)
        self.    pmt_adc_to_pes = self.all_pmt_adc_to_pes[DataPMT.Active == 1]
        self.   sipm_adc_to_pes = DataSiPM.adc_to_pes.values    .astype(np.double)
        self.coeff_c            = DataPMT.coeff_c.values        .astype(np.double)
        self.coeff_blr          = DataPMT.coeff_blr.values      .astype(np.double)
        self.noise_rms          = DataPMT.noise_rms.values      .astype(np.double)

        ## Charge resolution for sensor simulation
        pmt_single_pe_rms       = DataPMT.Sigma.values .astype(np.double)
        self.pmt_pe_resolution  = np.divide(pmt_single_pe_rms                             ,
                                            self.all_pmt_adc_to_pes                       ,
                                            out   = np.zeros_like(self.all_pmt_adc_to_pes),
                                            where = self.all_pmt_adc_to_pes != 0          )
        sipm_single_pe_rms      = DataSiPM.Sigma.values.astype(np.double)
        self.sipm_pe_resolution = np.divide(sipm_single_pe_rms                         ,
                                            self.sipm_adc_to_pes                       ,
                                            out   = np.zeros_like(self.sipm_adc_to_pes),
                                            where = self.sipm_adc_to_pes != 0          )

        sipm_x_masked = DataSiPM[DataSiPM.Active == 0].X.values
        sipm_y_masked = DataSiPM[DataSiPM.Active == 0].Y.values
        self.pos_sipm_masked = np.stack((sipm_x_masked, sipm_y_masked), axis=1)

    @property
    def monte_carlo(self):
        return self.run_number <= 0

    def conditional_print(self, evt, n_events_tot):
        if n_events_tot % self.print_mod == 0:
            print('event in file = {}, total = {}'
                  .format(evt, n_events_tot))

    def event_range_step(self):
        N = self.cnt.n_events_for_range
        self.cnt.n_events_for_range += 1

        if isinstance(self.last_event, int) and N >= self.last_event:
            # TODO: this side-effect should not be here
            print('reached event cutoff (= {})'
                  .format(self.last_event))
            return EventLoop.terminate_loop

        if self.first_event is not None and N < self.first_event:
            return EventLoop.skip_this_event

    def event_range_finished(self):
        N = self.cnt.n_events_for_range
        return isinstance(self.last_event, int) and N >= self.last_event


    def get_mc_info(self, h5in):
        "Return true MC information."
        if self.monte_carlo:
            return tbl.get_mc_info(h5in)
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
    def get_trigger_types(h5in):
        if 'Trigger/trigger' in h5in.root:
            return h5in.root.Trigger.trigger
        else:
            return None

    @staticmethod
    def get_trigger_channels(h5in):
        if 'Trigger/events' in h5in.root:
            return h5in.root.Trigger.events
        else:
            return None

    @staticmethod
    def trigger_type(evt, trg_types):
        return trg_types[evt][0] if trg_types else None

    @staticmethod
    def trigger_channels(evt, trg_channels):
        return trg_channels[evt] if trg_channels else None

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

    def mask_pmts(self, wfs):
        return csf.mask_sensors(wfs, self.pmt_active)

    def mask_sipms(self, wfs):
        return csf.mask_sensors(wfs, self.sipm_active)

    def file_loop(self):
        """
        The file loop of a Raw city:
        1. access RWF vectors for PMT and SiPMs
        2. access run and event info
        3. access MC track info
        4. calls event_loop passing a DataVector which holds rwf, mc and event info

        """
        for filename in self.input_files:
            print('Reading file...')
            if self.event_range_finished(): break
            print("Opening", filename, end="... ")
            with tb.open_file(filename, "r") as h5in:

                events_info  = self.get_run_and_event_info(h5in)
                trg_type     = self.get_trigger_types(h5in)
                trg_channels = self.get_trigger_channels(h5in)
                mc_info      = self.get_mc_info(h5in)
                dataVectors  = 0
                NEVT         = 0

                if self.raw_data_type == 'RWF':
                    NEVT, pmtrwf, sipmrwf, _ = self.get_rwf_vectors(h5in)
                    dataVectors = DataVectors(pmt=pmtrwf, sipm=sipmrwf,
                                              mc=mc_info, events=events_info,
                                              trg_type=trg_type,
                                              trg_channels=trg_channels)

                    self.event_loop(NEVT, dataVectors)
                elif self.raw_data_type == 'MCRD':
                    NEVT, pmtrd, sipmrd     = self.get_rd_vectors(h5in)
                    dataVectors = DataVectors(pmt=pmtrd,  sipm=sipmrd,
                                              mc=mc_info, events=events_info,
                                              trg_type=trg_type,
                                              trg_channels=trg_channels)

                    self.event_loop(NEVT, dataVectors)
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

    parameters = tuple("""raw_data_type n_baseline thr_trigger accum_discharge_length""".split())

    def __init__(self, **kwds):
        super().__init__(**kwds)
        conf = self.conf

        # BLR parameters
        self.n_baseline             = conf.n_baseline
        self.thr_trigger            = conf.thr_trigger
        self.accum_discharge_length = conf.accum_discharge_length

    def write_deconv_params(self, ofile):
        group = ofile.create_group(ofile.root, "DeconvParams")

        table = ofile.create_table(group,
                                   "DeconvParams",
                                   DECONV_PARAM,
                                   "deconvolution parameters",
                                   tbl.filters(self.compression))

        row = table.row
        row["N_BASELINE"]             = self.n_baseline
        row["THR_TRIGGER"]            = self.thr_trigger
        row["ACCUM_DISCHARGE_LENGTH"] = self.accum_discharge_length
        table.flush()

    def deconv_pmt(self, RWF, selection=None):
        """Deconvolve the RWF of the PMTs"""
        if selection is None:
            selection = self.pmt_active_list
        return blr.deconv_pmt(RWF,
                              self.coeff_c,
                              self.coeff_blr,
                              pmt_active             = selection,
                              n_baseline             = self.n_baseline,
                              thr_trigger            = self.thr_trigger,
                              accum_discharge_length = self.accum_discharge_length)


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
             The threshold can be used in two ways:
             - A single value for all sipms (thr_sipm_type = "common"). In this
               case, thr_sipm is the threshold value in pes.
             - A value for each SiPM, based on a common the percentage
               of noise reduction (thr_sipm_type = "individual"). In this case,
               thr_sipm is the percentage.
       """

    parameters = tuple("""n_mau thr_mau thr_csum_s1 thr_csum_s2 n_mau_sipm thr_sipm thr_sipm_type""".split())

    def __init__(self, **kwds):

        super().__init__(**kwds)
        conf = self.conf
        # Parameters of the PMT csum.
        self.n_MAU       = conf.n_mau
        self.thr_MAU     = conf.thr_mau
        self.thr_csum_s1 = conf.thr_csum_s1
        self.thr_csum_s2 = conf.thr_csum_s2

        # Parameters of the SiPM signal
        self.n_MAU_sipm  = conf.n_mau_sipm
        if   conf.thr_sipm_type.lower() == "common":
            # In this case, the threshold is a value of threshold in pes
            self.thr_sipm = conf.thr_sipm
        elif conf.thr_sipm_type.lower() == "individual":
            # In this case, the threshold is a percentual value
            noise_sampler = SiPMsNoiseSampler(self.run_number)
            self.thr_sipm = noise_sampler.compute_thresholds(conf.thr_sipm)
        else:
            raise ValueError(("Wrong value in thr_sipm_type. It must"
                              "be either 'Common' or 'Individual'"))

    def calibrate_pmts(self, CWF):
        """Return the csum and csum_mau calibrated sums."""
        return csf.calibrate_pmts(CWF, self.pmt_adc_to_pes,
                                    n_MAU = self.  n_MAU,
                                  thr_MAU = self.thr_MAU)

    def pmt_zero_suppression(self, wf, thr):
        return pkf.indices_and_wf_above_threshold(wf, thr)

    def calibrate_sipms(self, SiRWF):
        """Return the calibrated signal in the SiPMs."""

        return csf.calibrate_sipms(SiRWF,
                                   self.sipm_adc_to_pes,
                                   thr   = self.  thr_sipm,
                                   bls_mode=csf.BlsMode.mode)


class PmapCity(CalibratedCity):
    """A PMAP city extends a CalibratedCity, computing the S1, S2 and S2Si
       objects that togehter constitute a PMAP.
    """

    parameters = tuple("""
      s1_tmin s1_tmax s1_stride s1_lmin s1_lmax s1_rebin_stride
      s2_tmin s2_tmax s2_stride s2_lmin s2_lmax s2_rebin_stride
      thr_sipm_s2
      """.split())

    def __init__(self, **kwds):
        super().__init__(**kwds)
        conf = self.conf
        self.s1_params = S12Params(time   = minmax(min = conf.s1_tmin,
                                                   max = conf.s1_tmax),
                                   stride              = conf.s1_stride,
                                   length = minmax(min = conf.s1_lmin,
                                                   max = conf.s1_lmax),
                                   rebin_stride        = conf.s1_rebin_stride)

        self.s2_params = S12Params(time   = minmax(min = conf.s2_tmin,
                                                   max = conf.s2_tmax),
                                   stride              = conf.s2_stride,
                                   length = minmax(min = conf.s2_lmin,
                                                   max = conf.s2_lmax),
                                   rebin_stride        = conf.s2_rebin_stride)

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
        ccwfs, ccwfs_mau, cwf_sum, cwf_sum_mau = self.calibrate_pmts(CWF)

        #ZS sum for S1 and S2
        s1_indx, s1_ene = self.pmt_zero_suppression(cwf_sum_mau, thr=self.thr_csum_s1)
        s2_indx, s2_ene = self.pmt_zero_suppression(cwf_sum    , thr=self.thr_csum_s2)
        return (S12Sum(s1_ene  = s1_ene,
                       s1_indx = s1_indx,
                       s2_ene  = s2_ene,
                       s2_indx = s2_indx),
                CCWf(ccwf = ccwfs  , ccwf_mau = ccwfs_mau  ),
                CSum(csum = cwf_sum, csum_mau = cwf_sum_mau))


    def pmaps(self, s1_indx, s2_indx, ccwf, sipmzs):
        """Computes s1, s2 and s2si objects (PMAPS)"""
        return pkf.get_pmap(ccwf, s1_indx, s2_indx, sipmzs,
                            self.s1_params._asdict(),
                            self.s2_params._asdict(),
                            self.thr_sipm_s2,
                            self.active_pmt_ids)


class DstCity(City):
    """A DstCity reads a list of KDSTs """

    parameters = tuple("""dst_group dst_node""".split())

    def __init__(self, **kwds):
        super().__init__(**kwds)

        conf = self.conf
        self._dst_group  = conf.dst_group
        self._dst_node   = conf.dst_node

        self.dsts = [load_dst(input_file, self._dst_group, self._dst_node)
                        for input_file in self.input_files]


class PCity(City):
    """A PCity reads PMAPS. Consequently it provides a file loop and an event loop
       that access and serves to the event_loop the corresponding PMAPS
       vectors.
    """
    parameters = tuple("""drift_v write_mc_info
                          s1_nmin s1_nmax s1_emin s1_emax s1_wmin s1_wmax s1_hmin s1_hmax s1_ethr
                          s2_nmin s2_nmax s2_emin s2_emax s2_wmin s2_wmax s2_hmin s2_hmax s2_ethr
                          s2_nsipmmin s2_nsipmmax""".split())

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.drift_v = self.conf.drift_v
        self.s1s2_selector = S12Selector(**kwds)
        self.write_mc_info = self.conf.write_mc_info and self.monte_carlo

        self.cnt.init(n_events_tot                 = 0,
                      n_empty_pmaps                = 0,
                      n_events_not_s2si            = 0,
                      n_events_not_s1s2_filter     = 0,
                      n_events_not_s2si_filter     = 0,
                      n_events_selected            = 0)

    def get_mc_info_writer(self, h5out):
        return mc_info_writer(h5out) if self.write_mc_info else None

    def create_dst_event(self, pmapVectors, filter_output):
        """Must be implemented by any city derived from PCity"""
        raise NotImplementedError("Concrete City must implement `create_dst_event`")


    def event_loop(self, pmapVectors):
        """actions:
        1. loops over all PMAPS
        2. filter pmaps
        3. write dst_event
        """

        write         = self.writers
        event_numbers = pmapVectors.events
        timestamps    = pmapVectors.timestamps
        pmaps         = pmapVectors.pmaps
        mc_info       = pmapVectors.mc

        for evt_number, evt_time in zip(event_numbers, timestamps):
            self.conditional_print(self.cnt.n_events_tot, self.cnt.n_events_selected)

            # Count events in and break if necessary before filtering
            what_next = self.event_range_step()
            if what_next is EventLoop.skip_this_event: continue
            if what_next is EventLoop.terminate_loop : break
            self.cnt.n_events_tot += 1

            pmap = pmaps.get(evt_number, None)
            if pmap is None:
                self.cnt.n_empty_pmaps += 1
                continue

            # filtering
            filter_output = self.filter_event(pmap)
            if not filter_output.passed:
                continue

            self.cnt.n_events_selected += 1

            # create DST event & write to file
            pmapVectors = PmapVectors(pmaps=pmap,
                                      events=evt_number,
                                      timestamps=evt_time,
                                      mc=None)
            evt = self.create_dst_event(pmapVectors, filter_output)
            write.dst(evt)
            if self.write_mc_info:
                write.mc(mc_info, evt_number)

    def file_loop(self):
        """
        actions:
        1. access pmaps (si_dicts )
        2. access run and event info
        3. call event_loop
        """

        for filename in self.input_files:
            if self.event_range_finished(): break
            print("Opening {filename}".format(**locals()), end="...\n")

            try:
                pmaps = self.get_pmaps_dicts(filename)
            except (ValueError, tb.exceptions.NoSuchNodeError):
                print("Empty file. Skipping.")
                continue

            with tb.open_file(filename) as h5in:
                mc_info = None
                if self.write_mc_info:
                    # Save time when we are not interested in mc tracks
                    mc_info = self.get_mc_info(h5in)

                event_numbers, timestamps = \
                self.event_numbers_and_timestamps_from_file_name(filename)

                pmapVectors = PmapVectors(pmaps      = pmaps,
                                          events     = event_numbers,
                                          timestamps = timestamps,
                                          mc         = mc_info)

                self.event_loop(pmapVectors)

    def filter_event(self, pmap):
        """Filter the event in terms of s1, s2, s2si"""
        # loop event away if any signal (s1, s2 or s2si) not present
        empty = S12SelectorOutput(False, [], [])

        # filters in s12 and s2si
        f = pmap_filter(self.s1s2_selector, pmap)
        self.cnt.n_events_not_s1s2_filter += int(not f.passed)
        return f


class KrCity(PCity):
    """A city that read pmaps and computes/writes a KrEvent"""

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.cnt.init(n_events_more_than_1_cluster = 0)

    parameters = tuple("lm_radius new_lm_radius msipm qlm qthr".split())

    def compute_xy_position(self, sr):
        """
        Computes position using the integral of the charge
        in each SiPM.
        """
        IDs = sr.ids
        Qs  = sr.sum_over_times
        xs, ys   = self.xs[IDs], self.ys[IDs]
        return corona(np.stack((xs, ys), axis=1), Qs,
                      Qthr           =  self.conf.qthr,
                      Qlm            =  self.conf.qlm,
                      lm_radius      =  self.conf.lm_radius,
                      new_lm_radius  =  self.conf.new_lm_radius,
                      msipm          =  self.conf.msipm,
                      masked_sipm    =  self.pos_sipm_masked)

    def compute_z_and_dt(self, ts2, ts1s):
        """
        Computes dt & z
        dt = ts2 - ts1 (in mus)
        z = dt * v_drift (i natural units)

        """
        dt   = ts2 - np.array(ts1s)
        z    = dt * self.drift_v
        dt  *= units.ns / units.mus  #in mus
        return z, dt

    def create_kr_event(self, pmapVectors, filter_output):
        """Create a Kr event:
        A Kr event treats the data as being produced by a point-like
        (krypton-like) interaction. Thus, the event is assumed to have
        negligible extension in z, and the transverse coordinates are
        computed integrating the temporal dependence of each sipm.
        """
        evt_number = pmapVectors.events
        evt_time   = pmapVectors.timestamps
        pmap       = pmapVectors.pmaps
        evt        = KrEvent(evt_number, evt_time * 1e-3)

        evt.nS1 = 0
        for passed, peak in zip(filter_output.s1_peaks, pmap.s1s):
            if not passed: continue

            evt.nS1 += 1
            evt.S1w.append(peak.width)
            evt.S1h.append(peak.height)
            evt.S1e.append(peak.total_energy)
            evt.S1t.append(peak.time_at_max_energy)

        evt.nS2 = 0
        for passed, peak in zip(filter_output.s2_peaks, pmap.s2s):
            if not passed: continue

            evt.nS2 += 1
            evt.S2w.append(peak.width/units.mus)
            evt.S2h.append(peak.height)
            evt.S2e.append(peak.total_energy)
            evt.S2t.append(peak.time_at_max_energy)

            try:
                clusters = self.compute_xy_position(peak.sipms)
                # if there is more than one cluster compare the energy measured
                # in the tracking plane with the energy measured in the energy plane
                # and thake the cluster where both energies are closer.
                c = 0
                if len(clusters) == 1:
                    c = clusters[0]
                else:
                    cQ = [c.Q for c in clusters]
                    self.cnt.n_events_more_than_1_cluster += 1
                    print('found case with more than one cluster')
                    print('clusters charge = {}'.format(cQ))

                    c_closest = np.amax([c.Q for c in clusters])

                    print('c_closest = {}'.format(c_closest))
                    c = clusters[loc_elem_1d(cQ, c_closest)]
                    print('c_chosen = {}'.format(c))

                Z, DT = self.compute_z_and_dt(evt.S2t[-1], evt.S1t)
                Zrms  = peak.rms / units.mus

                evt.Nsipm.append(c.nsipm)
                evt.S2q  .append(c.Q)
                evt.X    .append(c.X)
                evt.Y    .append(c.Y)
                evt.Xrms .append(c.Xrms)
                evt.Yrms .append(c.Yrms)
                evt.R    .append(c.R)
                evt.Phi  .append(c.Phi)
                evt.DT   .append(DT)
                evt.Z    .append(Z)
                evt.Zrms .append(Zrms)
            except XYRecoFail:
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
                evt.Zrms .append(NN)
        return evt


class HitCity(KrCity):
    """A city that reads PMAPS and computes/writes a hit event"""
    parameters = tuple("rebin".split())

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.rebin  = self.conf.rebin

    def compute_xy_position(self, sr, slice_no):
        """Compute x-y position for each time slice. """
        IDs = sr.ids
        Qs  = sr.time_slice(slice_no)
        xs, ys = self.xs[IDs], self.ys[IDs]

        return corona(np.stack((xs, ys), axis=1), Qs,
                      Qthr           =  self.conf.qthr,
                      Qlm            =  self.conf.qlm,
                      lm_radius      =  self.conf.lm_radius,
                      new_lm_radius  =  self.conf.new_lm_radius,
                      msipm          =  self.conf.msipm,
                      masked_sipm    =  self.pos_sipm_masked)

    def compute_xy_peak_position(self, sr):
        """
        Computes position using the integral of the charge
        in each SiPM. Config parameters set equal to the standard kDST values.
        """
        IDs = sr.ids
        Qs = sr.sum_over_times
        xs, ys   = self.xs[IDs], self.ys[IDs]
        return corona(np.stack((xs, ys), axis=1), Qs,
                      Qthr           =  1.,
                      Qlm            =  0.,
                      lm_radius      =  -1.,
                      new_lm_radius  =  -1.,
                      msipm          =  1)

    def split_energy(self, e, clusters):
        if len(clusters) == 1:
            return [e]
        qs = np.array([c.Q for c in clusters])
        return e * qs / np.sum(qs)

    def create_hits_event(self, pmapVectors, filter_output):
        """Create a hits_event:
        A hits event treats the data as being produced by a sequence
        of time slices. Thus, the event is assumed to have
        finite extension in z, and the transverse coordinates of the event are
        computed for each time slice in each sipm, creating a hit collection.
        """
        evt_number = pmapVectors.events
        evt_time   = pmapVectors.timestamps
        pmap       = pmapVectors.pmaps

        hitc = HitCollection(evt_number, evt_time * 1e-3)

        # in order to compute z one needs to define one S1
        # for time reference. By default the filter will only
        # take events with exactly one s1. Otherwise, the
        # convention is to take the first peak in the S1 object
        # as reference.
        first_s1 = np.where(filter_output.s1_peaks)[0][0]
        s1_t     = pmap.s1s[first_s1].time_at_max_energy

        # here hits are computed for each peak and each slice.
        # In case of an exception, a hit is still created with a NN cluster.
        # (NN cluster is a cluster where the energy is an IC not number NN)
        # this allows to keep track of the energy associated to non reonstructed hits.
        for peak_no, (passed, peak) in enumerate(zip(filter_output.s2_peaks,
                                                     pmap.s2s)):
            if not passed: continue

            peak = pmf.rebin_peak(peak, self.rebin)

            try:
                eventCluster = self.compute_xy_peak_position(peak.sipms)
                # if there is more than one cluster compare the energy measured
                # in the tracking plane with the energy measured in the energy plane
                # and thake the cluster where both energies are closer.
                c = 0
                if len(eventCluster) == 1:
                    c = eventCluster[0]
                else:
                    cQ = [c.Q for c in eventCluster]
                    self.cnt.n_events_more_than_1_cluster += 1
                    print('found case with more than one cluster')
                    print('eventCluster charge = {}'.format(cQ))

                    c_closest = np.amax([c.Q for c in eventCluster])

                    print('c_closest = {}'.format(c_closest))
                    c = eventCluster[loc_elem_1d(cQ, c_closest)]
                    print('c_chosen = {}'.format(c))
                xy_peak = xy(c.X, c.Y)
            except:
                xy_peak = xy(0, 0)

            for slice_no, t_slice in enumerate(peak.times):
                z_slice = (t_slice - s1_t) * units.ns * self.drift_v
                e_slice = peak.pmts.sum_over_sensors[slice_no]
                try:
                    clusters = self.compute_xy_position(peak.sipms, slice_no)
                    es       = self.split_energy(e_slice, clusters)
                    for c, e in zip(clusters, es):
                        hit       = Hit(peak_no, c, z_slice, e, xy_peak)
                        hitc.hits.append(hit)
                except XYRecoFail:
                    c = Cluster(NN, xy(0,0), xy(0,0), 0)
                    hit       = Hit(peak_no, c, z_slice, e_slice, xy_peak)
                    hitc.hits.append(hit)

        return hitc


class TrackCity(HitCity):
    """A city that reads PMPAS and computes/writes a track event"""
    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.voxel_dimensions  = self.conf.voxel_dimensions # type: np.ndarray
        self.blob_radius       = self.blob_radius # type: float

    def voxelize_hits(self, hits : Sequence[Hit]) -> Sequence[Voxel]:
        """1. Hits are enclosed by a bounding box.
           2. Boundix box is discretized (via a hitogramdd).
           3. The energy of all the hits insidex each discreet "voxel" is added.

         """
        return paf.voxelize_hits(hits, self.voxel_dimensions)


class TriggerEmulationCity(PmapCity):
    """Emulates the trigger in the FPGA.
       1. It is a PmapCity since the FPGA performs deconvolution and PMAP
       searches to set the trigger.
    """

    parameters = tuple("""tr_channels min_number_channels
        min_height max_height min_width max_width min_charge max_charge
        data_mc_ratio""".split())

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.trigger_params   = self.trigger_parameters()

        self.IC_ids_selection = convert_channel_id_to_IC_id(self.DataPMT,
                                                            self.trigger_params.trigger_channels)

        if not np.all(np.in1d(self.IC_ids_selection, self.pmt_active_list)):
            raise ValueError("Attempt to trigger in masked PMT")

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
        CWFs = self.deconv_pmt(RWF, self.IC_ids_selection.tolist())

        peak_data = {}
        for pmt_id, cwf in zip(self.IC_ids_selection, CWFs):
            # Emulate zero suppression in the FPGA
            wfm_index, _ = \
            pkf.indices_and_wf_above_threshold(cwf, thr = self.trigger_params.height.min)

            # Emulate peak search (s2) in the FPGA
            s2 = pkf.find_peaks(cwf, wfm_index,
                                Pk      = S2,
                                pmt_ids = [-1],
                                **self.s2_params._asdict())
            peak_data[pmt_id] = s2

        return peak_data


class MonteCarloCity(TriggerEmulationCity):
    """A MonteCarloCity city:
     1. Simulates the response of sensors (energy plane and tracking plane)
        that transforms MCRD in RWF.
     2. Emulates the trigger prepocessor: the functionality is provided
        by the inheritance from TriggerEmulationCity.
    """

    parameters = tuple("""sipm_noise_cut""".split())

    def __init__(self, **kwds):
        super().__init__(**kwds)
        # Create instance of the noise sampler
        self.sp               = self.get_sensor_rd_params(self.input_files[0])
        self.noise_sampler    = SiPMsNoiseSampler(self.run_number, self.sp.SIPMWL, True)


    def simulate_sipm_response(self, event, sipmrd):
        """Add noise with the NoiseSampler class and return
        the noisy waveform (in adc counts)."""
        return sf.simulate_sipm_response(event, sipmrd,
                                         self.noise_sampler,
                                         self.sipm_adc_to_pes,
                                         self.sipm_pe_resolution,
                                         self.run_number)


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
        return sf.simulate_pmt_response(event, pmtrd,
                                        self.all_pmt_adc_to_pes,
                                        self.pmt_pe_resolution,
                                        self.run_number)

    @property
    def FE_t_sample(self):
        return FE.t_sample

    @staticmethod
    def write_simulation_parameters_table(filename):
        write_FEE_table(filename)
