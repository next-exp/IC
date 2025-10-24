from functools       import wraps
from functools       import partial
from collections.abc import Sequence
from argparse        import Namespace
from glob            import glob
from os.path         import expandvars
from itertools       import count
from itertools       import repeat
from typing          import Callable
from typing          import Iterator
from typing          import Mapping
from typing          import Generator
from typing          import List
from typing          import Dict
from typing          import Tuple
from typing          import Union
from typing          import Any
from typing          import Optional

import tables as tb
import numpy  as np
import pandas as pd
import warnings
import math
import os

from .. dataflow                  import                  dataflow as  fl
from .. dataflow.dataflow         import                      sink
from .. dataflow.dataflow         import                      pipe
from .. evm    .ic_containers     import                SensorData
from .. evm    .event_model       import                   KrEvent
from .. evm    .event_model       import                       Hit
from .. evm    .event_model       import                   Cluster
from .. evm    .event_model       import             HitCollection
from .. evm    .event_model       import                    MCInfo
from .. core                      import           system_of_units as units
from .. core   .exceptions        import                XYRecoFail
from .. core   .exceptions        import           MCEventNotFound
from .. core   .exceptions        import              NoInputFiles
from .. core   .exceptions        import              NoOutputFile
from .. core   .exceptions        import InvalidInputFileStructure
from .. core   .exceptions        import          SensorIDMismatch
from .. core   .configure         import          event_range_help
from .. core   .configure         import         check_annotations
from .. core   .random_sampling   import              NoiseSampler
from .. detsim                    import          buffer_functions as  bf
from .. detsim                    import          sensor_functions as  sf
from .. detsim .sensor_utils      import             trigger_times
from .. evm    .pmaps             import                      PMap
from .. calib                     import           calib_functions as  cf
from .. calib                     import   calib_sensors_functions as csf
from .. reco                      import            peak_functions as pkf
from .. reco                      import           pmaps_functions as pmf
from .. reco                      import            hits_functions as hif
from .. reco                      import             wfm_functions as wfm
from .. reco                      import         paolina_functions as plf
from .. reco   .corrections       import                 read_maps
from .. reco   .corrections       import      apply_all_correction
from .. reco   .corrections       import     get_df_to_z_converter
from .. reco   .xy_algorithms     import                    corona
from .. reco   .xy_algorithms     import                barycenter
from .. filters.s1s2_filter       import               S12Selector
from .. filters.s1s2_filter       import         S12SelectorOutput
from .. filters.s1s2_filter       import               pmap_filter
from .. database                  import                   load_db
from .. sierpe                    import                       blr
from .. io                        import                 mcinfo_io
from .. io     .pmaps_io          import                load_pmaps
from .. io     .hits_io           import              hits_from_df
from .. io     .dst_io            import                  load_dst
from .. io     .event_filter_io   import       event_filter_writer
from .. io     .pmaps_io          import               pmap_writer
from .. io     .rwf_io            import             buffer_writer
from .. io     .mcinfo_io         import            load_mchits_df
from .. io     .mcinfo_io         import          load_mcstringmap
from .. io     .mcinfo_io         import         is_oldformat_file
from .. io     .dst_io            import                 df_writer
from .. types  .ic_types          import                  NoneType
from .. types  .ic_types          import                        xy
from .. types  .ic_types          import                        NN
from .. types  .ic_types          import                       NNN
from .. types  .ic_types          import                    minmax
from .. types  .ic_types          import        types_dict_summary
from .. types  .ic_types          import         types_dict_tracks
from .. types  .symbols           import                    WfType
from .. types  .symbols           import               RebinMethod
from .. types  .symbols           import                SiPMCharge
from .. types  .symbols           import                   BlsMode
from .. types  .symbols           import             SiPMThreshold
from .. types  .symbols           import                EventRange
from .. types  .symbols           import                 HitEnergy
from .. types  .symbols           import                    XYReco
from .. types  .symbols           import              NormStrategy



def city(city_function):
    @wraps(city_function)
    def proxy(**kwds):
        conf = Namespace(**kwds)

        # TODO: remove these in the config parser itself, before
        # they ever gets here
        if hasattr(conf, 'config_file'):       del conf.config_file
        # TODO: these will disappear once hierarchical config files
        # are removed
        if hasattr(conf, 'print_config_only'): del conf.print_config_only
        if hasattr(conf, 'hide_config'):       del conf.hide_config
        if hasattr(conf, 'no_overrides'):      del conf.no_overrides
        if hasattr(conf, 'no_files'):          del conf.no_files
        if hasattr(conf, 'full_files'):        del conf.full_files

        # TODO: we have decided to remove verbosity.
        # Needs to be removed form config parser
        if hasattr(conf, 'verbosity'):         del conf.verbosity

        # TODO Check raw_data_type in parameters for RawCity

        if 'files_in' not in kwds: raise NoInputFiles
        if 'file_out' not in kwds: raise NoOutputFile

        # always a sequence, so we can generalize the code below
        if isinstance(conf.files_in, str):
            conf.files_in = [conf.files_in]

        input_files = []
        for pattern in map(expandvars, conf.files_in):
            globbed_files = glob(pattern)
            if len(globbed_files) == 0:
                raise FileNotFoundError(f"Input pattern {pattern} did not match any files.")
            input_files.extend(globbed_files)

        if len(set(input_files)) != len(input_files):
            warnings.warn("files_in contains repeated values. Ignoring duplicate files.", UserWarning)
            input_files = [f for i, f in enumerate(input_files) if f not in input_files[:i]]

        conf.files_in = input_files
        conf.file_out = expandvars(conf.file_out)

        conf.event_range  = event_range(conf)
        # TODO There were deamons! self.daemons = tuple(map(summon_daemon, kwds.get('daemons', [])))

        args   = vars(conf)
        result = check_annotations(city_function)(**args)
        if os.path.exists(conf.file_out):
            write_city_configuration(conf.file_out, city_function.__name__, args)
            copy_cities_configuration(conf.files_in[0], conf.file_out)
            index_tables(conf.file_out)
        return result
    return proxy


@check_annotations
def create_timestamp(rate: float) -> float:
    """
    Get rate value safely: It raises warning if rate <= 0 and
    it sets a physical rate value in Hz.

    Parameters
    ----------
    rate : float
           Value of the rate in Hz.

    Returns
    -------
    Function to calculate timestamp for the given rate with
    event_number as parameter.
    """

    if rate == 0:
        warnings.warn("Zero rate is unphysical, using default "
                      "rate = 0.5 Hz instead", stacklevel=2)
        rate = 0.5 * units.hertz
    elif rate < 0:
        warnings.warn(f"Negative rate is unphysical, using "
                      f"rate = {abs(rate) / units.hertz} Hz instead",
                      stacklevel=2)
        rate = abs(rate)

    def create_timestamp_(event_number: Union[int, float]) -> float:
        """
        Calculates timestamp for a given Event Number and Rate.

        Parameters
        ----------
        event_number : Union[int, float]
                       ID value of the current event.

        Returns
        -------
        Calculated timestamp : float
        """

        period = 1. / rate
        timestamp = abs(event_number * period) + np.random.uniform(0, period)
        return timestamp

    return create_timestamp_


def index_tables(file_out):
    """
    -finds all tables in output_file
    -checks if any columns in the tables have been marked to be indexed by writers
    -indexes those columns
    """
    with tb.open_file(file_out, 'r+') as h5out:
        for table in h5out.walk_nodes(classname='Table'):        # Walk over all tables in h5out
            if 'columns_to_index' not in table.attrs:  continue  # Check for columns to index
            for colname in table.attrs.columns_to_index:         # Index those columns
                table.colinstances[colname].create_index()


def dict_to_string(arg : dict,
                   parent_key : str= ""):
    '''
    A function that recusively flattens nested dictionaries and converts the values to strings.

    Each nested key is combined with its parent keys and associated with its respective value 
    as a string.

    Parameters
    ----------
    arg        : the dictionary to flatten which can contain nested dictionaries
    parent_key : the string containing the prefix to use for keys in the flattened dictionary (default "").

    Returns
    -------
    flat_dict : flattened dictionary containing no nested dictionaries and where all values are strings
    '''
    flat_dict = {}
    for k, v in arg.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            flat_dict.update(dict_to_string(v, new_key))
        else:
            flat_dict[new_key] = str(v)
    return flat_dict


def write_city_configuration( filename : str
                            , city_name: str
                            , args     : dict):
    args = dict_to_string(args)

    with tb.open_file(filename, "a") as file:
        df = (pd.Series(args)
                .to_frame()
                .reset_index()
                .rename(columns={"index": "variable", 0: "value"}))
        df_writer(file, df, "config", city_name, f"configuration for {city_name}", str_col_length=300)


def copy_cities_configuration( file_in : str, file_out : str):
    with tb.open_file(file_in, "r") as fin:
        if "config" not in fin.root:
            warnings.warn("Input file does not contain /config group", UserWarning)
            return

        with tb.open_file(file_out, "a") as fout:
            if "config" not in fout.root:
                fout.create_group(fout.root, "config")
            for table in fin.root.config:
                fin.copy_node(table, fout.root.config, recursive=True)


def _check_invalid_event_range_spec(er):
    return (len(er) not in (1, 2)                   or
            (len(er) == 2 and EventRange.all in er) or
            er[0] is EventRange.last                )


def event_range(conf):
    # event_range not specified
    if not hasattr(conf, 'event_range')           : return None, 1
    er = conf.event_range

    if not isinstance(er, Sequence): er = (er,)
    if _check_invalid_event_range_spec(er):
        message = "Invalid spec for event range. Only the following are accepted:\n" + event_range_help
        raise ValueError(message)

    if   len(er) == 1 and er[0] is EventRange.all : return (None,)
    elif len(er) == 2 and er[1] is EventRange.last: return (er[0], None)
    else                                          : return er


def print_every(N):
    counter = count()
    return fl.branch(fl.map  (lambda _: next(counter), args="event_number", out="index"),
                     fl.slice(None, None, N),
                     fl.sink (lambda data: print(f"events processed: {data['index']}, event number: {data['event_number']}")))


def print_every_alternative_implementation(N):
    @fl.coroutine
    def print_every_loop(target):
        with fl.closing(target):
            for i in count():
                data = yield
                if not i % N:
                    print(f"events processed: {i}, event number: {data['event_number']}")
                target.send(data)
    return print_every_loop


def get_actual_sipm_thr(thr_sipm_type, thr_sipm, detector_db, run_number):
    if   thr_sipm_type is SiPMThreshold.common:
        # In this case, the threshold is a value in pes
        sipm_thr = thr_sipm

    elif thr_sipm_type is SiPMThreshold.individual:
        # In this case, the threshold is a percentual value
        noise_sampler = NoiseSampler(detector_db, run_number)
        sipm_thr      = noise_sampler.compute_thresholds(thr_sipm)

    else:
        raise ValueError(f"Unrecognized thr type: {thr_sipm_type}. "
                          "Only valid options are `common` and `individual`")

    return sipm_thr


def collect():
    """Return a future/sink pair for collecting streams into a list."""
    def append(l,e):
        l.append(e)
        return l
    return fl.reduce(append, initial=[])()


@check_annotations
def copy_mc_info(files_in     : List[str],
                 h5out        : tb.File  ,
                 event_numbers: List[int],
                 db_file      :      str ,
                 run_number   :      int ) -> None:
    """
    Copy to an output file the MC info of a list of selected events.

    Parameters
    ----------
    files_in     : List of strings
                   Name of the input files.
    h5out        : tables.File
                   The output h5 file.
    event_numbers: List[int]
                   List of event numbers for which the MC info is copied
                   to the output file.
    db_file      : str
                   Name of database to be used where necessary to
                   read the MC info (used for pre-2020 format files)
    run_number   : int
                   Run number for database (used for pre-2020 format files)
    """

    writer = mcinfo_io.mc_writer(h5out)

    copied_events   = []
    for f in files_in:
        if mcinfo_io.check_mc_present(f):
            evt_copied  = mcinfo_io.safe_copy_nexus_eventmap(h5out        ,
                                                             event_numbers,
                                                             f            )
            mcinfo_io.copy_mc_info(f, writer, evt_copied['nexus_evt'],
                                   db_file, run_number)
            copied_events.extend(evt_copied['evt_number'])
        else:
            warnings.warn('File does not contain MC tables.\
             Use positve run numbers for data', UserWarning)
            continue
    if len(np.setdiff1d(event_numbers, copied_events)) != 0:
        raise MCEventNotFound('Some events not found in MC tables')


@check_annotations
def wf_binner(max_buffer: float) -> Callable:
    """
    Returns a function to be used to convert the raw
    input MC sensor info into data binned according to
    a set bin width, effectively
    padding with zeros inbetween the separate signals.

    Parameters
    ----------
    max_buffer : float
                 Maximum event time to be considered in nanoseconds
    """
    def bin_sensors(sensors  : pd.DataFrame,
                    bin_width: float       ,
                    t_min    : float       ,
                    t_max    : float       ) -> Tuple[np.ndarray, pd.Series]:
        return bf.bin_sensors(sensors, bin_width, t_min, t_max, max_buffer)
    return bin_sensors


@check_annotations
def signal_finder(buffer_len   : float,
                  bin_width    : float,
                  bin_threshold:   int) -> Callable:
    """
    Decides where there is signal-like
    charge according to the configuration
    and the PMT sum in order to give
    a useful position for buffer selection.
    Currently simple threshold on binned charge.

    Parameters
    ----------
    buffer_len    : float
                    Configured buffer length in mus
    bin_width     : float
                    Sampling width for sensors
    bin_threshold : int
                    PE threshold for selection
    """
    # The stand_off is the minumum number of samples
    # necessary between candidate triggers.
    stand_off = int(buffer_len / bin_width)
    def find_signal(wfs: pd.Series) -> List[int]:
        return bf.find_signal_start(wfs, bin_threshold, stand_off)
    return find_signal


# TODO: consider caching database
def deconv_pmt(dbfile, run_number, n_baseline,
               selection=None, pedestal_function=csf.means):
    DataPMT    = load_db.DataPMT(dbfile, run_number = run_number)
    pmt_active = np.nonzero(DataPMT.Active.values)[0].tolist() if selection is None else selection
    coeff_c    = DataPMT.coeff_c  .values.astype(np.double)
    coeff_blr  = DataPMT.coeff_blr.values.astype(np.double)

    def deconv_pmt(RWF):
        CWF = pedestal_function(RWF[:, :n_baseline]) - RWF
        return np.array(tuple(map(blr.deconvolve_signal, CWF[pmt_active],
                                  coeff_c              , coeff_blr      )))
    return deconv_pmt


def get_run_number(h5in):
    if   "runInfo" in h5in.root.Run: return h5in.root.Run.runInfo[0]['run_number']
    elif "RunInfo" in h5in.root.Run: return h5in.root.Run.RunInfo[0]['run_number']

    raise tb.exceptions.NoSuchNodeError(f"No node runInfo or RunInfo in file {h5in}")


def get_pmt_wfs(h5in, wf_type):
    if   wf_type is WfType.rwf : return h5in.root.RD.pmtrwf
    elif wf_type is WfType.mcrd: return h5in.root.   pmtrd
    else                       : raise  TypeError(f"Invalid WfType: {type(wf_type)}")

def get_sipm_wfs(h5in, wf_type):
    if   wf_type is WfType.rwf : return h5in.root.RD.sipmrwf
    elif wf_type is WfType.mcrd: return h5in.root.   sipmrd
    else                       : raise  TypeError(f"Invalid WfType: {type(wf_type)}")


def get_trigger_info(h5in):
    group            = h5in.root.Trigger if "Trigger" in h5in.root else ()
    trigger_type     = group.trigger if "trigger" in group else repeat(None)
    trigger_channels = group.events  if "events"  in group else repeat(None)
    return trigger_type, trigger_channels


def get_event_info(h5in):
    return h5in.root.Run.events


def get_number_of_active_pmts(detector_db, run_number):
    datapmt = load_db.DataPMT(detector_db, run_number)
    return np.count_nonzero(datapmt.Active.values.astype(bool))


def check_nonempty_indices(s1_indices, s2_indices):
    return s1_indices.size and s2_indices.size


def check_empty_pmap(pmap):
    return bool(pmap.s1s) or bool(pmap.s2s)


def length_of(iterable):
    if   isinstance(iterable, tb.table.Table  ): return iterable.nrows
    elif isinstance(iterable, tb.earray.EArray): return iterable.shape[0]
    elif isinstance(iterable, np.ndarray      ): return iterable.shape[0]
    elif isinstance(iterable, NoneType        ): return None
    elif isinstance(iterable, Iterator        ): return None
    elif isinstance(iterable, Sequence        ): return len(iterable)
    elif isinstance(iterable, Mapping         ): return len(iterable)
    else:
        raise TypeError(f"Cannot determine size of type {type(iterable)}")


def check_lengths(*iterables):
    lengths  = map(length_of, iterables)
    nonnones = filter(lambda x: x is not None, lengths)
    if np.any(np.diff(list(nonnones)) != 0):
        raise InvalidInputFileStructure("Input data tables have different sizes")


@check_annotations
def mcsensors_from_file(paths     : List[str],
                        db_file   :      str ,
                        run_number:      int ,
                        rate      :    float ) -> Generator:
    """
    Loads the nexus MC sensor information into
    a pandas DataFrame using the IC function
    load_mcsensor_response_df.
    Returns info event by event as a
    generator in the structure expected by
    the dataflow.

    paths      : List of strings
                 List of input file names to be read
    db_file    : string
                 Name of detector database to be used
    run_number : int
                 Run number for database
    rate       : float
                 Rate value in base unit (ns^-1) to generate timestamps
    """

    timestamp = create_timestamp(rate)

    pmt_ids  = load_db.DataPMT(db_file, run_number).SensorID

    for file_name in paths:
        sns_resp = mcinfo_io.load_mcsensor_response_df(file_name              ,
                                                       return_raw = False     ,
                                                       db_file    = db_file   ,
                                                       run_no     = run_number)

        if not is_oldformat_file(file_name):
            nexus_sns_pos = pd.read_hdf(file_name, 'MC/sns_positions')
            pmt_condition = nexus_sns_pos.sensor_name.str.casefold().str.contains('pmt')
            nexus_pmt_ids = nexus_sns_pos[pmt_condition].sensor_id

            if not nexus_pmt_ids.isin(pmt_ids).all():
                raise SensorIDMismatch('Some PMT IDs in nexus file do not appear in database')


        for evt in mcinfo_io.get_event_numbers_in_file(file_name):

            try:
                ## Assumes two types of sensor, all non pmt
                ## assumed to be sipms. NEW, NEXT100 and DEMOPP safe
                ## Flex with this structure too.
                pmt_indx  = sns_resp.loc[evt].index.isin(pmt_ids)
                pmt_resp  = sns_resp.loc[evt][ pmt_indx]
                sipm_resp = sns_resp.loc[evt][~pmt_indx]
            except KeyError:
                pmt_resp = sipm_resp = pd.DataFrame(columns=sns_resp.columns)

            yield dict(event_number = evt      ,
                       timestamp    = timestamp(evt),
                       pmt_resp     = pmt_resp ,
                       sipm_resp    = sipm_resp)


def wf_from_files(paths, wf_type):
    for path in paths:
        with tb.open_file(path, "r") as h5in:
            try:
                event_info  = get_event_info  (h5in)
                run_number  = get_run_number  (h5in)
                pmt_wfs     = get_pmt_wfs     (h5in, wf_type)
                sipm_wfs    = get_sipm_wfs    (h5in, wf_type)
                (trg_type ,
                 trg_chann) = get_trigger_info(h5in)
            except tb.exceptions.NoSuchNodeError:
                continue

            check_lengths(pmt_wfs, sipm_wfs, event_info, trg_type, trg_chann)

            for pmt, sipm, evtinfo, trtype, trchann in zip(pmt_wfs, sipm_wfs, event_info, trg_type, trg_chann):
                event_number, timestamp         = evtinfo.fetch_all_fields()
                if trtype  is not None: trtype  = trtype .fetch_all_fields()[0]

                yield dict(pmt=pmt, sipm=sipm, run_number=run_number,
                           event_number=event_number, timestamp=timestamp,
                           trigger_type=trtype, trigger_channels=trchann)


def pmap_from_files(paths):
    for path in paths:
        try:
            pmaps = load_pmaps(path, lazy=True)
        except tb.exceptions.NoSuchNodeError:
            continue

        with tb.open_file(path, "r") as h5in:
            try:
                run_number  = get_run_number(h5in)
                event_info  = get_event_info(h5in)
            except tb.exceptions.NoSuchNodeError:
                continue
            except IndexError:
                continue

            for evtinfo, (evt, pmap) in zip(event_info, pmaps):
                event_number, timestamp = evtinfo.fetch_all_fields()
                if event_number != evt:
                    raise InvalidInputFileStructure("Inconsistent data: event number mismatch")
                yield dict(pmap=pmap, run_number=run_number,
                           event_number=event_number, timestamp=timestamp)


@check_annotations
def hits_and_kdst_from_files( paths : List[str]
                            , group : str
                            , node  : str ) -> Iterator[Dict[str,Union[HitCollection, pd.DataFrame, MCInfo, int, float]]]:
    """
    Reader of hits files. For each event it produces a dictionary containing
    - hits        : a DataFrame
    - kdst        : a DataFrame
    - run_number  : int
    - event_number: int
    - timestamp   : float with time in milliseconds
    """
    for path in paths:
        try:
            hits_df = load_dst (path, group, node)
            kdst_df = load_dst (path, 'DST' , 'Events')
        except tb.exceptions.NoSuchNodeError:
            continue

        with tb.open_file(path, "r") as h5in:
            try:
                run_number  = get_run_number(h5in)
                event_info  = get_event_info(h5in)
            except (tb.exceptions.NoSuchNodeError, IndexError):
                continue

            check_lengths(event_info, kdst_df.event.unique())

            for evtinfo in event_info:
                event_number, timestamp = evtinfo.fetch_all_fields()
                this_event = lambda df: df.event == event_number
                hits = hits_df.loc[this_event]
                kdst = kdst_df.loc[this_event]
                if not len(hits):
                    warnings.warn(f"Event {event_number} does not contain hits", UserWarning)

                yield dict(hits         = hits,
                           kdst         = kdst,
                           run_number   = run_number,
                           event_number = event_number,
                           timestamp    = timestamp)


@check_annotations
def dst_from_files(paths: List[str], group: str, node:str) -> Iterator[Dict[str,Union[pd.DataFrame, int, np.ndarray]]]:
    """
    Reader for a generic dst.
    """
    for path in paths:
        try:
            df = load_dst(path, group, node)
            with tb.open_file(path, "r") as h5in:
                run_number  = get_run_number(h5in)
                evt_numbers = get_event_info(h5in).col("evt_number")
                timestamps  = get_event_info(h5in).col("timestamp")
        except (tb.exceptions.NoSuchNodeError, IndexError):
            continue

        yield dict( dst           = df
                  , run_number    = run_number
                  , event_numbers = evt_numbers
                  , timestamps    = timestamps
                  )


@check_annotations
def MC_hits_from_files(files_in : List[str], rate: float) -> Generator:
    timestamp = create_timestamp(rate)
    for filename in files_in:
        try:
            hits_df = load_mchits_df(filename)
        except tb.exceptions.NoSuchNodeError:
            continue

        l_type = hits_df.dtypes['label']
        map_df = load_mcstringmap(filename) if l_type == np.int32 else None

        for evt, hits in hits_df.groupby(level=0):
            yield dict(event_number = evt,
                       x            = hits.x     .values,
                       y            = hits.y     .values,
                       z            = hits.z     .values,
                       energy       = hits.energy.values,
                       time         = hits.time  .values,
                       label        = hits.label .values,
                       timestamp    = timestamp(evt),
                       name         = map_df.name .values if map_df is not None else "",
                       name_id      = map_df.index.values if map_df is not None else  0)


@check_annotations
def dhits_from_files(paths: List[str]) -> Iterator[Dict[str,Union[HitCollection, pd.DataFrame, MCInfo, int, float]]]:
    """Reader of the files, yields HitsCollection, pandas DataFrame with
    run_number, event_number and timestamp."""
    for path in paths:
        try:
            dhits_df = load_dst (path, 'DECO', 'Events')
        except tb.exceptions.NoSuchNodeError:
            continue

        kdst_df = load_dst(path, 'DST', 'Events', ignore_errors=True)

        with tb.open_file(path, "r") as h5in:
            try:
                run_number  = get_run_number(h5in)
            except (tb.exceptions.NoSuchNodeError, IndexError):
                continue

            event_info = load_dst(path, 'Run', 'events')
            for evt in dhits_df.event.unique():
                this_event = lambda df: df.event==evt
                timestamp = event_info[event_info.evt_number==evt].timestamp.values[0]
                dhits = hits_from_df(dhits_df.loc[this_event])
                kdst  =               kdst_df.loc[this_event] if isinstance(kdst_df, pd.DataFrame) \
                                                              else None
                ### It makes no sense to use the 'io.hits_io.hits_from_df' function here
                ### (as well as in 'hits_and_kdst_from_files') since the majority of the
                ### 'evm.Hit' parameters don't appear in the dst (particularly running
                ### Isaura) and must be set to -1. This proceure should be revisited and
                ### rethought in the near future, with the aim of changing the event model.
                yield dict(hits         = dhits[evt],
                           kdst         = kdst      ,
                           run_number   = run_number,
                           event_number = evt       ,
                           timestamp    = timestamp )


def sensor_data(path, wf_type):
    with tb.open_file(path, "r") as h5in:
        if   wf_type is WfType.rwf :   (pmt_wfs, sipm_wfs) = (h5in.root.RD .pmtrwf,   h5in.root.RD .sipmrwf)
        elif wf_type is WfType.mcrd:   (pmt_wfs, sipm_wfs) = (h5in.root.    pmtrd ,   h5in.root.    sipmrd )
        else                       :   raise TypeError(f"Invalid WfType: {type(wf_type)}")
        _, NPMT ,  PMTWL =  pmt_wfs.shape
        _, NSIPM, SIPMWL = sipm_wfs.shape
        return SensorData(NPMT=NPMT, PMTWL=PMTWL, NSIPM=NSIPM, SIPMWL=SIPMWL)

####### Transformers ########

def build_pmap(detector_db, run_number, pmt_samp_wid, sipm_samp_wid,
               s1_lmax, s1_lmin, s1_rebin_stride, s1_stride, s1_tmax, s1_tmin,
               s2_lmax, s2_lmin, s2_rebin_stride, s2_stride, s2_tmax, s2_tmin, thr_sipm_s2):
    s1_params = dict(time        = minmax(min = s1_tmin,
                                          max = s1_tmax),
                    length       = minmax(min = s1_lmin,
                                          max = s1_lmax),
                    stride       = s1_stride,
                    rebin_stride = s1_rebin_stride)

    s2_params = dict(time        = minmax(min = s2_tmin,
                                          max = s2_tmax),
                    length       = minmax(min = s2_lmin,
                                          max = s2_lmax),
                    stride       = s2_stride,
                    rebin_stride = s2_rebin_stride)

    datapmt = load_db.DataPMT(detector_db, run_number)
    pmt_ids = datapmt.SensorID[datapmt.Active.astype(bool)].values

    def build_pmap(ccwf, s1_indx, s2_indx, sipmzs): # -> PMap
        return pkf.get_pmap(ccwf, s1_indx, s2_indx, sipmzs,
                            s1_params, s2_params, thr_sipm_s2, pmt_ids,
                            pmt_samp_wid, sipm_samp_wid)

    return build_pmap


def calibrate_pmts(dbfile, run_number, n_maw, thr_maw):
    DataPMT    = load_db.DataPMT(dbfile, run_number = run_number)
    adc_to_pes = np.abs(DataPMT.adc_to_pes.values)
    adc_to_pes = adc_to_pes[adc_to_pes > 0]

    def calibrate_pmts(cwf):# -> CCwfs:
        return csf.calibrate_pmts(cwf,
                                  adc_to_pes = adc_to_pes,
                                  n_maw      = n_maw,
                                  thr_maw    = thr_maw)
    return calibrate_pmts


def calibrate_sipms(dbfile, run_number, thr_sipm):
    DataSiPM   = load_db.DataSiPM(dbfile, run_number)
    adc_to_pes = np.abs(DataSiPM.adc_to_pes.values)

    def calibrate_sipms(rwf):
        return csf.calibrate_sipms(rwf,
                                   adc_to_pes = adc_to_pes,
                                   thr        = thr_sipm,
                                   bls_mode   = BlsMode.mode)

    return calibrate_sipms


def calibrate_with_mean(dbfile, run_number):
    DataSiPM   = load_db.DataSiPM(dbfile, run_number)
    adc_to_pes = np.abs(DataSiPM.adc_to_pes.values)
    def calibrate_with_mean(wfs):
        return csf.subtract_baseline_and_calibrate(wfs, adc_to_pes)
    return calibrate_with_mean

def calibrate_with_maw(dbfile, run_number, n_maw_sipm):
    DataSiPM   = load_db.DataSiPM(dbfile, run_number)
    adc_to_pes = np.abs(DataSiPM.adc_to_pes.values)
    def calibrate_with_maw(wfs):
        return csf.subtract_baseline_maw_and_calibrate(wfs, adc_to_pes, n_maw_sipm)
    return calibrate_with_maw


def zero_suppress_wfs(thr_csum_s1, thr_csum_s2):
    def ccwfs_to_zs(ccwf_sum, ccwf_sum_maw):
        return (pkf.indices_and_wf_above_threshold(ccwf_sum_maw, thr_csum_s1).indices,
                pkf.indices_and_wf_above_threshold(ccwf_sum    , thr_csum_s2).indices)
    return ccwfs_to_zs


def compute_pe_resolution(rms, adc_to_pes):
    return np.divide(rms                              ,
                     adc_to_pes                       ,
                     out   = np.zeros_like(adc_to_pes),
                     where = adc_to_pes != 0          )


def simulate_sipm_response(detector, run_number, wf_length, noise_cut, filter_padding):
    datasipm      = load_db.DataSiPM (detector, run_number)
    baselines     = load_db.SiPMNoise(detector, run_number)[-1]
    noise_sampler = NoiseSampler(detector, run_number, wf_length, True)

    adc_to_pes    = datasipm.adc_to_pes.values
    thresholds    = noise_cut * adc_to_pes + baselines
    single_pe_rms = datasipm.Sigma.values.astype(np.double)
    pe_resolution = compute_pe_resolution(single_pe_rms, adc_to_pes)

    def simulate_sipm_response(sipmrd):
        wfs = sf.simulate_sipm_response(sipmrd, noise_sampler, adc_to_pes, pe_resolution)
        return wfm.noise_suppression(wfs, thresholds, filter_padding)
    return simulate_sipm_response


####### Filters ########

def peak_classifier(**params):
    """
    Applies S12Selector to PMaps, which labels each peak as valid or invalid
    using 4 criteria:
    width, height, energy (pmts integral) and number of SiPMs.
    """
    selector = S12Selector(**params)
    return partial(pmap_filter, selector)


def compute_xy_position(dbfile, run_number, algo, **reco_params):
    if algo is XYReco.corona:
        datasipm    = load_db.DataSiPM(dbfile, run_number)
        reco_params = dict(all_sipms = datasipm, **reco_params)
        algorithm   = corona
    else:
        algorithm   = barycenter

    def compute_xy_position(xys, qs):
        return algorithm(xys, qs, **reco_params)

    return compute_xy_position


def compute_z_and_dt(t_s2, t_s1, drift_v):
    dt  = t_s2 - np.array(t_s1)
    z   = dt * drift_v
    dt *= units.ns / units.mus
    return z, dt


def build_pointlike_event(dbfile, run_number, drift_v,
                          reco, charge_type):
    datasipm   = load_db.DataSiPM(dbfile, run_number)
    sipm_xs    = datasipm.X.values
    sipm_ys    = datasipm.Y.values
    sipm_xys   = np.stack((sipm_xs, sipm_ys), axis=1)

    sipm_noise = NoiseSampler(dbfile, run_number).signal_to_noise

    def build_pointlike_event(pmap, selector_output, event_number, timestamp):
        evt = KrEvent(event_number, timestamp * 1e-3)

        evt.nS1 = 0
        for passed, peak in zip(selector_output.s1_peaks, pmap.s1s):
            if not passed: continue

            evt.nS1 += 1
            evt.S1w.append(peak.width)
            evt.S1h.append(peak.height)
            evt.S1e.append(peak.total_energy)
            evt.S1t.append(peak.time_at_max_energy)

        evt.nS2 = 0

        for passed, peak in zip(selector_output.s2_peaks, pmap.s2s):
            if not passed: continue

            evt.nS2 += 1
            evt.S2w.append(peak.width / units.mus)
            evt.S2h.append(peak.height)
            evt.S2e.append(peak.total_energy)
            evt.S2t.append(peak.time_at_max_energy)

            xys = sipm_xys[peak.sipms.ids           ]
            qs  = peak.sipm_charge_array(sipm_noise, charge_type,
                                         single_point = True)
            try:
                clusters = reco(xys, qs)
            except XYRecoFail:
                c    = NNN()
                Z    = tuple(NN for _ in range(0, evt.nS1))
                DT   = tuple(NN for _ in range(0, evt.nS1))
                Zrms = NN
            else:
                c = clusters[0]
                Z, DT = compute_z_and_dt(evt.S2t[-1], evt.S1t, drift_v)
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
            evt.qmax .append(max(qs))

        return evt

    return build_pointlike_event


def get_s1_time(pmap, selector_output):
    # in order to compute z one needs to define one S1
    # for time reference. By default the filter will only
    # take events with exactly one s1. Otherwise, the
    # convention is to take the first peak in the S1 object
    # as reference.
    if np.any(selector_output.s1_peaks):
        first_s1 = np.where(selector_output.s1_peaks)[0][0]
        s1_t     = pmap.s1s[first_s1].time_at_max_energy
    else:
        first_s2 = np.where(selector_output.s2_peaks)[0][0]
        s1_t     = pmap.s2s[first_s2].times[0]

    return s1_t


def try_global_reco(reco, xys, qs):
    try              : cluster = reco(xys, qs)[0]
    except XYRecoFail: return xy.empty()
    else             : return xy(cluster.X, cluster.Y)


def sipm_positions(dbfile, run_number):
    datasipm = load_db.DataSiPM(dbfile, run_number)
    sipm_xs  = datasipm.X.values
    sipm_ys  = datasipm.Y.values
    sipm_xys = np.stack((sipm_xs, sipm_ys), axis=1)
    return sipm_xys


def hit_builder( detector_db : str
               , run_number  : int
               , drift_v     : float
               , rebin_method: RebinMethod
               , rebin_slices: Union[int, float]
               , global_reco : XYReco
               , slice_reco  : XYReco
               , charge_type : SiPMCharge
               ) -> Callable:
    """
    Builds hits from PMaps using a general clustering algorithm. For a given
    PMap, and the output of the peak-selector output does the following:
    - Filters out peaks rejected by the selector
    - Picks up the S1 (always the first one, if there are more, they are ignored)
    - Rebins each S2 according to `rebin_method` and `rebin_slices`
    - For each S2:
      - Compute the overall position of the signal according to `global_reco`
        (typically barycenter in XYZ)
      - For each (rebinned) slice of the S2:
        - Clusterize the SiPM responses according to `slice_reco`
          - Failing XY reconstructions (e.g. not enough SiPMs with signal)
            generate "empty" (a.k.a. NN) clusters
        - Assign each cluster the corresponding fraction of the energy in the
          slice

    Parameters
    ----------
    detector_db: str
      Detector database to use

    run_number: int
      Run number being processed

    drift_v: float
      Drift velocity in the data

    rebin_method: RebinMethod
      Which rebinning (resampling) algorithm to use

    rebin_slices: int or float
      Configuration option for `rebin_method`. It's interpretation depends on
      the method:
      If stride, `rebin_slices` represents the number of consecutive slices co
      merge into one.
      If threshold, `rebin_slices` represents the minimum charge a slice must
      have for it not to be rebinned.

    global_reco: Callable
      Reconstruction function to use for the event as a whole

    slice_reco: Callable
      Reconstruction function to use on each slice

    charge_type: SiPMCharge
      Interpretation of the SiPM charge.

    Returns
    -------
    build_hits: Callable
      A function that computes hits.
    """
    sipm_xys   = sipm_positions(detector_db, run_number)
    sipm_noise =   NoiseSampler(detector_db, run_number).signal_to_noise

    def build_hits( pmap           : PMap
                  , selector_output: S12SelectorOutput
                  , event_number   : int
                  , timestamp      : float
                  ) -> HitCollection:
        hitc = HitCollection(event_number, timestamp * 1e-3)
        s1_t = get_s1_time(pmap, selector_output)

        # here hits are computed for each peak and each slice.
        # In case of an exception, a hit is still created with a NN cluster.
        # (NN cluster is a cluster where the energy is an IC not number NN)
        # this allows to keep track of the energy associated to non reonstructed hits.
        for peak_no, (passed, peak) in enumerate(zip(selector_output.s2_peaks,
                                                     pmap.s2s)):
            if not passed: continue

            peak = pmf.rebin_peak(peak, rebin_method, rebin_slices)
            xys  = sipm_xys[peak.sipms.ids]
            qs   = peak.sipm_charge_array(sipm_noise, charge_type,
                                          single_point = True)

            xy_peak     = try_global_reco(global_reco, xys, qs)
            sipm_charge = peak.sipm_charge_array(sipm_noise        ,
                                                 charge_type       ,
                                                 single_point=False)

            slice_zs = (peak.times - s1_t) * units.ns * drift_v
            slice_es = peak.pmts.sum_over_sensors
            xys      = sipm_xys[peak.sipms.ids]

            for (z_slice, e_slice, sipm_qs) in zip(slice_zs, slice_es, sipm_charge):
                try:
                    clusters = slice_reco(xys, sipm_qs)
                    qs       = np.array([c.Q for c in clusters])
                    es       = hif.e_from_q(qs, e_slice)
                    for c, e in zip(clusters, es):
                        hit  = Hit(peak_no, c, z_slice, e, xy_peak)
                        hitc.hits.append(hit)
                except XYRecoFail:
                    hit = Hit(peak_no, Cluster.empty(), z_slice,
                              e_slice, xy_peak)
                    hitc.hits.append(hit)

        return hitc
    return build_hits


def sipms_as_hits( detector_db : str
                 , run_number  : int
                 , drift_v     : float
                 , rebin_method: RebinMethod
                 , rebin_slices: Union[int, float]
                 , q_thr       : float
                 , global_reco : Callable
                 , charge_type : SiPMCharge
                 ) -> Callable:
    """
    Builds hits from PMaps taking each SiPM as an individual hit. For a given
    PMap, and the output of the peak-selector output does the following:
    - Filters out peaks rejected by the selector
    - Picks up the S1 (always the first one, if there are more, they are ignored)
    - Rebins each S2 according to `rebin_method` and `rebin_slices`
    - For each S2:
      - Compute the overall position of the signal according to `global_reco`
        (typically barycenter in XYZ)
      - For each (rebinned) slice of the S2:
        - Filters out SiPMs below `q_thr`
        - If no hits survive, the entire slice is summarized in an "empty"
          (a.k.a. NN) hit
        - Assigns each SiPM (now hit) the corresponding fraction of the energy
          in the slice. NN-hits carry the full slice energy.

    Parameters
    ----------
    detector_db: str
      Detector database to use

    run_number: int
      Run number being processed

    drift_v: float
      Drift velocity in the data

    rebin_method: RebinMethod
      Which rebinning (resampling) algorithm to use

    rebin_slices: int or float
      Configuration option for `rebin_method`. It's interpretation depends on
      the method:
      If stride, `rebin_slices` represents the number of consecutive slices co
      merge into one.
      If threshold, `rebin_slices` represents the minimum charge a slice must
      have for it not to be rebinned.

    q_thr: float
      Charge threshold applied to each hit

    global_reco: Callable
      Reconstruction function to use for the event as a whole

    slice_reco: Callable
      Reconstruction function to use on each slice

    charge_type: SiPMCharge
      Interpretation of the SiPM charge.

    Returns
    -------
    build_hits: Callable
      A function that computes hits.
    """
    sipm_xys   = sipm_positions(detector_db, run_number)
    sipm_noise =   NoiseSampler(detector_db, run_number).signal_to_noise

    def build_hits( pmap           : PMap
                  , selector_output: S12SelectorOutput
                  , event_number   : int
                  , timestamp      : float
                  ) -> pd.DataFrame:
        s1_t = get_s1_time(pmap, selector_output)
        hits = []
        for peak_no, (passed, peak) in enumerate(zip(selector_output.s2_peaks,
                                                     pmap.s2s)):
            if not passed: continue

            peak = pmf.rebin_peak(peak, rebin_method, rebin_slices)
            xys  = sipm_xys[peak.sipms.ids]
            qs   = peak.sipm_charge_array(sipm_noise, charge_type,
                                          single_point = True)

            xy_peak = try_global_reco(global_reco, xys, qs)

            sipm_charge = peak.sipm_charge_array(sipm_noise        ,
                                                 charge_type       ,
                                                 single_point=False)

            slice_zs = (peak.times - s1_t) * units.ns * drift_v
            slice_es = peak.pmts.sum_over_sensors
            xys      = sipm_xys[peak.sipms.ids]

            for (slice_z, slice_e, sipm_qs) in zip(slice_zs, slice_es, sipm_charge):
                sipm_xs, sipm_ys, sipm_qs, sipm_es = hif.sipms_above_threshold(xys, sipm_qs, q_thr, slice_e)
                sipm_hs  = dict( event    = event_number
                               , time     = timestamp * 1e-3
                               , npeak    = peak_no
                               , Xpeak    = xy_peak[0]
                               , Ypeak    = xy_peak[1]
                               , nsipm    = 1
                               , X        = sipm_xs
                               , Y        = sipm_ys
                               , Xrms     = 0.
                               , Yrms     = 0.
                               , Z        = slice_z
                               , Q        = sipm_qs
                               , E        = sipm_es
                               , Qc       = -1.
                               , Ec       = -1.
                               , track_id = -1
                               , Ep       = -1.)
                hits.append(pd.DataFrame(sipm_hs))

        hits = pd.concat(hits, ignore_index=True)
        hits = hits.astype(dict(npeak=np.uint16, nsipm=np.uint16))
        return hits

    return build_hits


def waveform_binner(bins):
    def bin_waveforms(wfs):
        return cf.bin_waveforms(wfs, bins)
    return bin_waveforms


def waveform_integrator(limits):
    def integrate_wfs(wfs):
        return cf.spaced_integrals(wfs, limits)[:, ::2]
    return integrate_wfs


# Compound components
def compute_and_write_pmaps(detector_db, run_number, pmt_samp_wid, sipm_samp_wid,
                  s1_lmax, s1_lmin, s1_rebin_stride, s1_stride, s1_tmax, s1_tmin,
                  s2_lmax, s2_lmin, s2_rebin_stride, s2_stride, s2_tmax, s2_tmin, thr_sipm_s2,
                  h5out, sipm_rwf_to_cal=None):

    # Filter events without signal over threshold
    indices_pass    = fl.map(check_nonempty_indices,
                             args = ("s1_indices", "s2_indices"),
                             out = "indices_pass")
    empty_indices   = fl.count_filter(bool, args = "indices_pass")

    # Build the PMap
    compute_pmap     = fl.map(build_pmap(detector_db, run_number, pmt_samp_wid, sipm_samp_wid,
                                         s1_lmax, s1_lmin, s1_rebin_stride, s1_stride, s1_tmax, s1_tmin,
                                         s2_lmax, s2_lmin, s2_rebin_stride, s2_stride, s2_tmax, s2_tmin, thr_sipm_s2),
                              args = ("ccwfs", "s1_indices", "s2_indices", "sipm"),
                              out  = "pmap")

    # Filter events with zero peaks
    pmaps_pass      = fl.map(check_empty_pmap, args = "pmap", out = "pmaps_pass")
    empty_pmaps     = fl.count_filter(bool, args = "pmaps_pass")

    # Define writers...
    write_pmap_         = pmap_writer        (h5out,              )
    write_indx_filter_  = event_filter_writer(h5out, "s12_indices")
    write_pmap_filter_  = event_filter_writer(h5out, "empty_pmap" )

    # ... and make them sinks
    write_pmap         = sink(write_pmap_        , args=(        "pmap", "event_number"))
    write_indx_filter  = sink(write_indx_filter_ , args=("event_number", "indices_pass"))
    write_pmap_filter  = sink(write_pmap_filter_ , args=("event_number",   "pmaps_pass"))

    fn_list = (indices_pass,
               fl.branch(write_indx_filter),
               empty_indices.filter,
               sipm_rwf_to_cal,
               compute_pmap,
               pmaps_pass,
               fl.branch(write_pmap_filter),
               empty_pmaps.filter,
               fl.branch(write_pmap))

    # Filter out simp_rwf_to_cal if it is not set
    compute_pmaps = pipe(*filter(None, fn_list))

    return compute_pmaps, empty_indices, empty_pmaps


@check_annotations
def check_max_time(max_time: float, buffer_length: float) -> Union[int, float]:
    """
    `max_time` must be greater than `buffer_length`. If not, raise warning
        and set `max_time` == `buffer_length`.

    :param max_time: Maximal length of the event that will be taken into
        account starting from the first detected signal, all signals after
        that are simply lost.
    :param buffer_length: Length of buffers.
    :return: `max_time` if `max_time` >= `buffer_length`, else `buffer_length`.
    """
    if max_time % units.mus:
        message = "Invalid value for max_time, it has to be a multiple of 1 mus"
        raise ValueError(message)

    if max_time < buffer_length:
        warnings.warn("`max_time` shorter than `buffer_length`, "
                      "setting `max_time` to `buffer_length`",
                      stacklevel=2)
        return buffer_length
    else:
        return max_time


@check_annotations
def calculate_and_save_buffers(buffer_length    : float        ,
                               max_time         : float        ,
                               pre_trigger      : float        ,
                               pmt_wid          : float        ,
                               sipm_wid         : float        ,
                               trigger_threshold: int          ,
                               h5out            : tb.File      ,
                               run_number       : int          ,
                               npmt             : int          ,
                               nsipm            : int          ,
                               nsamp_pmt        : int          ,
                               nsamp_sipm       : int          ,
                               order_sensors    : Union[NoneType, Callable]):
    find_signal       = fl.map(signal_finder(buffer_length, pmt_wid,
                                             trigger_threshold     ),
                               args = "pmt_bin_wfs"                 ,
                               out  = "pulses"                      )

    filter_events_signal = fl.map(lambda x: len(x) > 0,
                                  args= 'pulses',
                                  out = 'passed_signal')
    events_passed_signal = fl.count_filter(bool, args='passed_signal')
    write_signal_filter  = fl.sink(event_filter_writer(h5out, "signal"),
                                   args=('event_number', 'passed_signal'))

    event_times       = fl.map(trigger_times                             ,
                               args = ("pulses", "timestamp", "pmt_bins"),
                               out  = "evt_times"                        )

    calculate_buffers = fl.map(bf.buffer_calculator(buffer_length, pre_trigger,
                                                    pmt_wid     ,    sipm_wid),
                               args = ("pulses",
                                       "pmt_bins" ,  "pmt_bin_wfs",
                                       "sipm_bins", "sipm_bin_wfs")        ,
                               out  = "buffers"                            )

    saved_buffers = "buffers" if order_sensors is None else "ordered_buffers"
    max_subevt    =  math.ceil(max_time / buffer_length)
    buffer_writer_    = sink(buffer_writer( h5out
                                          , run_number = run_number
                                          , n_sens_eng = npmt
                                          , n_sens_trk = nsipm
                                          , length_eng = nsamp_pmt
                                          , length_trk = nsamp_sipm
                                          , max_subevt = max_subevt),
                             args = ("event_number", "evt_times"  ,
                                     saved_buffers                ))

    find_signal_and_write_buffers = ( find_signal
                                    , filter_events_signal
                                    , fl.branch(write_signal_filter)
                                    , events_passed_signal.filter
                                    , event_times
                                    , calculate_buffers
                                    , order_sensors
                                    , fl.branch(buffer_writer_))

    # Filter out order_sensors if it is not set
    buffer_definition = pipe(*filter(None, find_signal_and_write_buffers))
    return buffer_definition


@check_annotations
def Efield_copier(energy_type: HitEnergy):
    def copy_Efield(hitc : HitCollection) -> HitCollection:
        mod_hits = []
        for hit in hitc.hits:
            hit = Hit(hit.npeak,
                      Cluster(hit.Q, xy(hit.X, hit.Y), hit.var, hit.nsipm),
                      hit.Z,
                      hit.E,
                      xy(hit.Xpeak, hit.Ypeak),
                      s2_energy_c=getattr(hit, energy_type.value),
                      Ep=getattr(hit, energy_type.value))
            mod_hits.append(hit)
        mod_hitc = HitCollection(hitc.event, hitc.time, hits=mod_hits)
        return mod_hitc
    return copy_Efield


@check_annotations
def make_event_summary(event_number  : int          ,
                       topology_info : pd.DataFrame ,
                       paolina_hits  : HitCollection,
                       out_of_map    : bool
                       ) -> pd.DataFrame:
    """
    For a given event number, timestamp, topology info dataframe, paolina hits and kdst information returns a
    dataframe with the whole event summary.

    Parameters
    ----------
    event_number  : int
    topology_info : DataFrame
        Dataframe containing track information,
        output of track_blob_info_creator_extractor
    paolina_hits  : HitCollection
        Hits table passed through paolina functions,
        output of track_blob_info_creator_extractor
    kdst          : DataFrame
        Kdst information read from penthesilea output


    Returns
    ----------
    DataFrame containing relevant per event information.
    """
    es = pd.DataFrame(columns=list(types_dict_summary.keys()))
    if len(paolina_hits.hits) == 0: return es

    ntrks = len(topology_info.index)
    nhits = len(paolina_hits.hits)

    S2ec = sum(h.Ec for h in paolina_hits.hits)
    S2qc = -1 #not implemented yet

    pos   = [h.pos for h in paolina_hits.hits]
    x, y, z = map(np.array, zip(*pos))
    r = np.sqrt(x**2 + y**2)

    e     = [h.Ec  for h in paolina_hits.hits]
    ave_pos = np.average(pos, weights=e, axis=0)
    ave_r   = np.average(r  , weights=e, axis=0)


    list_of_vars  = [event_number, S2ec, S2qc, ntrks, nhits,
                     *ave_pos, ave_r,
                     min(x), min(y), min(z), min(r), max(x), max(y), max(z), max(r),
                     out_of_map]

    es.loc[0] = list_of_vars
    #change dtype of columns to match type of variables
    es = es.apply(lambda x : x.astype(types_dict_summary[x.name]))
    return es



def track_writer(h5out):
    """
    For a given open table returns a writer for topology info dataframe
    """
    def write_tracks(df):
        return df_writer(h5out              = h5out              ,
                         df                 = df                 ,
                         group_name         = 'Tracking'         ,
                         table_name         = 'Tracks'           ,
                         descriptive_string = 'Track information',
                         columns_to_index   = ['event']          )
    return write_tracks


def summary_writer(h5out):
    """
    For a given open table returns a writer for summary info dataframe
    """
    def write_summary(df):
        return df_writer(h5out              = h5out                      ,
                         df                 = df                         ,
                         group_name         = 'Summary'                  ,
                         table_name         = 'Events'                   ,
                         descriptive_string = 'Event summary information',
                         columns_to_index   = ['event']                  )
    return write_summary


@check_annotations
def track_blob_info_creator_extractor(vox_size         : Tuple[float, float, float],
                                      strict_vox_size  : bool                      ,
                                      energy_threshold : float                     ,
                                      min_voxels       : int                       ,
                                      blob_radius      : float                     ,
                                      max_num_hits     : int
                                     ) -> Callable:
    """
    For a given paolina parameters returns a function that extract tracks / blob information from a HitCollection.

    Parameters
    ----------
    vox_size         : [float, float, float]
        (maximum) size of voxels for track reconstruction
    strict_vox_size  : bool
        if False allows per event adaptive voxel size,
        smaller of equal thatn vox_size
    energy_threshold : float
        if energy of end-point voxel is smaller
        the voxel will be dropped and energy redistributed to the neighbours
    min_voxels       : int
        after min_voxel number of voxels is reached no dropping will happen.
    blob_radius      : float
        radius of blob

    Returns
    ----------
    A function that from a given HitCollection returns a pandas DataFrame with per track information.
    """
    def create_extract_track_blob_info(hitc):
        df = pd.DataFrame(columns=list(types_dict_tracks.keys()))
        if len(hitc.hits) > max_num_hits:
            return df, hitc, True
        #track_hits is a new Hitcollection object that contains hits belonging to tracks, and hits that couldnt be corrected
        track_hitc = HitCollection(hitc.event, hitc.time)
        out_of_map = np.any(np.isnan([h.Ep for h in hitc.hits]))
        if out_of_map:
            #add nan hits to track_hits, the track_id will be -1
            track_hitc.hits.extend  ([h for h in hitc.hits if np.isnan   (h.Ep)])
            hits_without_nan       = [h for h in hitc.hits if np.isfinite(h.Ep)]
            #create new Hitcollection object but keep the name hitc
            hitc      = HitCollection(hitc.event, hitc.time)
            hitc.hits = hits_without_nan

        hit_energies = np.array([getattr(h, HitEnergy.Ep.value) for h in hitc.hits])

        if len(hitc.hits) > 0 and (hit_energies>0).any():
            voxels           = plf.voxelize_hits(hitc.hits, vox_size, strict_vox_size, HitEnergy.Ep)
            (    mod_voxels,
             dropped_voxels) = plf.drop_end_point_voxels(voxels, energy_threshold, min_voxels)

            for v in dropped_voxels:
                track_hitc.hits.extend(v.hits)

            tracks = plf.make_track_graphs(mod_voxels)
            tracks = sorted(tracks, key=plf.get_track_energy, reverse=True)

            vox_size_x = voxels[0].size[0]
            vox_size_y = voxels[0].size[1]
            vox_size_z = voxels[0].size[2]
            del(voxels)

            track_hits = []
            for c, t in enumerate(tracks, 0):
                tID = c
                energy = plf.get_track_energy(t)
                numb_of_hits   = len([h for vox in t.nodes() for h in vox.hits])
                numb_of_voxels = len(t.nodes())
                numb_of_tracks = len(tracks   )
                pos   = [h.pos for v in t.nodes() for h in v.hits]
                x, y, z = map(np.array, zip(*pos))
                r = np.sqrt(x**2 + y**2)

                e     = [h.Ep for v in t.nodes() for h in v.hits]
                ave_pos = np.average(pos, weights=e, axis=0)
                ave_r   = np.average(r  , weights=e, axis=0)
                distances = plf.shortest_paths(t)
                extr1, extr2, length = plf.find_extrema_and_length(distances)
                extr1_pos = extr1.XYZ
                extr2_pos = extr2.XYZ

                e_blob1, e_blob2, hits_blob1, hits_blob2, blob_pos1, blob_pos2 = plf.blob_energies_hits_and_centres(t, blob_radius)

                overlap = float(sum(h.Ep for h in set(hits_blob1).intersection(set(hits_blob2))))
                list_of_vars = [hitc.event, tID, energy, length, numb_of_voxels,
                                numb_of_hits, numb_of_tracks,
                                min(x), min(y), min(z), min(r), max(x), max(y), max(z), max(r),
                                *ave_pos, ave_r, *extr1_pos,
                                *extr2_pos, *blob_pos1, *blob_pos2,
                                e_blob1, e_blob2, overlap,
                                vox_size_x, vox_size_y, vox_size_z]

                df.loc[c] = list_of_vars

                for vox in t.nodes():
                    for hit in vox.hits:
                        hit.track_id = tID
                        track_hits.append(hit)

            #change dtype of columns to match type of variables
            df = df.apply(lambda x : x.astype(types_dict_tracks[x.name]))
            track_hitc.hits.extend(track_hits)
        return df, track_hitc, out_of_map

    return create_extract_track_blob_info


def sort_hits(hitc):
    # sort hits in z, then in x, then in y
    sorted_hits = sorted(hitc.hits, key=lambda h: (h.Z, h.X, h.Y))
    return HitCollection(hitc.event, hitc.time, sorted_hits)


def hitc_to_df(hitc: HitCollection):
    hits = []
    for hit in hitc.hits:
        hits.append(pd.DataFrame(dict( event    = hitc.event
                                     , time     = hitc.time
                                     , npeak    = hit .npeak
                                     , Xpeak    = hit .Xpeak
                                     , Ypeak    = hit .Ypeak
                                     , nsipm    = hit .nsipm
                                     , X        = hit .X
                                     , Y        = hit .Y
                                     , Xrms     = hit .Xrms
                                     , Yrms     = hit .Yrms
                                     , Z        = hit .Z
                                     , Q        = hit .Q
                                     , E        = hit .E
                                     , Qc       = hit .Qc
                                     , Ec       = hit .Ec
                                     , track_id = hit .track_id
                                     , Ep       = hit .Ep), index=[0]))
    df = pd.concat(hits, ignore_index=True)
    df = df.astype(dict(event=np.int64, npeak=np.uint16, nsipm=np.uint16, Qc=np.float64, Ec=np.float64, Ep=np.float64))
    return df


def compute_and_write_tracks_info(paolina_params, h5out,
                                  hit_type, filter_hits_table_name,
                                  hits_writer):

    filter_events_nohits = fl.map(lambda x : len(x.hits) > 0,
                                      args = 'hits',
                                      out  = 'hits_passed')
    hits_passed          = fl.count_filter(bool, args="hits_passed")


    copy_Efield          = fl.map(Efield_copier(hit_type),
                                            args = 'hits',
                                            out  = 'Ep_hits')

    # Create tracks and compute topology-related information
    create_extract_track_blob_info = fl.map(track_blob_info_creator_extractor(**paolina_params),
                                            args = 'Ep_hits',
                                            out  = ('topology_info', 'paolina_hits', 'out_of_map'))

    sort_hits_ = fl.map(sort_hits, item="paolina_hits")

    # Filter empty topology events
    filter_events_topology         = fl.map(lambda x : len(x) > 0,
                                            args = 'topology_info',
                                            out  = 'topology_passed')
    events_passed_topology         = fl.count_filter(bool, args="topology_passed")

    # Create table with summary information
    make_final_summary             = fl.map(make_event_summary,
                                            args = ('event_number', 'topology_info', 'paolina_hits', 'out_of_map'),
                                            out  = 'event_info')

    # Define writers and make them sinks
    write_tracks          = fl.sink(   track_writer     (h5out=h5out)             , args="topology_info"      )
    write_summary         = fl.sink( summary_writer     (h5out=h5out)             , args="event_info"         )
    write_topology_filter = fl.sink( event_filter_writer(h5out, "topology_select"), args=("event_number", "topology_passed"    ))

    write_no_hits_filter  = fl.sink( event_filter_writer(h5out, filter_hits_table_name), args=("event_number", "hits_passed"))


    make_and_write_summary  = make_final_summary, write_summary
    select_and_write_tracks = events_passed_topology.filter, write_tracks

    to_hits_df = fl.map(hitc_to_df)
    write_hits = ("paolina_hits", to_hits_df, fl.sink(hits_writer))

    fork_pipes = filter(None, ( make_and_write_summary
                              , write_topology_filter
                              , write_hits
                              , select_and_write_tracks))

    return pipe( filter_events_nohits
               , fl.branch(write_no_hits_filter)
               , hits_passed.filter
               , copy_Efield
               , create_extract_track_blob_info
               , sort_hits_
               , filter_events_topology
               , fl.fork(*fork_pipes)
               )


@check_annotations
def hits_merger(same_peak : bool) -> Callable:
    return partial(hif.merge_NN_hits, same_peak=same_peak)


@check_annotations
def hits_thresholder(threshold_charge : float, same_peak : bool ) -> Callable:
    """
    Applies a threshold to hits and redistributes the charge/energy.

    Parameters
    ----------
    threshold_charge : float
        minimum pes of a hit
    same_peak        : bool
        whether to reassign NN hits' energy only to the hits from the same peak

    Returns
    ----------
    A function that takes HitCollection as input and returns another object with
    only non NN hits of charge above threshold_charge.
    The energy of NN hits is redistributed among neighbors.
    """

    def threshold_hits_and_merge_nn(hits: pd.DataFrame) -> pd.DataFrame:
        thr_hits = hif.threshold_hits(    hits, threshold_charge     )
        mrg_hits = hif.merge_NN_hits (thr_hits, same_peak = same_peak)
        return mrg_hits

    return threshold_hits_and_merge_nn


@check_annotations
def hits_corrector( filename     : str
                  , apply_temp   : bool
                  , norm_strat   : NormStrategy
                  , norm_options : Optional[dict] = dict()
                  , apply_z      : Optional[bool] = False
                  ) -> Callable:
    """
    Applies energy correction map and converts drift time to z.

    Parameters
    ----------
    map_fname  : string (filepath)
        filename of the map
    apply_temp : bool
        whether to apply temporal corrections
        must be set to False if no temporal correction dataframe exists in map file

    Returns
    ----------
    A function that takes a HitCollection as input and returns
    the same object with modified Ec and Z fields.
    """
    maps      = read_maps(os.path.expandvars(filename))
    get_coef  = apply_all_correction( maps
                                    , apply_temp = apply_temp
                                    , norm_strat = norm_strat
                                    , **norm_options)
    time_to_Z = get_df_to_z_converter(maps) if maps.t_evol is not None and apply_z else identity

    def correct(hits : pd.DataFrame) -> pd.DataFrame:
        corr_factors = get_coef(hits.X, hits.Y, hits.Z, hits.time)
        return hits.assign( Ec = hits.E * corr_factors
                          , Z  = time_to_Z(hits.Z) )

    return correct


def identity(x : Any) -> Any:
    return x
