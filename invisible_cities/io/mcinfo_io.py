import numpy  as np
import tables as tb
import pandas as pd

from enum      import auto
from functools import partial

import warnings

from .. reco            import        tbl_functions as   tbl
from .. core            import      system_of_units as units
from .. core.exceptions import NoParticleInfoInFile

from .. evm.event_model import                MCHit
from .. evm.event_model import               MCInfo
from .. evm.nh5         import      MCGeneratorInfo
from .. evm.nh5         import         MCExtentInfo
from .. evm.nh5         import            MCHitInfo
from .. evm.nh5         import       MCParticleInfo
from .. database        import              load_db as    DB
from .  dst_io          import             load_dst
from .  dst_io          import            df_writer
from .. types.ic_types  import     AutoNameEnumBase

from typing import Callable
from typing import     Dict
from typing import     List
from typing import  Mapping
from typing import Optional
from typing import Sequence
from typing import     Type
from typing import    Union


class MCTableType(AutoNameEnumBase):
    configuration    = auto()
    events           = auto()
    event_mapping    = auto()
    extents          = auto()
    generators       = auto()
    hits             = auto()
    particles        = auto()
    sensor_positions = auto()
    sns_positions    = auto()
    sns_response     = auto()
    waveforms        = auto()


def mc_writer(h5out : tb.file.File) -> Callable:
    """
    Writes the MC tables to the output file.

    parameters
    ----------
    h5out : pytables file
            the file to which the MC info is to be written

    returns
    -------
    write_mctables : Callable
        Function which writes to the file.
    """
    mcwriter_ = partial(df_writer         ,
                        h5out      = h5out,
                        group_name =  'MC')
    def write_mctables(table_dict : Dict):
        """
        Writer function

        parameters
        ----------
        table_dict : Dict
                     Dictionary with key MCTableType
                     and value pd.DataFrame with the
                     information from the input file
                     which has to be written.
        """
        first_indx = check_last_merge_index(h5out) + 1
        for key, tbl in table_dict.items():
            if (key is MCTableType.configuration or
                key is MCTableType.event_mapping   ):
                try:
                    orig_indx = tbl.file_index.unique()
                    new_indx  = np.arange(first_indx                 ,
                                          first_indx + len(orig_indx))
                    tbl.file_index.replace(orig_indx, new_indx, inplace=True)
                except AttributeError:
                    tbl['file_index'] = first_indx
            mcwriter_(df=tbl, table_name=key.name)
    return write_mctables


def check_last_merge_index(h5out : tb.file.File) -> int:
    """
    Get the last file index used to index merged files
    in the configuration table and mapping

    parameters
    ----------
    h5out: pytables file
           the file for output

    returns
    -------
    indx: int
          integer for the last saved file index
    """
    if 'MC' in h5out.root:
        if 'event_mapping' in h5out.root.MC:
            return h5out.root.MC.event_mapping.cols.file_index[-1]
    return -1


def read_mc_tables(file_in : str                        ,
                   evt_arr : Optional[np.ndarray] = None,
                   db_file : Optional[str]        = None,
                   run_no  : Optional[int]        = None) -> Dict:
    """
    Reads all the MC tables present in
    file_in and stores the Dataframes for
    each into a dictionary.

    parameters
    ----------
    file_in: str
             Name of the input file
    evt_arr: optional np.ndarray
             list of events or None. default None for
             all events.
    db_file: optional str
             Name of the database to be used.
             Only required for old format MC files
    run_no : optional int
             Run number for database access
             Only required for old format MC files

    returns
    -------
    tbl_dict : Dict
               A dictionary with key type MCTableType
               and values of type pd.DataFrame
               containing the MC info for all tables
               sorted into new format in the case
               of an old format MC file.
    """
    mctbls = get_mc_tbl_list(file_in)
    if evt_arr is None:
        evt_arr = get_event_numbers_in_file(file_in)
    tbl_dict = {}
    for tbl in mctbls:
        if   tbl is MCTableType.hits                 :
            hits = load_mchits_df(file_in).reset_index()
            tbl_dict[tbl] = hits[hits.event_id.isin(evt_arr)]
        elif tbl is MCTableType.particles            :
            parts = load_mcparticles_df(file_in).reset_index()
            tbl_dict[tbl] = parts[parts.event_id.isin(evt_arr)]
        elif tbl is MCTableType.extents              :
            pass
        elif (tbl is MCTableType.sns_response or
              tbl is MCTableType.waveforms      )    :
            sns = load_mcsensor_response_df(file_in, return_raw=True)
            tbl_key = MCTableType.sns_response
            tbl_dict[tbl_key] = sns[sns.event_id.isin(evt_arr)]
        elif (tbl is MCTableType.sensor_positions or
              tbl is MCTableType.sns_positions      ):
            pos = load_mcsensor_positions(file_in, db_file, run_no)
            tbl_key = MCTableType.sns_positions
            tbl_dict[tbl_key] = pos
        elif tbl is MCTableType.generators           :
            gen = load_mcgenerators(file_in)
            tbl_dict[tbl] = gen[gen.evt_number.isin(evt_arr)]
        elif tbl is MCTableType.configuration        :
            config = load_mcconfiguration(file_in)
            tbl_dict[tbl] = config
        elif tbl is MCTableType.events               :
            pass
        elif tbl is MCTableType.event_mapping        :
            evt_map = load_mcevent_mapping(file_in)
            tbl_dict[tbl] = evt_map[evt_map.event_id.isin(evt_arr)]
        else                                         :
            raise TypeError("MC table has no reader")
    return tbl_dict


def copy_mc_info(file_in      : str                       ,
                 writer       : Type[mc_writer]           ,
                 which_events : Optional[List[int]] = None,
                 db_file      : Optional[str]       = None,
                 run_number   : Optional[int]       = None) -> None:
    """
    Copy from the input file to the output file the MC info of certain events.

    Parameters
    ----------
    file_in      : str
        Input file name.
    writer       : instance of class mcinfo_io.mc_info_writer
        MC info writer to h5 file.
    which_events : None or list of ints
        List of IDs (i.e. event numbers) that identify the events to be copied
        to the output file. If None, all events in the input file are copied.
    db_file      : None or str
        Name of the database to be used.
        Only required in cas of old format MC file
    run_number   : None or int
        Run number to be used for database access.
        Only required in case of old format MC file
    """
    if which_events is None:
        which_events = get_event_numbers_in_file(file_in)
    tbl_dict = read_mc_tables(file_in, which_events, db_file, run_number)
    if MCTableType.event_mapping not in tbl_dict.keys():
        new_key = MCTableType.event_mapping
        tbl_dict[new_key] = pd.DataFrame(which_events, columns=['event_id'])
    writer(tbl_dict)


def is_oldformat_file(file_name : str) -> bool:
    """
    Checks if the file type is pre 2020 or not

    parameters
    ----------
    file_name : str
                File name of the input file

    return
    ------
    bool: True if MC.extents table found: pre-2020 format
          False if MC.extents not found : 2020-- format
    """
    with tb.open_file(file_name) as h5in:
        return hasattr(h5in.root, 'MC/extents')


def get_mc_tbl_list(file_name: str) -> List[MCTableType]:
    """
    Returns a list of the tables in
    the MC group of a given file

    parameters
    ----------
    file_name : str
                Name of the input file

    returns
    -------
    tbl_list : List[MCTableType]
               A list of the MC tables which are present
               in the input file.
    """
    with tb.open_file(file_name, 'r') as h5in:
        mc_group = h5in.root.MC
        return [MCTableType[tbl.name] for tbl in mc_group]


def get_event_numbers_in_file(file_name: str) -> np.ndarray:
    """
    Get a list of the event numbers in the file
    based on the MC tables

    parameters
    ----------
    file_name : str
                File name of the input file

    returns
    -------
    evt_arr   : np.ndarray
                numpy array containing the list of all
                event numbers in file.
    """
    with tb.open_file(file_name, 'r') as h5in:
        if is_oldformat_file(file_name):
            evt_list = h5in.root.MC.extents.cols.evt_number[:]
        else:
            evt_list = _get_list_of_events_new(h5in)
        return evt_list


def _get_list_of_events_new(h5in : tb.file.File) -> np.ndarray:
    mc_tbls  = ['hits', 'particles', 'sns_response']
    def try_unique_evt_itr(group, itr):
        for elem in itr:
            try:
                yield np.unique(getattr(group, elem).cols.event_id)
            except tb.exceptions.NoSuchNodeError:
                pass

    evt_list = list(try_unique_evt_itr(h5in.root.MC, mc_tbls))
    if len(evt_list) == 0:
        raise AttributeError("At least one of MC/hits, MC/particles, \
        MC/sns_response must be present to use get_list_of_events.")
    return np.unique(np.concatenate(evt_list)).astype(int)


def load_mcconfiguration(file_name : str) -> pd.DataFrame:
    """
    Load the MC.configuration table from file into
    a pd.DataFrame

    parameters
    ----------
    file_name : str
                Name of the file with MC info

    returns
    -------
    config     : pd.DataFrame
                 DataFrame with all nexus configuration
                 parameters
    """
    config           = load_dst(file_name, 'MC', 'configuration')
    if is_oldformat_file(file_name):
        config.param_key = config.param_key.str.replace('.*Pmt.*\_binning.*' ,
                                                        'Pmt_binning'        )
        config.param_key = config.param_key.str.replace('.*SiPM.*\_binning.*',
                                                        'SiPM_binning'       )
        config           = config.drop_duplicates('param_key').reset_index(drop=True)
    return config


def load_mcsensor_positions(file_name : str,
                            db_file   : Optional[str] = None,
                            run_no    : Optional[int] = None) -> pd.DataFrame:
    """
    Load the sensor positions stored in the MC group
    into a pd.DataFrame

    parameters
    ----------
    file_name : str
                Name of the file containing MC info
    db_file   : None or str
                Name of the database to be used.
                Only required for pre-2020 format files
    run_no    : None or int
                Run number to be used for database access.
                Only required for pre-2020 format files

    returns
    -------
    sns_pos   : pd.DataFrame
                DataFrame containing information about
                the positions and types of sensors in
                the MC simulation.
    """
    if is_oldformat_file(file_name):
        sns_pos = load_dst(file_name, 'MC', 'sensor_positions')
        if sns_pos.shape[0] > 0:
            if db_file is None or run_no is None:
                warnings.warn(f' Database and file number needed', UserWarning)
                return pd.DataFrame(columns=['sensor_id', 'sensor_name',
                                             'x', 'y', 'z'])
            ## Add a column to the DataFrame so all info
            ## is present like in the new format
            pmt_ids   = DB.DataPMT(db_file, run_no).SensorID
            sns_names = get_sensor_binning(file_name).index
            pmt_name  = sns_names.str.contains('Pmt')
            pmt_pos   = sns_pos.sensor_id.isin(pmt_ids)
            sns_pos.loc[pmt_pos, 'sensor_name'] = sns_names[pmt_name][0]
            sns_pos.sensor_name.fillna(sns_names[~pmt_name][0], inplace=True)
    else:
        sns_pos = load_dst(file_name, 'MC', 'sns_positions'   )
    return sns_pos


def load_mcgenerators(file_name : str) -> pd.DataFrame:
    """
    Load the generator information to a pd.DataFrame

    parameters
    ----------
    file_name : str
                Name of the file containing MC info.

    returns
    -------
    pd.DataFrame with the generator information
    available for the MC events in the file.
    """
    return load_dst(file_name, 'MC', 'generators')


def load_mcevent_mapping(file_name : str) -> pd.DataFrame:
    """
    Load the event mapping information into a pd.DataFrame

    parameters
    ----------
    file_name : str
                Name of the file containing MC info.

    returns
    -------
    pd.DataFrame with the mapping information between
    configurations and files in case of a merged file.

    """
    return load_dst(file_name, 'MC', 'event_mapping')


def load_mchits_df(file_name : str) -> pd.DataFrame:
    """
    Opens file and calls read_mchits_df

    parameters
    ----------
    file_name : str
                The name of the file to be read

    returns
    -------
    hits : pd.DataFrame
           DataFrame with the information stored in file
           about the energy deposits in the ACTIVE volume
           in nexus simulations.
    """
    if is_oldformat_file(file_name):
        ## Is this a pre-Feb 2020 file?
        return load_mchits_dfold(file_name)
    else:
        return load_mchits_dfnew(file_name)


def load_mchits_dfnew(file_name : str) -> pd.DataFrame:
    """
    Returns MC hit information for 2020-- format files

    parameters
    ----------
    file_name : str
                Name of the file containing MC info

    returns
    -------
    hits : pd.DataFrame
           DataFrame with the information stored in file
           about the energy deposits in the ACTIVE volume
           in nexus simulations.
    """
    hits = load_dst(file_name, 'MC', 'hits')
    hits.set_index(['event_id', 'particle_id', 'hit_id'],
                    inplace=True)
    return hits


def load_mchits_dfold(file_name : str) -> pd.DataFrame:
    """
    Loads the MC hit information into a pandas DataFrame.
    For pre-2020 format files

    parameters
    ----------
    file_name : str
                Name of the file containing MC info

    returns
    -------
    hits : pd.DataFrame
           DataFrame with the information stored in file
           about the energy deposits in the ACTIVE volume
           in nexus simulations.
    """
    extents = load_dst(file_name, 'MC', 'extents')
    with tb.open_file(file_name) as h5in:
        hits_tb = h5in.root.MC.hits

        # Generating hits DataFrame
        hits = pd.DataFrame({'hit_id'     : hits_tb.col('hit_indx')           ,
                             'particle_id': hits_tb.col('particle_indx')      ,
                             'x'          : hits_tb.col('hit_position')[:, 0] ,
                             'y'          : hits_tb.col('hit_position')[:, 1] ,
                             'z'          : hits_tb.col('hit_position')[:, 2] ,
                             'time'       : hits_tb.col('hit_time')           ,
                             'energy'     : hits_tb.col('hit_energy')         ,
                             'label'      : hits_tb.col('label').astype('U13')})

        evt_hit_df = extents[['last_hit', 'evt_number']]
        evt_hit_df.set_index('last_hit', inplace=True)

        hits = hits.merge(evt_hit_df          ,
                          left_index  =   True,
                          right_index =   True,
                          how         = 'left')
        hits.rename(columns={"evt_number": "event_id"}, inplace=True)
        hits.event_id.fillna(method='bfill', inplace=True)
        hits.event_id = hits.event_id.astype(int)

        # Setting the indexes
        hits.set_index(['event_id', 'particle_id', 'hit_id'], inplace=True)

        return hits


def cast_mchits_to_dict(hits_df: pd.DataFrame) -> Mapping[int, List[MCHit]]:
    """
    Casts the mchits dataframe to an
    old style mapping.

    paramerters
    -----------
    hits_df : pd.DataFrame
              DataFrame containing event deposit information

    returns
    -------
    hit_dict : Mapping
               The same hit information cast into a dictionary
               using MCHit objects.
    """
    hit_dict = {}
    for evt, evt_hits in hits_df.groupby(level=0):
        hit_dict[evt] = [MCHit( hit.iloc[0, :3].values,
                               *hit.iloc[0, 3:].values)
                         for _, hit in evt_hits.groupby(level=2)]
    return hit_dict


def load_mcparticles_df(file_name: str) -> pd.DataFrame:
    """
    Opens file and calls read_mcparticles_df

    parameters
    ----------
    file_name : str
                The name of the file to be read

    returns
    -------
    parts : pd.DataFrame
            DataFrame containing MC particle information.
    """
    if is_oldformat_file(file_name):
        return load_mcparticles_dfold(file_name)
    else:
        return load_mcparticles_dfnew(file_name)


def load_mcparticles_dfnew(file_name: str) -> pd.DataFrame:
    """
    Loads MC particle info from file into a pd.DataFrame

    parameters
    ----------
    file_name : str
                Name of file containing MC info.

    returns
    -------
    particles : pd.DataFrame
                DataFrame containg the MC particle info
                stored in file_name.
    """
    particles         = load_dst(file_name, 'MC', 'particles')
    particles.primary = particles.primary.astype('bool')
    particles.set_index(['event_id', 'particle_id'], inplace=True)
    return particles


def load_mcparticles_dfold(file_name: str) -> pd.DataFrame:
    """
    A reader for the MC particle output based
    on pandas DataFrames.

    parameters
    ----------
    file_name: string
               Name of the file to be read

    returns
    -------
    parts : pd.DataFrame
            DataFrame containing the MC particle info
            contained in file_name.
    """
    extents = load_dst(file_name, 'MC', 'extents')
    with tb.open_file(file_name, mode='r') as h5in:
        p_tb = h5in.root.MC.particles

        # Generating parts DataFrame
        parts = pd.DataFrame({'particle_id'       : p_tb.col('particle_indx'),
                              'particle_name'     : p_tb.col('particle_name').astype('U20'),
                              'primary'           : p_tb.col('primary').astype('bool'),
                              'mother_id'         : p_tb.col('mother_indx'),
                              'initial_x'         : p_tb.col('initial_vertex')[:, 0],
                              'initial_y'         : p_tb.col('initial_vertex')[:, 1],
                              'initial_z'         : p_tb.col('initial_vertex')[:, 2],
                              'initial_t'         : p_tb.col('initial_vertex')[:, 3],
                              'final_x'           : p_tb.col('final_vertex')[:, 0],
                              'final_y'           : p_tb.col('final_vertex')[:, 1],
                              'final_z'           : p_tb.col('final_vertex')[:, 2],
                              'final_t'           : p_tb.col('final_vertex')[:, 3],
                              'initial_volume'    : p_tb.col('initial_volume').astype('U20'),
                              'final_volume'      : p_tb.col('final_volume').astype('U20'),
                              'initial_momentum_x': p_tb.col('momentum')[:, 0],
                              'initial_momentum_y': p_tb.col('momentum')[:, 1],
                              'initial_momentum_z': p_tb.col('momentum')[:, 2],
                              'kin_energy'        : p_tb.col('kin_energy'),
                              'creator_proc'      : p_tb.col('creator_proc').astype('U20')})

        # Adding event info
        evt_part_df = extents[['last_particle', 'evt_number']]
        evt_part_df.set_index('last_particle', inplace=True)
        parts = parts.merge(evt_part_df         ,
                            left_index  =   True,
                            right_index =   True,
                            how         = 'left')
        parts.rename(columns={"evt_number": "event_id"}, inplace=True)
        parts.event_id.fillna(method='bfill', inplace=True)
        parts.event_id = parts.event_id.astype(int)

        ## Add columns present in new format
        missing_columns = ['final_momentum_x', 'final_momentum_y',
                           'final_momentum_z', 'length', 'final_proc']
        parts = parts.reindex(parts.columns.tolist() + missing_columns, axis=1)

        # Setting the indexes
        parts.set_index(['event_id', 'particle_id'], inplace=True)

        return parts


def get_sensor_binning(file_name : str) -> pd.DataFrame:
    """
    Looks in the configuration table of the
    input file and extracts the binning used
    for all types of sensitive detector.

    parameters
    ----------
    file_name : str
                Name of the file containing MC info.

    returns
    -------
    bins      : pd.DataFrame
                DataFrame containing the sensor types
                and the sampling width (binning) used
                in full simulation.
    """
    config         = load_mcconfiguration(file_name).set_index('param_key')
    bins           = config[config.index.str.contains('binning')].copy()
    if bins.empty:
        warnings.warn(f' No binning info available.', UserWarning)
        return pd.DataFrame(columns=['sns_name', 'bin_width'])
    bins.drop('file_index', axis=1, inplace=True, errors='ignore')
    bins.columns   = ['bin_width']
    bins.index     = bins.index.rename('sns_name')
    bins.index     = bins.index.str.strip('_binning')
    ## Combine value and unit in configuration to get
    ## binning in standard units.
    bins.bin_width = bins.bin_width.str.split(expand=True).apply(
        lambda x: float(x[0]) * getattr(units, x[1]), axis=1)
    ## Drop protects probably out of date 2020 file NextFlex
    return bins.drop(bins[bins.index.str.contains('Geom')].index)


def get_sensor_types(file_name : str) -> pd.DataFrame:
    """
    returns a DataFrame linking sensor_ids to
    sensor type names.
    !! Only valid for new format data, otherwise use
    !! database.
    raises exception if old format file used

    parameters
    ----------
    file_name : str
                name of the file with nexus sensor info.

    returns
    -------
    sns_pos : pd.DataFrame
              Sensor position info for the MC sensors
              which saw light in this simulation.
    """
    if is_oldformat_file(file_name):
        raise TypeError('Old format files not valid for get_sensor_types')
    sns_pos = load_dst(file_name, 'MC', 'sns_positions').copy()
    sns_pos.drop(['x', 'y', 'z'], axis=1, inplace=True)
    return sns_pos


def load_mcsensor_response_df(file_name  : str                   ,
                              return_raw : Optional[bool] = False,
                              db_file    : Optional[ str] =  None,
                              run_no     : Optional[ int] =  None
                              ) -> pd.DataFrame:
    """
    A reader for the MC sensor output based
    on pandas DataFrames.

    parameters
    ----------
    file_name  : string
                 Name of the file to be read
    return_raw : bool
                 Return without conversion of time_bin to
                 time if True
    db_file    : None orstring
                 Name of the detector database to be accessed.
                 Only required for pre-2020 format files.
    run_no     : None or int
                 Run number for database access.
                 Only required for pre-2020 format files.

    returns
    -------
    sns_resp : pd.DataFrame
               DataFrame containing the sensor response info
               contained in file_name
    """
    if is_oldformat_file(file_name):
        sns_resp = load_mcsensors_dfold(file_name)
        if return_raw:
            return sns_resp
        assert(db_file), "Database name required for this file"
        assert( run_no), "Run number required for database access"
        return convert_timebin_to_time(sns_resp                          ,
                                       get_sensor_binning     (file_name),
                                       load_mcsensor_positions(file_name,
                                                               db_file  ,
                                                               run_no   ))
    else:
        sns_resp = load_dst(file_name, 'MC', 'sns_response')
        if return_raw:
            return sns_resp
        return convert_timebin_to_time(sns_resp                          ,
                                       get_sensor_binning     (file_name),
                                       load_mcsensor_positions(file_name))


def load_mcsensors_dfold(file_name : str) -> pd.DataFrame:
    """
    Load MC sensor info for pre-2020 format files

    parameters
    ----------
    file_name : str
                Name of the file containing MC info.

    returns
    -------
    sns       : pd.DataFrame
                DataFrame containing the sensor response info
                contained in file_name
    """
    extents  = load_dst(file_name, 'MC', 'extents')
    sns      = load_dst(file_name, 'MC', 'waveforms')
    evt_sns  = extents[['last_sns_data', 'evt_number']]
    evt_sns.set_index('last_sns_data', inplace=True)
    sns      = sns.merge(evt_sns             ,
                         left_index  =   True,
                         right_index =   True,
                         how         = 'left')
    sns.evt_number.fillna(method='bfill', inplace=True)
    sns.evt_number = sns.evt_number.astype(int)
    sns.rename(columns = {'evt_number': 'event_id'}, inplace=True)
    return sns


def get_mc_info(h5in):
    """Return MC info bank"""

    extents   = h5in.root.MC.extents
    hits      = h5in.root.MC.hits
    particles = h5in.root.MC.particles

    try:
        h5in.root.MC.particles[0]
    except:
        raise NoParticleInfoInFile('Trying to access particle information: this file could have sensor response only.')

    if len(h5in.root.MC.hits) == 0:
        hits = np.zeros((0,), dtype=('3<f4, <f4, <f4, S20, <i4, <i4'))
        hits.dtype.names = ('hit_position', 'hit_time', 'hit_energy', 'label', 'particle_indx', 'hit_indx')

    if 'generators' in h5in.root.MC:
        generator_table = h5in.root.MC.generators
    else:
        generator_table = np.zeros((0,), dtype=('<i4,<i4,<i4,S20'))
        generator_table.dtype.names = ('evt_number', 'atomic_number', 'mass_number', 'region')

    return MCInfo(extents, hits, particles, generator_table)


def convert_timebin_to_time(sns_resp : pd.DataFrame,
                            sns_bins : pd.DataFrame,
                            sns_pos  : pd.DataFrame) -> pd.DataFrame:
    """
    Convert the time bin to an event time.

    parameters
    ----------
    sns_resp : pd.DataFrame
               MC sensor response information as saved in the file.
    sns_bins : pd.DataFrame
               MC sensor bin sample width information
    sns_pos  : pd.DataFrame
               Sensor position and type information.

    returns
    -------
    sns_merge : pd.DataFrame
                Merge of the parameter information with sensor
                response information but with event time info
                instead of time_bin info.
    """
    sns_pos.set_index('sensor_name', inplace=True)
    sns_merge = sns_resp.merge(sns_pos.join(sns_bins), on='sensor_id')
    sns_merge['time'] = sns_merge.bin_width * sns_merge.time_bin
    sns_merge.drop(['x', 'y', 'z', 'time_bin', 'bin_width'],
                   axis=1, inplace=True)
    sns_merge.set_index(['event_id', 'sensor_id'], inplace=True)
    return sns_merge


def read_mcinfo_evt (mctables: (tb.Table, tb.Table, tb.Table, tb.Table), event_number: int, last_row=0,
                     return_only_hits: bool=False) -> ([tb.Table], [tb.Table], [tb.Table]):
    h5extents    = mctables[0]
    h5hits       = mctables[1]
    h5particles  = mctables[2]
    h5generators = mctables[3]

    particle_rows  = []
    hit_rows       = []
    generator_rows = []

    event_range = (last_row, int(1e9))
    for iext in range(*event_range):
        this_row = h5extents[iext]
        if this_row['evt_number'] == event_number:
            # the indices of the first hit and particle are 0 unless the first event
            #  written is to be skipped: in this case they must be read from the extents
            ihit = ipart = 0
            if iext > 0:
                previous_row = h5extents[iext-1]

                ihit         = int(previous_row['last_hit']) + 1
                if not return_only_hits:
                    ipart        = int(previous_row['last_particle']) + 1

            ihit_end  = this_row['last_hit']
            if len(h5hits) != 0:
                while ihit <= ihit_end:
                    hit_rows.append(h5hits[ihit])
                    ihit += 1

            if return_only_hits: break

            ipart_end = this_row['last_particle']
            while ipart <= ipart_end:
                particle_rows.append(h5particles[ipart])
                ipart += 1

            # It is possible for the 'generators' dataset to be empty. In this case, do not add any rows to 'generators'.
            if len(h5generators) != 0:
                generator_rows.append(h5generators[iext])

            break

    return hit_rows, particle_rows, generator_rows


def _read_mchit_info(h5f, event_range=(0, int(1e9))) -> Mapping[int, Sequence[MCHit]]:
    """Returns all hits in the event"""
    mc_info = get_mc_info(h5f)
    h5extents = mc_info.extents
    events_in_file = len(h5extents)

    all_events = {}

    for iext in range(*event_range):
        if iext >= events_in_file:
            break

        evt_number     = h5extents[iext]['evt_number']
        hit_rows, _, _ = read_mcinfo_evt(mc_info, evt_number, iext, True)

        hits = []
        for h5hit in hit_rows:
            hit = MCHit(h5hit['hit_position'],
                        h5hit['hit_time'],
                        h5hit['hit_energy'],
                        h5hit['label'].decode('utf-8','ignore'))
            hits.append(hit)

        all_events[evt_number] = hits

    return all_events
