import numpy  as np
import tables as tb
import pandas as pd

from enum      import auto
from functools import partial

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


class mc_info_writer:
    """Write MC info to file."""
    def __init__(self, h5file, compression = 'ZLIB4'):

        self.h5file      = h5file
        self.compression = compression
        self._create_tables()
        self.reset()
        self.current_tables = None

        self.last_written_hit      = 0
        self.last_written_particle = 0
        self.first_extent_row      = True
        self.first_file            = True


    def reset(self):
        # last visited row
        self.last_row              = 0

    def _create_tables(self):
        """Create tables in MC group in file h5file."""
        if '/MC' in self.h5file:
            MC = self.h5file.root.MC
        else:
            MC = self.h5file.create_group(self.h5file.root, "MC")

        self.extent_table = self.h5file.create_table(MC, "extents",
                                                     description = MCExtentInfo,
                                                     title       = "extents",
                                                     filters     = tbl.filters(self.compression))

        # Mark column to index after populating table
        self.extent_table.set_attr('columns_to_index', ['evt_number'])

        self.hit_table = self.h5file.create_table(MC, "hits",
                                                  description = MCHitInfo,
                                                  title       = "hits",
                                                  filters     = tbl.filters(self.compression))

        self.particle_table = self.h5file.create_table(MC, "particles",
                                                       description = MCParticleInfo,
                                                       title       = "particles",
                                                       filters     = tbl.filters(self.compression))

        self.generator_table = self.h5file.create_table(MC, "generators",
                                                        description = MCGeneratorInfo,
                                                        title       = "generators",
                                                        filters     = tbl.filters(self.compression))



    def __call__(self, mctables: (tb.Table, tb.Table, tb.Table, tb.Table),
                 evt_number: int):
        if mctables is not self.current_tables:
            self.current_tables = mctables
            self.reset()

        extents = mctables[0]
        # Note: filtered out events do not make it to evt_number, but are included in extents dataset
        for iext in range(self.last_row, len(extents)):
            this_row = extents[iext]
            if this_row['evt_number'] == evt_number:
                if iext == 0:
                    if self.first_file:
                        modified_hit          = this_row['last_hit']
                        modified_particle     = this_row['last_particle']
                        self.first_extent_row = False
                        self.first_file       = False
                    else:
                        modified_hit          = this_row['last_hit'] + self.last_written_hit + 1
                        modified_particle     = this_row['last_particle'] + self.last_written_particle + 1
                        self.first_extent_row = False

                elif self.first_extent_row:
                    previous_row          = extents[iext-1]
                    modified_hit          = this_row['last_hit'] - previous_row['last_hit'] + self.last_written_hit - 1
                    modified_particle     = this_row['last_particle'] - previous_row['last_particle'] + self.last_written_particle - 1
                    self.first_extent_row = False
                    self.first_file       = False
                else:
                    previous_row      = extents[iext-1]
                    modified_hit      = this_row['last_hit'] - previous_row['last_hit'] + self.last_written_hit
                    modified_particle = this_row['last_particle'] - previous_row['last_particle'] + self.last_written_particle

                modified_row                  = self.extent_table.row
                modified_row['evt_number']    = evt_number
                modified_row['last_hit']      = modified_hit
                modified_row['last_particle'] = modified_particle
                modified_row.append()

                self.last_written_hit      = modified_hit
                self.last_written_particle = modified_particle

                break

        self.extent_table.flush()

        hits, particles, generators = read_mcinfo_evt(mctables, evt_number, self.last_row)
        self.last_row = iext + 1

        for h in hits:
            new_row = self.hit_table.row
            new_row['hit_position']  = h[0]
            new_row['hit_time']      = h[1]
            new_row['hit_energy']    = h[2]
            new_row['label']         = h[3]
            new_row['particle_indx'] = h[4]
            new_row['hit_indx']      = h[5]
            new_row.append()
        self.hit_table.flush()

        for p in particles:
            new_row = self.particle_table.row
            new_row['particle_indx']  = p[0]
            new_row['particle_name']  = p[1]
            new_row['primary']        = p[2]
            new_row['mother_indx']    = p[3]
            new_row['initial_vertex'] = p[4]
            new_row['final_vertex']   = p[5]
            new_row['initial_volume'] = p[6]
            new_row['final_volume']   = p[7]
            new_row['momentum']       = p[8]
            new_row['kin_energy']     = p[9]
            new_row['creator_proc']   = p[10]
            new_row.append()
        self.particle_table.flush()

        for g in generators:
            new_row = self.generator_table.row
            new_row['evt_number']    = g[0]
            new_row['atomic_number'] = g[1]
            new_row['mass_number']   = g[2]
            new_row['region']        = g[3]
            new_row.append()
        self.generator_table.flush()


def copy_mc_info(h5in : tb.File, writer : Type[mc_info_writer], which_events : List[int]=None):
    """Copy from the input file to the output file the MC info of certain events.

    Parameters
    ----------
    h5in  : tb.File
        Input h5 file.
    writer : instance of class mcinfo_io.mc_info_writer
        MC info writer to h5 file.
    which_events : None or list of ints
        List of IDs (i.e. event numbers) that identify the events to be copied
        to the output file. If None, all events in the input file are copied.
    """
    try:
        if which_events is None:
            which_events = h5in.root.MC.extents.cols.evt_number[:]
        mcinfo = get_mc_info(h5in)
        for n in which_events:
            writer(mctables=mcinfo, evt_number=n)
    except tb.exceptions.NoSuchNodeError:
        raise tb.exceptions.NoSuchNodeError(f"No MC info in file {h5in}.")
    except IndexError:
        raise IndexError(f"No event {n} in file {h5in}.")
    except NoParticleInfoInFile as err:
        print(f"Warning: {h5in}", err)
        pass


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
        raise tb.exceptions.NoSuchNodeError
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
        binning_keys = config.param_key.str.contains('binning')
        ## Split value and unit in binning rows and combin into
        ## standard unit value using system_of_units
        new_names    = config[binning_keys].param_key.str.split(
            '/', expand=True).apply(lambda x: x[2] + '_binning', axis=1)
        config.loc[binning_keys, 'param_key'] = new_names
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
            ## Add a column to the DataFrame so all info
            ## is present like in the new format
            sns_names = get_sensor_binning(file_name).index
            pmt_ids   = DB.DataPMT(db_file, run_no).SensorID
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
        missing_columns = ['final_momentum_x','final_momentum_y',
                           'final_momentum_z',          'length']
        parts = parts.reindex(parts.columns.tolist() + missing_columns, axis=1)

        # Setting the indexes
        parts.set_index(['event_id', 'particle_id'], inplace=True)

        return parts


def get_sensor_binning(file_name : str) -> pd.DataFrame:
    """
    Looks in the configuration table of the
    input file and extracts the binning used
    for both types of sensitive detector.

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
    if is_oldformat_file(file_name):
        return sensor_binning_old(file_name)
    else:
        return sensor_binning_new(file_name)


def sensor_binning_old(file_name : str) -> pd.DataFrame:
    """
    Returns the correct sampling used in nexus fullsim
    in the case of a pre-2020 format file.

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
    # Is a pre-Feb 2020 MC file
    config         = load_dst(file_name, 'MC',
                              'configuration').set_index('param_key')
    bins           = config[config.index.str.contains('binning')]
    bins.columns   = ['bin_width']
    bins.index     = bins.index.rename('sns_name')
    ## Protect against badly written old files
    bins           = bins[bins.index.str.contains('Geom')]
    ##
    bins.index     = bins.index.str.split('/', expand=True).levels[2]
    bins.bin_width = bins.bin_width.str.split(expand=True).apply(
        lambda x: float(x[0]) * getattr(units, x[1]), axis=1)
    return bins


def sensor_binning_new(file_name : str) -> pd.DataFrame:
    """
    Returns the correct sampling used in nexus fullsim
    in the case of a 2020-- format file.

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
    sns_names      = load_dst(file_name, 'MC', 'sns_positions').sensor_name
    if len(sns_names) == 0:
        return pd.DataFrame(index=['sns_name'], columns=['bin_width'])
    config         = load_dst(file_name, 'MC',
                              'configuration').set_index('param_key')
    ## In the configuration table the binning is named
    ## <sensor type>_binning, adjust to search.
    sns_binnings   = [nm + '_binning' for nm in sns_names.unique()]
    bins           = config.loc[sns_binnings].copy()
    bins.drop('file_index', axis=1, inplace=True, errors='ignore')
    bins.columns   = ['bin_width']
    bins.index     = bins.index.rename('sns_name')
    bins.index     = bins.index.str.strip('_binning')
    bins.bin_width = bins.bin_width.str.split(expand=True).apply(
        lambda x: float(x[0]) * getattr(units, x[1]), axis=1)
    return bins[~bins.index.duplicated()]


def get_sensor_types(file_name : str) -> pd.DataFrame:
    """
    returns a DataFrame linking sensor_ids to
    sensor type names.
    !! Only valid for new format data, otherwise use
    !! database.

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
