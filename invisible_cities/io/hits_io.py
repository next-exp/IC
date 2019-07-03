from typing                import Dict

import numpy               as     np
import pandas              as     pd

from . dst_io              import load_dst
from ..evm.event_model     import Hit
from ..evm.event_model     import Cluster
from ..evm.event_model     import HitCollection
from .. types.ic_types     import xy
from .  table_io           import make_table
from .. evm .nh5           import HitsTable
from .. types.ic_types     import NN


def hits_from_df (dst : pd.DataFrame, skip_NN : bool = False) -> Dict[int, HitCollection]:
    """
    Function that transforms pandas DataFrame dst to HitCollection
    ------
    Parameters
    ------
    dst : pd.DataFrame
        DataFrame with obligatory columns :
                event, time, npeak, nsipm, X, Y, Xrms, Yrms, Z,  Q, E
        If Qc, Ec, track_id are not inside dst the default value is set to -1
        If Xpeak, Ypeak not in dst the default value is -1000
    ------
    Returns
    ------
    Dictionary {event_number : HitCollection}
    """
    all_events = {}
    for (event, time) , hits_df in dst.groupby(['event', 'time']):
        #pandas is not consistent with numpy dtypes so we have to change it by hand
        event = np.int32(event)
        hits  = []
        for i, row in hits_df.iterrows():
            if skip_NN and row.Q == NN:
                continue
            hit = Hit(row.npeak,
                      Cluster(row.Q, xy(row.X, row.Y), xy(row.Xrms**2, row.Yrms**2),
                              row.nsipm, row.Z, row.E, Qc = getattr(row, 'Qc', -1)),
                      row.Z, row.E, xy(getattr(row, 'Xpeak', -1000) , getattr(row, 'Ypeak', -1000)),
                      s2_energy_c = getattr(row, 'Ec', -1), track_id = getattr(row, 'track_id', -1)) 

            hits.append(hit)

        if len(hits)>0:
            all_events.update({event : HitCollection(event, time)})
            all_events[event].hits.extend(hits)

    return all_events

# reader
def load_hits(DST_file_name : str, group_name : str = 'RECO', table_name : str = 'Events', skip_NN : bool = False
             )-> Dict[int, HitCollection]:
    """
    Function to load hits into HitCollection object.

    ------
    Parameters
    ------
    DST_file_name : str
    group_name    : str (default 'RECO')
        Name of the group inside pytable
    table_name    : str (default 'Events')
        Name of the table inside the group
    skip_NN       : bool (default False)
        whether to skip NN hits
    ------
    Returns
    ------
    Dictionary {event_number : HitCollection}
    """
    dst = load_dst(DST_file_name, group_name, table_name)
    hits_dict = {}
    # Check dst is defined and non-empty
    if dst is not None and len(dst):
        hits_dict = hits_from_df (dst, skip_NN)
    return hits_dict

def load_hits_skipping_NN(DST_file_name : str, group_name : str = 'RECO', table_name : str = 'Events'
                          )-> Dict[int, HitCollection]:
    """
    Function to load hits into HitCollection object.

    ------
    Parameters
    ------
    DST_file_name : str
    group_name    : str (default 'RECO')
        Name of the group inside pytable
    table_name    : str (default 'Events')
        Name of the table inside the group
    ------
    Returns
    ------
    Dictionary {event_number : HitCollection} with no NN hits
    """
    return load_hits (DST_file_name = DST_file_name, group_name = group_name, table_name = table_name, skip_NN = True)


# writers
def hits_writer(hdf5_file, group_name='RECO', table_name='Events', *, compression='ZLIB4'):
    hits_table  = make_table(hdf5_file,
                             group       = group_name,
                             name        = table_name,
                             fformat     = HitsTable,
                             description = 'Hits',
                             compression = compression)
    # Mark column to index after populating table
    hits_table.set_attr('columns_to_index', ['event'])

    def write_hits(hits_event):
        hits_event.store(hits_table)
    return write_hits
