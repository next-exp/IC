from typing                import Dict

from functools             import partial

import numpy               as     np
import pandas              as     pd

from . dst_io              import load_dst
from . dst_io              import df_writer
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
                event, npeak, X, Y, Z,  Q, E
        If time, Ec, track_id are not inside dst the default value is set to -1
        If Xpeak, Ypeak not in dst the default value is -1000
    ------
    Returns
    ------
    Dictionary {event_number : HitCollection}
    """
    all_events = {}
    times = getattr(dst, 'time', [-1]*len(dst))
    for (event, time) , df in dst.groupby(['event', times]):
        #pandas is not consistent with numpy dtypes so we have to change it by hand
        event = np.int64(event)
        hits  = []
        for i, row in df.iterrows():
            Q = getattr(row,'Q', row.E)
            if skip_NN and Q == NN:
                continue
            Xpeak   = getattr(row, 'Xpeak'   , -1000)     # for backwards compatibility
            Ypeak   = getattr(row, 'Ypeak'   , -1000)     # for backwards compatibility
            Ec      = getattr(row, 'Ec'      , -1   )     # for backwards compatibility
            trackID = getattr(row, 'track_id', -1   )     # for backwards compatibility

            hit = Hit(row.npeak            ,
                      Cluster(Q               ,
                              xy(row.X, row.Y),
                              xy(    0,     0),
                              nsipm =      1  ,
                              z     = row.Z   ,
                              E     = row.E   ,
                              Qc    =  0      ),
                      row.Z                ,
                      row.E                ,
                      xy(Xpeak, Ypeak)     ,
                      s2_energy_c = Ec     ,
                      track_id    = trackID,
                      )

            hits.append(hit)

        if len(hits):
            all_events[event] = HitCollection(event, time, hits=hits)

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
def hits_writer(hdf5_file, group_name, table_name, *, compression=None):
    return partial( df_writer, hdf5_file
                  , group_name         = group_name
                  , table_name         = table_name
                  , descriptive_string = "Hits"
                  , columns_to_index   = ["event"]
                  , compression        = compression)
