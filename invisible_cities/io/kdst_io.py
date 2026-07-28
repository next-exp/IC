from .. io.dst_io import df_writer
from .. evm.event_model import kr_events_type

def kdst_writer(h5out, compression=None):
    """
    For a given open table returns a writer for KDST dataframe info
    """
    def write_kdst(df):
        df = df.astype(kr_events_type)
        return df_writer(h5out              = h5out        ,
                         df                 = df           ,
                         compression        = compression  ,
                         group_name         = 'DST'        ,
                         table_name         = 'Events'     ,
                         descriptive_string = 'KDST Events',
                         columns_to_index   = ['event']    )
    return write_kdst
