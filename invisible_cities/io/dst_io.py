import tables as tb
import pandas as pd

from .  table_io           import make_table
from .. evm .nh5           import HitsTable


def load_dst(filename, group, node):
    with tb.open_file(filename) as h5in:
        table = getattr(getattr(h5in.root, group), node).read()
        return pd.DataFrame.from_records(table)


def load_dsts(dst_list, group, node):
    dsts = [load_dst(filename, group, node) for filename in dst_list]
    return pd.concat(dsts)

def hits_writer(hdf5_file, *, compression='ZLIB4'):
    hits_table  = make_table(hdf5_file,
                             group       = 'RECO',
                             name        = 'Events',
                             fformat     = HitsTable,
                             description = 'Hits',
                             compression = compression)

    hits_table.cols.event.create_index()

    def write_hits(hits_event):
        hits_event.store(hits_table)
    return write_hits
