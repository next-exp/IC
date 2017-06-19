from .  table_io import make_table
from .. reco.nh5 import HitsTable


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
