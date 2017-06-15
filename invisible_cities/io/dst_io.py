# import tables as tb
#
# from .. reco import tbl_functions as tbl
# from .. reco import nh5           as table_formats

from . table_io import make_table
from .. reco import nh5           as table_formats


def hits_writer(hdf5_file, *, compression='ZLIB4'):
    hits_table  = make_table(hdf5_file,
                          group='RECO',
                          name='Events',
                          fformat=table_formats.HitsTable,
                          description='Hits',
                          compression=compression)

    def write_hits(hits_event):
        hits_event.store(hits_table)
    return write_hits
