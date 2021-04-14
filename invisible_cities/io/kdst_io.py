from .  table_io import make_table
from .. evm.nh5  import KrTable


def kr_writer(hdf5_file, *, compression='ZLIB4'):
    kr_table = make_table(hdf5_file,
                          group       = 'DST',
                          name        = 'Events',
                          fformat     = KrTable,
                          description = 'KDST Events',
                          compression = compression)
    # Mark column to index after populating table
    kr_table.set_attr('columns_to_index', ['event'])

    def write_kr(kr_event):
        kr_event.store(kr_table)
    return write_kr
