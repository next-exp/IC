from .  table_io  import make_table
from .. evm.nh5   import NtupleTable

def ntuple_writer(hdf5_file, *, compression='ZLIB4'):
    ntuple_table = make_table(hdf5_file,
                          group       = 'DST',
                          name        = 'Events',
                          fformat     = NtupleTable,
                          description = 'KDST Events',
                          compression = compression)

    # Mark column to index after populating table
    ntuple_table.set_attr('columns_to_index', ['event'])

    def write_ntuple(ntuple_event):
        ntuple_event.store(ntuple_table)

    return write_ntuple
