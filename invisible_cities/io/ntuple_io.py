from .  table_io  import make_table
from .. evm.nh5   import NtupleTable
from .  hits_io   import hits_writer
from .  voxels_io import voxels_writer
from .  tracks_io import tracks_writer

def ntuple_writer(hdf5_file, *, compression='ZLIB4'):
    ntuple_table = make_table(hdf5_file,
                          group       = 'DST',
                          name        = 'Events',
                          fformat     = NtupleTable,
                          description = 'KDST Events',
                          compression = compression)

    # Mark column to index after populating table
    ntuple_table.set_attr('columns_to_index', ['event'])

    write_hits   = hits_writer(hdf5_file,   compression=compression)
    write_voxels = voxels_writer(hdf5_file, compression=compression)
    write_tracks = tracks_writer(hdf5_file, compression=compression)

    def write_ntuple(ntuple_event):
        hitc, voxels, tc, nte = ntuple_event

        write_hits(hitc)
        write_voxels(voxels)
        write_tracks(tc)

        nte.store(ntuple_table)
    return write_ntuple
