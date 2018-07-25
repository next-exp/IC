
import tables

from .  dst_io             import load_dst
from .. io.table_io        import make_table
from .. evm.event_model    import Voxel
from .. evm.event_model    import VoxelCollection
from .. evm.nh5            import VoxelsTable

def voxels_writer(hdf5_file, *, compression='ZLIB4'):

    voxels_table  = make_table(hdf5_file,
                             group       = 'Voxels',
                             name        = 'Voxels',
                             fformat     = VoxelsTable,
                             description = 'Voxels',
                             compression = compression)
    # Mark column to index after populating table
    voxels_table.set_attr('columns_to_index', ['event'])

    def write_voxels(voxels_event):
        voxels_event.store(voxels_table)

    return write_voxels

def load_voxels(DST_file_name):
    """Return the Voxels as PD DataFrames."""

    dst = tables.open_file(DST_file_name,'r')
    dst_size = len(dst.root.Voxels.Voxels)
    all_events = {}

    event = dst.root.Voxels.Voxels[:]['event']
    time  = dst.root.Voxels.Voxels[:]['time']
    X     = dst.root.Voxels.Voxels[:]['X']
    Y     = dst.root.Voxels.Voxels[:]['Y']
    Z     = dst.root.Voxels.Voxels[:]['Z']
    E     = dst.root.Voxels.Voxels[:]['E']
    size  = dst.root.Voxels.Voxels[:]['size']

    for i in range(dst_size):
        current_event = all_events.setdefault(event[i],
                                              VoxelCollection(event, time))
        voxel = Voxel(X[i], Y[i], Z[i], E[i], size[i])
        current_event.voxels.append(voxel)
    return all_events
