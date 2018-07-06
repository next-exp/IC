
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
    dst_size = len(dst.root.TrueVoxels.Voxels)
    all_events = {}

    event = dst.root.TrueVoxels.Voxels[:]['event']
    time  = dst.root.TrueVoxels.Voxels[:]['time']
    X     = dst.root.TrueVoxels.Voxels[:]['X']
    Y     = dst.root.TrueVoxels.Voxels[:]['Y']
    Z     = dst.root.TrueVoxels.Voxels[:]['Z']
    E     = dst.root.TrueVoxels.Voxels[:]['E']
    size  = dst.root.TrueVoxels.Voxels[:]['size']

    for i in range(dst_size):
        current_event = all_events.setdefault(event[i],
                                              VoxelCollection(event, time, []))
        voxel = Voxel(X[i], Y[i], Z[i], E[i], size[i])
        current_event.voxels.append(voxel)
    return all_events
