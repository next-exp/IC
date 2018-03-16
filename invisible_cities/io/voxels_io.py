
import tables

from .. io.table_io        import make_table
from .. evm.event_model    import Voxel
from .. evm.event_model    import VoxelCollection
from .. evm.nh5            import VoxelsTable

def true_voxels_writer(hdf5_file, *, compression='ZLIB4'):

    voxels_table  = make_table(hdf5_file,
                             group       = 'TrueVoxels',
                             name        = 'Voxels',
                             fformat     = VoxelsTable,
                             description = 'Voxels',
                             compression = compression)
    # Mark column to index after populating table
    voxels_table.set_attr('columns_to_index', ['event'])

    def write_voxels(evt_number,voxels_event):
        row = voxels_table.row
        for voxel in voxels_event:
            row["event"] = evt_number
            row["X"    ] = voxel.X
            row["Y"    ] = voxel.Y
            row["Z"    ] = voxel.Z
            row["E"    ] = voxel.E
            row["size" ] = voxel.size
            row.append()

    return write_voxels

def load_voxels(DST_file_name):
    """Return the Voxels as PD DataFrames."""

    dst = load_dst(DST_file_name, 'TrueVoxels', 'Voxels')
    dst_size = len(dst)
    all_events = {}

    event = dst.event.values
    X     = dst.X   .values
    Y     = dst.Y   .values
    Z     = dst.Z   .values
    E     = dst.E   .values
    size  = dst.size.values

    for i in range(dst_size):
        current_event = all_events.setdefault(event[i],
                                              VoxelCollection())
        voxel = Voxel(X[i], Y[i], Z[i], E[i], size[i])
        current_event.hits.append(voxel)
    return all_events
