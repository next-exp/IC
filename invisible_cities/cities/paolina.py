from collections import namedtuple

Hit   = namedtuple('Hit',   'x, y, z, E')
Voxel = namedtuple('Voxel', 'x, y, z, E')

def voxelize_hits(hits, voxel_dimensions):
    return [Voxel(1,1,1, 123.2)]
