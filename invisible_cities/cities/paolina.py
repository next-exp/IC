from functools   import reduce
from functools   import partial
from itertools   import combinations
from operator    import itemgetter
from collections import namedtuple

import numpy    as np
import networkx as nx


class Hit:

    def __init__(self, x,y,z, E):
        self.pos = np.array((x,y,z))
        self.E   = E

    def __str__(self):
        return '<{} {} {}>'.format(self.__class__.__name__,
                                   self.pos.tolist(), self.E)
    __repr__ = __str__


class Voxel:

    def __init__(self, x,y,z, E):
        self.pos = np.array((x,y,z))
        self.E   = E

    __str__  = Hit.__str__
    __repr__ =     __str__

    def __eq__(self, other):
        try:
            return np.array_equal(self.pos, other.pos) and self.E == other.E
        except AttributeError:
            return False

    def __hash__(self):
        return hash((self.E, tuple(self.pos)))

MAX3D = np.array([float(' inf')] * 3)
MIN3D = np.array([float('-inf')] * 3)

def bounding_box(seq):
    posns = [x.pos for x in seq]
    return (reduce(np.minimum, posns, MAX3D),
            reduce(np.maximum, posns, MIN3D))


def voxelize_hits(hits, voxel_dimensions):
    if not hits:
        return []
    hlo, hhi = bounding_box(hits)
    hranges = hhi - hlo
    bins = np.ceil(hranges / voxel_dimensions).astype(int)
    hit_positions = np.array([h.pos for h in hits])
    hit_energies  =          [h.E   for h in hits]
    E, edges = np.histogramdd(hit_positions, bins=bins, weights=hit_energies)

    def centres(a):
        return (a[1:] + a[:-1]) / 2

    cx, cy, cz = map(centres, edges)
    nz = np.nonzero(E)

    return [Voxel(cx[x], cy[y], cz[z], E[x,y,z])
            for (x,y,z) in np.stack(nz).T]

def make_track_graphs(voxels, voxel_dimensions):

    def neighbours(va, vb):
        return ((abs(va.pos - vb.pos) / voxel_dimensions) < 1.5).all()

    voxel_graph = nx.Graph()
    voxel_graph.add_nodes_from(voxels)
    for va, vb in combinations(voxels, 2):
        if neighbours(va, vb):
            voxel_graph.add_edge(va, vb,
                                 distance = np.linalg.norm(va.pos - vb.pos))

    return tuple(nx.connected_component_subgraphs(voxel_graph))


shortest_paths = partial(nx.all_pairs_dijkstra_path_length, weight='distance')


def find_extrema(distance : 'dict of dicts'):
    first, last, max_distance = None, None, 0
    for source, target in combinations(distance, 2):
        d = distance[source][target]
        if d > max_distance:
            first, last, max_distance = source, target, d
    return set((first, last))


def energy_within_radius(distances, radius):
    return sum(v.E for (v, d) in distances.items() if d < radius)


def blob_energies(track_graph, radius):
    distances = shortest_paths(track_graph)
    a,b = find_extrema(distances)
    Ea = energy_within_radius(distances[a], radius)
    Eb = energy_within_radius(distances[b], radius)
    return set((Ea, Eb))
