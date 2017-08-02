from functools   import reduce
from functools   import partial
from itertools   import combinations

import numpy    as np
import networkx as nx

from networkx           import Graph
from .. evm.event_model import Voxel
from .. core.exceptions import NoHits
from .. core.exceptions import NoVoxels
from .. evm.event_model import BHit

from typing import Sequence
from typing import List
from typing import Tuple
from typing import Dict

MAX3D = np.array([float(' inf')] * 3)
MIN3D = np.array([float('-inf')] * 3)

def bounding_box(seq : BHit) ->Sequence[np.ndarray]:
    """Returns two arrays defining the coordinates of a box that bounds the voxels"""
    posns = [x.pos for x in seq]
    return (reduce(np.minimum, posns, MAX3D),
            reduce(np.maximum, posns, MIN3D))


def voxelize_hits(hits : Sequence[BHit], voxel_dimensions : np.ndarray) ->List[Voxel]:
    """1. Hits are enclosed by a bounding box.
       2. Boundix box is discretized (via a hitogramdd).
       3. The energy of all the hits insidex each discreet "voxel" is added.

     """
    if not hits:
        raise NoHits
    hlo, hhi = bounding_box(hits)
    hranges = hhi - hlo
    bins = np.ceil(hranges / voxel_dimensions).astype(int)
    hit_positions = np.array([h.pos for h in hits])
    hit_energies  =          [h.E   for h in hits]
    E, edges = np.histogramdd(hit_positions, bins=bins, weights=hit_energies)

    def centres(a : np.ndarray) -> np.ndarray:
        return (a[1:] + a[:-1]) / 2

    cx, cy, cz = map(centres, edges)
    nz = np.nonzero(E)

    return [Voxel(cx[x], cy[y], cz[z], E[x,y,z])
            for (x,y,z) in np.stack(nz).T]


def make_track_graphs(voxels : Voxel,  voxel_dimensions : np.ndarray) ->Sequence[Graph]:
    """Creates a graph where the voxels are the nodes and the edges are any
    pair of neighbour voxel. Two voxels are considered to be neighbours if
    their distance normalized to their size is smaller than a scale factor.

    """

    def neighbours(va : Voxel, vb : Voxel, scale : float = 1.5) ->bool:
        return ((abs(va.pos - vb.pos) / voxel_dimensions) < scale).all()

    voxel_graph = nx.Graph()
    voxel_graph.add_nodes_from(voxels)
    for va, vb in combinations(voxels, 2):
        if neighbours(va, vb):
            voxel_graph.add_edge(va, vb,
                                 distance = np.linalg.norm(va.pos - vb.pos))

    return tuple(nx.connected_component_subgraphs(voxel_graph))


def voxels_from_track_graph(track: Graph) ->List[Voxel]:
    """Create and return a list of voxels from a track graph"""

    voxels = [Voxel(t.X, t.Y, t.Z, t.E) for t in track.nodes()]
    return voxels


def shortest_paths(track_graph : Graph) -> Dict[Voxel, Dict[Voxel, float]]:
    """Compute shortest path lengths between all nodes in a weighted graph."""
    f = partial(nx.all_pairs_dijkstra_path_length, weight='distance')
    return f (track_graph)


#shortest_paths = partial(nx.all_pairs_dijkstra_path_length, weight='distance')


def find_extrema(distance : Dict[Voxel, Dict[Voxel, float]]) -> Tuple[Voxel, Voxel]:
    """Find the extrema of the track """
    if not distance:
        raise NoVoxels
    if len(distance) == 1:
        only_voxel = next(iter(distance))
        return (only_voxel, only_voxel)
    first, last, max_distance = None, None, 0
    for source, target in combinations(distance, 2):
        d = distance[source][target]
        if d > max_distance:
            first, last, max_distance = source, target, d
    return (first, last)


def energy_within_radius(distances : Dict[Voxel, Dict[Voxel, float]], radius : float) -> float:
    return sum(v.E for (v, d) in distances.items() if d < radius)


def voxels_within_radius(distances : Dict[Voxel, Dict[Voxel, float]],
                         radius : float) -> List[Voxel]:

    return [v for (v, d) in distances.items() if d < radius]


def blob_energies(track_graph : Graph, radius : float) ->Tuple[float, float]:
    """Return the energies around the extrema of the track. """
    distances = shortest_paths(track_graph)
    a,b = find_extrema(distances)
    Ea = energy_within_radius(distances[a], radius)
    Eb = energy_within_radius(distances[b], radius)
    return (Ea, Eb) if Ea < Eb else (Eb, Ea)


def blobs(track_graph : Graph, radius : float) ->Tuple[List[Voxel], List[Voxel]]:
    """Return the blobs (list of voxels) around the extrema of the track. """
    distances = shortest_paths(track_graph)
    a,b = find_extrema(distances)
    ba = voxels_within_radius(distances[a], radius)
    bb = voxels_within_radius(distances[b], radius)

    return (ba, bb)
