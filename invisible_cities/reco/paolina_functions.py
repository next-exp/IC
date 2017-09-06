from functools   import reduce
from functools   import partial
from itertools   import combinations

import numpy    as np
import networkx as nx

from networkx                   import Graph
from .. evm.event_model         import Voxel
from .. core.exceptions         import NoHits
from .. core.exceptions         import NoVoxels
from .. evm.event_model         import BHit
from .. evm.event_model         import Voxel
from .. evm.event_model         import Track
from .. evm.event_model         import Blob
from .. evm.event_model         import TrackCollection
from .. core.system_of_units_c  import units

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


def voxelize_hits(hits             : Sequence[BHit],
                  voxel_dimensions : np.ndarray)->List[Voxel]:
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
    voxels = [Voxel(cx[x], cy[y], cz[z], E[x,y,z])
            for (x,y,z) in np.stack(nz).T]

    return voxels


def make_track_graphs(voxels           : Voxel,
                      voxel_dimensions : np.ndarray,
                      contiguity       : float = 1) ->Sequence[Graph]:
    """Creates a graph where the voxels are the nodes and the edges are any
    pair of neighbour voxel. Two voxels are considered to be neighbours if
    their distance normalized to their size is smaller than a
    contiguity factor .

    """

    def neighbours(va : Voxel, vb : Voxel, scale : float = 1.0) ->bool:
        return ((abs(va.pos - vb.pos) / voxel_dimensions) < contiguity).all()

    voxel_graph = nx.Graph()
    voxel_graph.add_nodes_from(voxels)
    for va, vb in combinations(voxels, 2):
        if neighbours(va, vb):
            voxel_graph.add_edge(va, vb, distance = np.linalg.norm(va.pos - vb.pos))

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

def length(track_graph):
    return len(shortest_paths(track_graph))


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


def compute_blobs(track_graph : Graph, radius : float) ->Tuple[List[Voxel], List[Voxel]]:
    """Return the blobs (list of voxels) around the extrema of the track. """
    distances = shortest_paths(track_graph)
    a,b = find_extrema(distances)
    ba = voxels_within_radius(distances[a], radius)
    bb = voxels_within_radius(distances[b], radius)

    return a, b, ba, bb


def make_tracks(evt_number       : float,
                evt_time         : float,
                voxels           : List[Voxel],
                voxel_dimensions : np.ndarray,
                contiguity       : float = 1,
                blob_radius      : float = 30*units.mm) ->TrackCollection:
    """Makes a track collection. """

    tc = TrackCollection(evt_number, evt_time) # type: TrackCollection
    track_graphs = make_track_graphs(voxels, voxel_dimensions) # type: Sequence[Graph]
    for trk in track_graphs:
        # distances = shortest_paths(trk) # type: Dict[Voxel, Dict[Voxel, float]]
        # a,b       = find_extrema(distances) # type: Tuple[Voxel, Voxel]
        a, b, voxels_a, voxels_b    = compute_blobs(trk, blob_radius)

        blob_a = Blob(a, voxels_a) # type: Blob
        blob_b = Blob(b, voxels_b)
        blobs = (blob_a, blob_b) if blob_a.E < blob_b.E else (blob_b, blob_a)
        track = Track(voxels_from_track_graph(trk), blobs)
        tc.tracks.append(track)
    return tc
