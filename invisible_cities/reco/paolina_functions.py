from functools   import reduce
from itertools   import combinations
from enum        import Enum

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

def bounding_box(seq : BHit) -> Sequence[np.ndarray]:
    """Returns two arrays defining the coordinates of a box that bounds the voxels"""
    posns = [x.pos for x in seq]
    return (reduce(np.minimum, posns, MAX3D),
            reduce(np.maximum, posns, MIN3D))


def voxelize_hits(hits             : Sequence[BHit],
                  voxel_dimensions : np.ndarray,
                  strict_voxel_size: bool = False) -> List[Voxel]:
    """1. Hits are enclosed by a bounding box.
       2. Boundix box is discretized (via a hitogramdd).
       3. The energy of all the hits insidex each discreet "voxel" is added.
     """
    if not hits:
        raise NoHits
    hlo, hhi = bounding_box(hits)
    bounding_box_centre = (hhi + hlo) / 2
    bounding_box_size   =  hhi - hlo
    number_of_voxels = np.ceil(bounding_box_size / voxel_dimensions).astype(int)
    number_of_voxels = np.clip(number_of_voxels, a_min=1, a_max=None)
    if strict_voxel_size: half_range = number_of_voxels * voxel_dimensions / 2
    else                : half_range =          bounding_box_size          / 2
    voxel_edges_lo = bounding_box_centre - half_range
    voxel_edges_hi = bounding_box_centre + half_range

    # Expand the voxels a tiny bit, in order to include hits which
    # fall within the margin of error of the voxel bounding box.
    eps = 3e-12 # geometric mean of range that seems to work
    voxel_edges_lo -= eps
    voxel_edges_hi += eps

    hit_positions = np.array([h.pos for h in hits])
    hit_energies  =          [h.E   for h in hits]
    E, edges = np.histogramdd(hit_positions,
                              bins    = number_of_voxels,
                              range   = tuple(zip(voxel_edges_lo, voxel_edges_hi)),
                              weights = hit_energies)

    def centres(a : np.ndarray) -> np.ndarray:
        return (a[1:] + a[:-1]) / 2

    cx, cy, cz = map(centres, edges)
    nz = np.nonzero(E)
    return [Voxel(cx[x], cy[y], cz[z], E[x,y,z], voxel_dimensions) for (x,y,z) in np.stack(nz).T]


class Contiguity(Enum):
    FACE   = 1.2
    EDGE   = 1.5
    CORNER = 1.8


def make_track_graphs(voxels           : Voxel,
                      contiguity       : Contiguity = Contiguity.CORNER) -> Sequence[Graph]:
    """Create a graph where the voxels are the nodes and the edges are any
    pair of neighbour voxel. Two voxels are considered to be
    neighbours if their distance normalized to their size is smaller
    than a contiguity factor.
    """

    def neighbours(va : Voxel, vb : Voxel) -> bool:
        return np.linalg.norm((va.pos - vb.pos) / va.size) < contiguity.value

    voxel_graph = nx.Graph()
    voxel_graph.add_nodes_from(voxels)
    for va, vb in combinations(voxels, 2):
        if neighbours(va, vb):
            voxel_graph.add_edge(va, vb, distance = np.linalg.norm(va.pos - vb.pos))

    return tuple(nx.connected_component_subgraphs(voxel_graph))


def voxels_from_track_graph(track: Graph) -> List[Voxel]:
    """Create and return a list of voxels from a track graph."""
    return track.nodes()


def shortest_paths(track_graph : Graph) -> Dict[Voxel, Dict[Voxel, float]]:
    """Compute shortest path lengths between all nodes in a weighted graph."""
    return nx.all_pairs_dijkstra_path_length(track_graph, weight='distance')


def find_extrema_and_length(distance : Dict[Voxel, Dict[Voxel, float]]) -> Tuple[Voxel, Voxel, float]:
    """Find the extrema and the length of a track, given its dictionary of distances."""
    if not distance:
        raise NoVoxels
    if len(distance) == 1:
        only_voxel = next(iter(distance))
        return (only_voxel, only_voxel, 0.)
    first, last, max_distance = None, None, 0
    for source, target in combinations(distance, 2):
        d = distance[source][target]
        if d > max_distance:
            first, last, max_distance = source, target, d
    return (first, last, max_distance)


def find_extrema(track: Graph) -> Tuple[Voxel, Voxel]:
    """Find the pair of voxels separated by the greatest geometric
      distance along the track.

    """
    distances = shortest_paths(track)
    extremum_a, extremum_b, _ = find_extrema_and_length(distances)
    return extremum_a, extremum_b


def length(track: Graph) -> float:
    """Calculate the length of a track."""
    distances = shortest_paths(track)
    _, _, length = find_extrema_and_length(distances)
    return length


def energy_within_radius(distances : Dict[Voxel, Dict[Voxel, float]], radius : float) -> float:
    return sum(v.E for (v, d) in distances.items() if d < radius)


def voxels_within_radius(distances : Dict[Voxel, Dict[Voxel, float]],
                         radius : float) -> List[Voxel]:
    return [v for (v, d) in distances.items() if d < radius]


def blob_energies(track_graph : Graph, radius : float) -> Tuple[float, float]:
    """Return the energies around the extrema of the track."""
    distances = shortest_paths(track_graph)
    a, b, _ = find_extrema_and_length(distances)
    Ea = energy_within_radius(distances[a], radius)
    Eb = energy_within_radius(distances[b], radius)
    return (Ea, Eb) if Ea < Eb else (Eb, Ea)


def compute_blobs(track_graph : Graph, radius : float) -> Tuple[Voxel, Voxel, List[Voxel], List[Voxel]]:
    """Return the blobs (list of voxels) around the extrema of the track."""
    distances = shortest_paths(track_graph)
    a, b, _ = find_extrema_and_length(distances)
    ba = voxels_within_radius(distances[a], radius)
    bb = voxels_within_radius(distances[b], radius)
    return a, b, ba, bb


def make_tracks(evt_number       : float,
                evt_time         : float,
                voxels           : List[Voxel],
                voxel_dimensions : np.ndarray,
                contiguity       : float = 1,
                blob_radius      : float = 30 * units.mm) -> TrackCollection:
    """Make a track collection."""
    tc = TrackCollection(evt_number, evt_time) # type: TrackCollection
    track_graphs = make_track_graphs(voxels) # type: Sequence[Graph]
#    track_graphs = make_track_graphs(voxels, voxel_dimensions) # type: Sequence[Graph]
    for trk in track_graphs:
        a, b, voxels_a, voxels_b    = compute_blobs(trk, blob_radius)
        blob_a = Blob(a, voxels_a) # type: Blob
        blob_b = Blob(b, voxels_b)
        blobs = (blob_a, blob_b) if blob_a.E < blob_b.E else (blob_b, blob_a)
        track = Track(voxels_from_track_graph(trk), blobs)
        tc.tracks.append(track)
    return tc
