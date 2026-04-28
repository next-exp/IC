from itertools   import combinations

import copy

import numpy    as np
import pandas   as pd
import networkx as nx

from networkx           import Graph
from .. evm.event_model import Voxel
from .. core.exceptions import NoHits
from .. core.exceptions import NoVoxels
from .. evm.event_model import BHit
from .. evm.event_model import Track
from .. evm.event_model import Blob
from .. evm.event_model import TrackCollection
from .. core            import system_of_units as units
from .. types.symbols   import Contiguity
from .. types.symbols   import HitEnergy

from typing import Sequence
from typing import List
from typing import Tuple
from typing import Dict

def bounding_box(hits: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the (min, max) coordinates of the axis-aligned bounding box
    enclosing the hits in X, Y, Z.
    """
    xyz = hits["X Y Z".split()].values
    return xyz.min(axis=0), xyz.max(axis=0)

def round_hits_positions_in_place(hits, decimals):
    """
    Rounds the hits positions to `decimals` decimals to avoid floating point
    comparison issues. The operation is performed inplace to avoid an
    unnecessary copy.
    """
    hits.loc[:, "X Y Z".split()] = np.round(hits.loc[:, "X Y Z".split()], decimals)

def voxelize_hits(hits             : pd.DataFrame,
                  voxel_dimensions : np.ndarray,
                  strict_voxel_size: bool = False,
                  energy_type      : HitEnergy = HitEnergy.E) -> List[Voxel]:
    # 1. Find bounding box of all hits.
    # 2. Allocate hits to regular sub-boxes within bounding box, using histogramdd.
    # 3. Calculate voxel energies by summing energies of hits within each sub-box.
    if hits.empty:
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

    xyz = hits.loc[:, "X Y Z".split()].values
    E, edges = np.histogramdd(xyz,
                              bins    = number_of_voxels,
                              range   = tuple(zip(voxel_edges_lo, voxel_edges_hi)),
                              weights = hits[energy_type.value])

    def centres(a : np.ndarray) -> np.ndarray:
        return (a[1:] + a[:-1]) / 2
    def   sizes(a : np.ndarray) -> np.ndarray:
        return  a[1:] - a[:-1]

    (   cx,     cy,     cz) = map(centres, edges)
    size_x, size_y, size_z  = map(sizes  , edges)

    # find the bins where hits fall into
    # numpy.histogramdd() uses [,) intervals...
    # ...except for the last one, which is [,]: hits on the last edge,
    # if any, must fall into the last bin
    # @gonzaponte: This should not happen, as we are increasing the voxel
    # dimensions a few lines above
    hits_indices = np.array([ np.clip(np.digitize(coords, bins, right=False) - 1, 0, n-1)
                              for coords, bins, n in zip(xyz.T, edges, number_of_voxels)
                            ]).T

    voxels = []
    true_dimensions = np.array([size_x[0], size_y[0], size_z[0]])
    for indices_voxel in np.stack(np.nonzero(E)).T:
        x, y, z     = indices_voxel
        indices     = np.where(np.all(hits_indices == indices_voxel, axis=1))[0]
        hits_in_bin = hits.iloc[indices]

        voxels.append(Voxel(cx[x], cy[y], cz[z], E[x,y,z], true_dimensions, hits_in_bin, energy_type))

    return voxels


def neighbours(va : Voxel, vb : Voxel, contiguity : Contiguity = Contiguity.CORNER) -> bool:
    return np.linalg.norm((va.pos - vb.pos) / va.size) < contiguity.value


def make_track_graphs(voxels           : Sequence[Voxel],
                      contiguity       : Contiguity = Contiguity.CORNER) -> Sequence[Graph]:
    """Create a graph where the voxels are the nodes and the edges are any
    pair of neighbour voxel. Two voxels are considered to be
    neighbours if their distance normalized to their size is smaller
    than a contiguity factor.
    """

    voxel_graph = nx.Graph()
    voxel_graph.add_nodes_from(voxels)
    for va, vb in combinations(voxels, 2):
        if neighbours(va, vb, contiguity):
            voxel_graph.add_edge(va, vb, distance = np.linalg.norm(va.pos - vb.pos))

    return tuple(connected_component_subgraphs(voxel_graph))


def connected_component_subgraphs(G):
    return (G.subgraph(c).copy() for c in nx.connected_components(G))


def voxels_from_track_graph(track: Graph) -> List[Voxel]:
    """Create and return a list of voxels from a track graph."""
    return track.nodes()


def shortest_paths(track_graph : Graph) -> Dict[Voxel, Dict[Voxel, float]]:
    """Compute shortest path lengths between all nodes in a weighted graph."""
    def voxel_pos(x):
        return x[0].pos.tolist()

    distances = dict(nx.all_pairs_dijkstra_path_length(track_graph, weight='distance'))

    # sort the output so the result is reproducible
    distances = { v1 : {v2:d for v2, d in sorted(dmap.items(), key=voxel_pos)}
                  for v1, dmap in sorted(distances.items(), key=voxel_pos)}
    return distances



def find_extrema_and_length(distance : Dict[Voxel, Dict[Voxel, float]]) -> Tuple[Voxel, Voxel, float]:
    """Find the extrema and the length of a track, given its dictionary of distances."""
    if not distance:
        raise NoVoxels
    if len(distance) == 1:
        only_voxel = next(iter(distance))
        return (only_voxel, only_voxel, 0.)
    first, last, max_distance = None, None, 0
    for (voxel1, dist_from_voxel_1_to), (voxel2, _) in combinations(distance.items(), 2):
        d = dist_from_voxel_1_to[voxel2]
        if d > max_distance:
            first, last, max_distance = voxel1, voxel2, d
    return first, last, max_distance


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


def energy_of_voxels_within_radius(distances : Dict[Voxel, float], radius : float) -> float:
    return sum(v.E for (v, d) in distances.items() if d < radius)


def voxels_within_radius(distances : Dict[Voxel, float],
                         radius : float) -> List[Voxel]:
    return [v for (v, d) in distances.items() if d < radius]


def blob_centre(voxel: Voxel) -> Tuple[float, float, float]:
    """Calculate the blob position, starting from the end-point voxel."""
    positions = voxel.hits["X Y Z".split()]
    energies  = voxel.hits[voxel.Etype]
    if sum(energies):
        bary_pos = np.average(positions, weights=energies, axis=0)
    # Consider the case where voxels are built without associated hits
    else:
        bary_pos = voxel.pos

    return bary_pos


def hits_in_blob(track_graph : Graph,
                 radius      : float,
                 extreme     : Voxel) -> Sequence[BHit]:
    """Returns the hits that belong to a blob."""
    distances         = shortest_paths(track_graph)
    dist_from_extreme = distances[extreme]
    blob_pos          = blob_centre(extreme)
    diag              = np.linalg.norm(extreme.size)

    blob_hits = []
    # First, consider only voxels at a certain distance from the end-point, along the track.
    # We allow for 1 extra contiguity, because this distance is calculated between
    # the centres of the voxels, and not the hits. In the second step we will refine the
    # selection, using the euclidean distance between the blob position and the hits.
    for v in track_graph.nodes():
        voxel_distance = dist_from_extreme[v]
        if voxel_distance < radius + diag:
            hit_distances = np.linalg.norm(v.hits["X Y Z".split()] - blob_pos, axis=1)
            blob_hits.append(v.hits.loc[hit_distances < radius])

    return pd.concat(blob_hits, ignore_index=True)


def blob_energies_hits_and_centres(track_graph : Graph, radius : float) -> Tuple[float, float, Sequence[BHit], Sequence[BHit], Tuple[float, float, float], Tuple[float, float, float]]:
    """Return the energies, the hits and the positions of the blobs.
       For each pair of observables, the one of the blob of largest energy is returned first."""
    distances = shortest_paths(track_graph)
    a, b, _   = find_extrema_and_length(distances)
    ha = hits_in_blob(track_graph, radius, a)
    hb = hits_in_blob(track_graph, radius, b)

    voxels = list(track_graph.nodes())
    e_type = voxels[0].Etype

    # Consider the case where voxels are built without associated hits
    some_hits = len(ha) or len(hb)
    Ea = ha[e_type].sum() if some_hits else energy_of_voxels_within_radius(distances[a], radius)
    Eb = hb[e_type].sum() if some_hits else energy_of_voxels_within_radius(distances[b], radius)

    ca = blob_centre(a)
    cb = blob_centre(b)

    if Eb > Ea:
        return (Eb, Ea, hb, ha, cb, ca)
    else:
        return (Ea, Eb, ha, hb, ca, cb)


def blob_energies(track_graph : Graph, radius : float) -> Tuple[float, float]:
    """Return the energies around the extrema of the track.
       The largest energy is returned first."""
    E1, E2, _, _, _, _ = blob_energies_hits_and_centres(track_graph, radius)

    return E1, E2


def blob_energies_and_hits(track_graph : Graph, radius : float) -> Tuple[float, float, Sequence[BHit], Sequence[BHit]]:
    """Return the energies and the hits around the extrema of the track.
       The largest energy is returned first, as well as its hits."""
    E1, E2, h1, h2, _, _ = blob_energies_hits_and_centres(track_graph, radius)

    return (E1, E2, h1, h2)


def blob_centres(track_graph : Graph, radius : float) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Return the positions of the blobs.
       The blob of largest energy is returned first."""
    _, _, _, _, c1, c2 = blob_energies_hits_and_centres(track_graph, radius)

    return (c1, c2)


def make_tracks(evt_number       : float,
                evt_time         : float,
                voxels           : List[Voxel],
                voxel_dimensions : np.ndarray,
                contiguity       : Contiguity = Contiguity.CORNER,
                blob_radius      : float = 30 * units.mm,
                energy_type      : HitEnergy = HitEnergy.E) -> TrackCollection:
    """Make a track collection."""
    tc = TrackCollection(evt_number, evt_time) # type: TrackCollection
    track_graphs = make_track_graphs(voxels, contiguity) # type: Sequence[Graph]
    for trk in track_graphs:
        energy_a, energy_b, hits_a, hits_b = blob_energies_and_hits(trk, blob_radius)
        a, b                               = blob_centres(trk, blob_radius)
        blob_a = Blob(a, hits_a, blob_radius, energy_type) # type: Blob
        blob_b = Blob(b, hits_b, blob_radius, energy_type)
        blobs = (blob_a, blob_b)
        track = Track(voxels_from_track_graph(trk), blobs)
        tc.tracks.append(track)
    return tc


def drop_end_point_voxels(voxels           : Sequence[Voxel],
                          energy_threshold : float,
                          min_vxls         : int = 3,
                          contiguity       : Contiguity = Contiguity.CORNER) -> Sequence[Voxel]:
    """Eliminate voxels at the end-points of a track, recursively,
       if their energy is lower than a threshold. Returns 1 if the voxel
       has been deleted succesfully and 0 otherwise."""

    xyz    = "X Y Z".split()
    e_type = voxels[0].Etype

    def drop_voxel(voxels: Sequence[Voxel], the_vox: Voxel, contiguity: Contiguity = Contiguity.CORNER):
        """Eliminate an individual voxel from a set of voxels and give its energy to the hit
           that is closest to the barycenter of the eliminated voxel hits, provided that it
           belongs to a neighbour voxel."""
        the_neighbour_voxels = [v for v in voxels if neighbours(the_vox, v, contiguity)]

        #if there are no hits associated to voxels the pos will be an empty list
        if len(the_vox.hits) == 0:
            min_dist  = min(np.linalg.norm(the_vox.pos-v.pos) for v in the_neighbour_voxels)
            min_v     = [v for v in the_neighbour_voxels
                           if np.isclose(np.linalg.norm(the_vox.pos-v.pos), min_dist)]

            ### add dropped voxel energy to closest voxels, proportional to the  voxels energy
            sum_en_v = sum(v.E for v in min_v)
            for v in min_v:
                v.E += the_vox.E/sum_en_v * v.E
            return

        bary_pos = np.average(           the_vox.hits[xyz]
                             , weights = the_vox.hits[e_type]
                             , axis    =  0)

        # the energy of the dropped voxel is assigned to the closest hit.
        # However, several hits be at exactly the same distance (this happens
        # when hits are distributed in a regular pattern). We generalize this
        # behaviour by determining all hits in neighbouring voxels within a
        # minimum distance from the main voxels barycentre position and share
        # the voxel energy among them, proportionally to each hit's energy
        distance_to_bary = lambda hs: np.linalg.norm(hs[xyz] - bary_pos, axis=1)
        distances        = [(v, distance_to_bary(v.hits)) for v in the_neighbour_voxels]
        min_dist         = min(ds.min() for (_, ds) in distances)

        hits_at_min_dist = lambda ds: np.argwhere(np.isclose(ds, min_dist)).flatten()
        min_hits         = [(v, hits_at_min_dist(ds)) for (v, ds) in distances]
        min_hits         = [pair for pair in min_hits if pair[1].size]

        total_closest_e = sum(v.hits.iloc[idx][e_type].sum() for (v, idx) in min_hits)
        for (v, row_idx) in min_hits:
            col_idx        = v.hits.columns.get_loc(e_type)
            current_energy = v.hits.iloc[row_idx, col_idx].values
            v.hits.iloc[row_idx, col_idx] = current_energy * (1 + the_vox.E/total_closest_e)

        # recompute voxel energy because we have modified the hits it contains
        for v in voxels:
            v.E = v.hits[e_type].sum()

    def set_nan_energies(voxel):
        voxel.E = np.nan
        voxel.hits.loc[:, e_type] = np.nan

    mod_voxels     = copy.deepcopy(voxels)
    dropped_voxels = []

    modified = True
    while modified:
        modified = False
        trks = make_track_graphs(mod_voxels, contiguity)

        for t in trks:
            if len(t.nodes()) < min_vxls:
                continue

            for extreme in find_extrema(t):
                if extreme.E < energy_threshold:
                    ### be sure that the voxel to be eliminated has at least one neighbour
                    ### beyond itself
                    n_neighbours = sum(neighbours(extreme, v, contiguity) for v in mod_voxels)
                    if n_neighbours > 1:
                        mod_voxels    .remove(extreme)
                        dropped_voxels.append(extreme)
                        drop_voxel(mod_voxels, extreme)
                        set_nan_energies(extreme)
                        modified = True

    return mod_voxels, dropped_voxels


def get_track_energy(track):
    return sum([vox.Ehits for vox in track.nodes()])
