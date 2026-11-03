from functools import partial
from itertools import combinations

import numpy    as np
import pandas   as pd
import networkx as nx

from networkx import Graph

from .. core.exceptions import NoHits
from .. core.exceptions import NoVoxels
from .. types.symbols   import Contiguity
from .. types.symbols   import HitEnergy
from .. types.ic_types  import Blob
from .. types.df_types  import tracks_type

from typing import Tuple

_XYZ = list("XYZ")
_xyz = list("xyz")


def round_hits_positions_in_place(hits: pd.DataFrame, decimals: int) -> None:
    """
    Rounds the hits positions to `decimals` decimal places to avoid floating
    point comparison issues. The operation is performed inplace to avoid an
    unnecessary copy.

    Parameters
    ----------
    hits : pd.DataFrame
        Hit table containing the ``X``, ``Y``, and ``Z`` columns.
    decimals : int
        Number of decimal places to retain.

    Returns
    -------
    None
    """
    hits.loc[:, _XYZ] = np.round(hits.loc[:, _XYZ], decimals)


def get_track_energy(track: Graph, voxels: pd.DataFrame) -> float:
    """Return the summed voxel energy in a track.

    Parameters
    ----------
    track : networkx.Graph
        Graph whose nodes identify rows in ``voxels``.
    voxels : pd.DataFrame
        Voxel table with an ``e`` energy column.

    Returns
    -------
    float
        Sum of ``e`` over the track's nodes.
    """
    return sum([voxels.loc[vox].e for vox in track.nodes()])


def energy_of_voxels_within_radius(voxels    : pd.DataFrame,
                                   distances : pd.DataFrame,
                                   radius    : float) -> float:
    """
    Sum voxel energy for nodes within a path-distance radius.

    Parameters
    ----------
    voxels : pd.DataFrame
        Voxel table indexed by voxel ID, with an ``e`` energy column.
    distances : pd.DataFrame
        Pairwise path distances from one starting voxel. Must contain
        ``final`` and ``distance`` columns.
    radius : float
        Strict upper bound on path distance, in the same units as the graph
        edge weights.

    Returns
    -------
    float
        Sum of the energies of voxels with path distance less than ``radius``.
    """
    within_radius = distances[distances.distance < radius].final.values
    return sum([voxels.loc[vox].e for vox in within_radius])


def voxelize_hits( hits: pd.DataFrame
                 , voxel_size: np.ndarray
                 , energy_type: HitEnergy = HitEnergy.E
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]: # hits, voxels
    """
    Assign hits to voxels by discretizing their three-dimensional positions.

    Voxel IDs are hashes of integer voxel coordinates relative to the minimum
    hit position. The returned hit table is a copy of the input with a
    ``voxel_id`` column; the voxel table is indexed by voxel ID and stores the
    voxel-centre coordinates and summed energy in its ``e`` column.

    Parameters
    ----------
    hits : pd.DataFrame
        Non-empty hit table containing ``X``, ``Y``, ``Z``, and the selected
        energy column.
    voxel_size : numpy.ndarray, shape (3,)
        Voxel dimensions along the x, y, and z axes.
    energy_type : HitEnergy, optional
        Hit-energy column to sum into each voxel. Defaults to ``HitEnergy.E``.

    Returns
    -------
    hits_with_voxel_ids : pd.DataFrame
        Copy of ``hits`` with a ``voxel_id`` column.
    voxels : pd.DataFrame
        One row per voxel, indexed by voxel ID, with lowercase ``x``, ``y``,
        ``z``, and ``e`` columns.

    Raises
    ------
    NoHits
        If ``hits`` is empty.
    """
    if hits.empty:
        raise NoHits

    energy_type = energy_type.value

    hits  = hits.copy()
    xyz   = hits[_XYZ].values
    lower = xyz.min(axis=0)

    voxel_indices = (xyz - lower) // voxel_size
    voxel_ids     = [hash(tuple(idx)) for idx in voxel_indices]
    hits.insert(hits.shape[1], "voxel_i" , voxel_indices.T[0])
    hits.insert(hits.shape[1], "voxel_j" , voxel_indices.T[1])
    hits.insert(hits.shape[1], "voxel_k" , voxel_indices.T[2])
    hits.insert(hits.shape[1], "voxel_id", voxel_ids)

    # We need to keep only one entry per voxel to compute its position in a
    # simple fashion. We take the chance to compute the total energy as well.
    # +0.5 shifts the voxel position to its center rather than the lower edge
    single = hits.groupby("voxel_id").agg({ "voxel_i"  : "first"
                                          , "voxel_j"  : "first"
                                          , "voxel_k"  : "first"
                                          , energy_type: "sum"
                                          })
    voxels = pd.DataFrame(dict( x = lower[0] + (single.voxel_i + 0.5) * voxel_size[0]
                              , y = lower[1] + (single.voxel_j + 0.5) * voxel_size[1]
                              , z = lower[2] + (single.voxel_k + 0.5) * voxel_size[2]
                              , e = single[energy_type]
                              ), index=single.index)

    # voxel_* are no longer needed, so save some space
    hits.drop(columns="voxel_i voxel_j voxel_k".split(), inplace=True)
    return hits, voxels


def neighbours( va        : pd.Series
              , vb        : pd.Series
              , size      : np.ndarray
              , contiguity: Contiguity = Contiguity.CORNER
              ) -> bool:
    """
    Return whether two voxel centres satisfy the contiguity criterion.

    The Euclidean distance between the centres is normalized by voxel size
    along each axis and compared with ``contiguity.value``.

    Parameters
    ----------
    va, vb : pandas.Series
        Voxel rows containing lowercase ``x``, ``y``, and ``z`` coordinates.
    size : numpy.ndarray, shape (3,)
        Voxel dimensions along the x, y, and z axes.
    contiguity : Contiguity, optional
        Maximum normalized centre-to-centre distance. Defaults to
        ``Contiguity.CORNER``.

    Returns
    -------
    bool
        ``True`` if the normalized distance is less than the contiguity value.
    """
    return np.linalg.norm((va.loc[_xyz].values - vb.loc[_xyz].values) / size) < contiguity.value


def make_track_graphs( voxels     : pd.DataFrame
                     ,  voxel_size: np.ndarray
                     , contiguity : Contiguity = Contiguity.CORNER
                     ) -> Tuple[Graph, ...]:
    """
    Build one weighted graph for each connected component of neighboring voxels.

    Graph nodes are voxel IDs. An edge joins two voxels when their normalized
    centre-to-centre distance is below the selected contiguity value; its
    ``distance`` weight is their Euclidean separation.

    Parameters
    ----------
    voxels : pd.DataFrame
        Voxel table indexed by voxel ID, with lowercase ``x``, ``y``, and
        ``z`` coordinate columns.
    voxel_size : numpy.ndarray, shape (3,)
        Voxel dimensions along the x, y, and z axes.
    contiguity : Contiguity, optional
        Neighbor criterion. Defaults to ``Contiguity.CORNER``.

    Returns
    -------
    tuple of networkx.Graph
        Connected components of the voxel-neighbor graph, each copied into an
        independent graph.

    Notes
    -----
    Voxels are added in index order so graph traversal does not depend on the
    input DataFrame's row order.
    """
    voxel_graph = nx.Graph()
    voxel_ids = sorted(voxels.index)
    voxel_graph.add_nodes_from(voxel_ids)
    for i, j in combinations(voxel_ids, 2):
        vi = voxels.loc[i]
        vj = voxels.loc[j]
        if neighbours(vi, vj, voxel_size, contiguity):
            voxel_graph.add_edge(i, j, distance = np.linalg.norm(vi[_xyz] - vj[_xyz]))

    return tuple( voxel_graph.subgraph(c).copy()
                  for c in nx.connected_components(voxel_graph)
                )


def shortest_paths(track_graph: Graph) -> pd.DataFrame:
    """
    Compute pairwise shortest-path distances in a weighted track graph.

    Parameters
    ----------
    track_graph : networkx.Graph
        Track graph whose edges have a ``distance`` weight.

    Returns
    -------
    pd.DataFrame
        Long-form table with ``initial``, ``final``, and ``distance`` columns.
        Each reachable ordered pair of nodes has one row, including zero-length
        paths from each node to itself.
    """
    distances = dict(nx.all_pairs_dijkstra_path_length(track_graph, weight='distance'))
    distances = ((v1, v2, d) for v1, dmap in distances.items() for v2, d in dmap.items())
    distances = pd.DataFrame(distances, columns="initial final distance".split())
    return distances


def find_extrema(distances: pd.DataFrame) -> Tuple[int, int]:
    """
    Find the most widely separated voxel pair in a track.

    Parameters
    ----------
    distances : pd.DataFrame
        Pairwise path-distance table with ``initial``, ``final``, and
        ``distance`` columns, as returned by :func:`shortest_paths`.

    Returns
    -------
    extreme_id_1 : int
        Starting voxel ID of a pair with maximum shortest-path distance.
    extreme_id_2 : int
        Ending voxel ID of that pair. The IDs are not ordered by energy.

    Raises
    ------
    NoVoxels
        If ``distances`` is empty.
    """
    if distances.empty:
        raise NoVoxels

    idxmax = distances.distance.idxmax()
    v1, v2 = distances.loc[idxmax, "initial final".split()]
    return v1, v2


def get_track_length(distances: pd.DataFrame, v1: int, v2: int) -> float:
    """Return the shortest-path distance between two voxel IDs.

    Parameters
    ----------
    distances : pd.DataFrame
        Pairwise path-distance table with ``initial``, ``final``, and
        ``distance`` columns, as returned by :func:`shortest_paths`.
    v1, v2 : int
        Starting and ending voxel IDs.

    Returns
    -------
    float
        Distance from ``v1`` to ``v2`` along the track graph.
    """
    distances = distances.set_index("initial final".split())
    return distances.loc[(v1, v2), "distance"]


def hits_ave_pos(hits  : pd.DataFrame,
                 etype : HitEnergy = HitEnergy.E) -> np.ndarray:
    """
    Calculate the energy-weighted average hit position.

    If the selected energy sums to zero, use the unweighted mean position.

    Parameters
    ----------
    hits : pd.DataFrame
        Hit table containing ``X``, ``Y``, ``Z``, and the selected energy
        column.
    etype : HitEnergy, optional
        Energy column used as the weights. Defaults to ``HitEnergy.E``.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Weighted mean position in x, y, and z. If the selected energies sum
        to zero, this is the unweighted mean position.
    """
    # catch cases with no weight
    if hits[etype.value].sum() == 0:
        return np.average(hits[_XYZ].values , axis = 0)

    return np.average( hits[_XYZ].values
                     , weights=hits[etype.value].values
                     , axis=0)


def define_blobs(hits        : pd.DataFrame,
                 voxels      : pd.DataFrame,
                 distances   : pd.DataFrame,
                 blob_radius : float,
                 scan_radius : float | None,
                 extreme_id_1: int,
                 extreme_id_2: int,
                 voxel_size  : np.ndarray,
                 energy_type : HitEnergy = HitEnergy.E,
                ) -> Tuple[Blob, Blob]:
    """
    Identify blobs at the two ends of a track.

    Blob centres are the selected-energy-weighted hit positions in the
    endpoint voxels, or in the highest-encapsulating voxels when
    ``scan_radius`` is provided. If the selected energies in a seed voxel sum
    to zero, its unweighted mean hit position is used. Candidate hits must be
    within ``blob_radius`` of a centre and belong to a voxel close enough to
    the seed along the track graph.

    Parameters
    ----------
    hits : pd.DataFrame
        Hits belonging to this track.
    voxels : pd.DataFrame
        Voxel table indexed by voxel ID, with coordinates and summed ``e`` in
        the selected energy definition.
    distances : pd.DataFrame
        Pairwise path distances for this track, as returned by
        :func:`shortest_paths`.
    blob_radius : float
        Spatial radius used to select hits around each blob centre.
    scan_radius : float or None
        If provided, search this path-distance radius from each endpoint for
        the voxel that captures the most energy within ``blob_radius``. If
        ``None``, use the endpoint voxels directly.
    extreme_id_1, extreme_id_2 : int
        Endpoint voxel IDs, typically returned by :func:`find_extrema`.
    voxel_size : numpy.ndarray, shape (3,)
        Voxel dimensions along the x, y, and z axes. The voxel diagonal is used
        to conservatively select candidate voxels by path distance.
    energy_type : HitEnergy, optional
        Energy column used for blob weighting and energy totals. Defaults to
        ``HitEnergy.E``.

    Returns
    -------
    blob_1, blob_2 : Blob
        Blobs associated with ``extreme_id_1`` and ``extreme_id_2``,
        respectively. They are returned in endpoint order, not sorted by
        energy; use :func:`sort_blobs` to order them from lower to higher
        energy.
    """
    if scan_radius is None:
        blob_node_1 = extreme_id_1
        blob_node_2 = extreme_id_2
    else:
        blob_node_1 = find_highest_encapsulating_node(voxels,
                                                      extreme_id_1,
                                                      distances,
                                                      blob_radius,
                                                      scan_radius)

        blob_node_2 = find_highest_encapsulating_node(voxels,
                                                      extreme_id_2,
                                                      distances,
                                                      blob_radius,
                                                      scan_radius)


    blob_pos_1 = hits_ave_pos(hits.loc[hits.voxel_id==blob_node_1], energy_type)
    blob_pos_2 = hits_ave_pos(hits.loc[hits.voxel_id==blob_node_2], energy_type)

    blob_hits_1 = blob_hits(hits, distances, blob_pos_1, blob_radius, blob_node_1, voxel_size)
    blob_hits_2 = blob_hits(hits, distances, blob_pos_2, blob_radius, blob_node_2, voxel_size)

    blob_voxels_1 = blob_hits_1.voxel_id.unique()
    blob_voxels_2 = blob_hits_2.voxel_id.unique()

    blob_1 = Blob(blob_hits_1[energy_type.value].sum(), blob_pos_1, blob_hits_1.index.values, blob_voxels_1, blob_node_1)
    blob_2 = Blob(blob_hits_2[energy_type.value].sum(), blob_pos_2, blob_hits_2.index.values, blob_voxels_2, blob_node_2)
    return blob_1, blob_2


def sort_blobs(blob_1: Blob, blob_2: Blob) -> Tuple[Blob, Blob]:
    """Return the two blobs in ascending energy order.

    If their energies are equal, preserve the input order.

    Parameters
    ----------
    blob_1, blob_2 : Blob
        Blobs to order.

    Returns
    -------
    low_energy_blob, high_energy_blob : Blob
        The lower- and higher-energy blobs, respectively.
    """
    ascending = blob_1.energy <= blob_2.energy
    return (blob_1, blob_2) if ascending else (blob_2, blob_1)


def find_highest_encapsulating_node(voxels     : pd.DataFrame,
                                    extrema_id : int,
                                    distances  : pd.DataFrame,
                                    blob_radius: float,
                                    scan_radius: float) -> int:
    """
    Find the voxel that captures the most energy within a blob radius.

    Candidate voxels are limited to those within ``scan_radius`` of the
    endpoint, using shortest-path distance. The energy captured by each
    candidate is summed over voxels within ``blob_radius`` of that candidate,
    also using shortest-path distance.

    Parameters
    ----------
    voxels : pd.DataFrame
        Voxel table indexed by voxel ID, with an ``e`` energy column.
    extrema_id : int
        Endpoint voxel ID from which the candidate search begins.
    distances : pd.DataFrame
        Pairwise path distances for the track, as returned by
        :func:`shortest_paths`.
    blob_radius : float
        Path-distance radius used to sum captured voxel energy.
    scan_radius : float
        Path-distance radius around ``extrema_id`` in which to search.

    Returns
    -------
    int
        ID of the candidate voxel with the largest captured energy.

    Raises
    ------
    ValueError
        If no voxel lies within ``scan_radius`` of ``extrema_id``.
    """
    distances = distances.set_index("initial")
    d_extrema = distances.loc[extrema_id]
    nodes_within_radius = d_extrema.final.loc[d_extrema.distance < scan_radius].values

    def energy_within_radius(node: int) -> float:
        return energy_of_voxels_within_radius(voxels, distances.loc[node], blob_radius)

    highest_encapsulating_node = max(nodes_within_radius, key = energy_within_radius)
    return highest_encapsulating_node


def blob_hits(hits       : pd.DataFrame,
              distances  : pd.DataFrame,
              center     : np.ndarray,
              radius     : float,
              extreme_id : int,
              voxel_size : np.ndarray
             ) -> pd.DataFrame:
    """Select hits belonging to a blob around a seed voxel.

    Candidate voxels must be less than ``radius`` plus the voxel diagonal from
    the seed by path distance. Selected hits must also be less than ``radius``
    from ``center`` by Euclidean distance.

    Parameters
    ----------
    hits : pd.DataFrame
        Hit table containing ``X``, ``Y``, ``Z``, and ``voxel_id`` columns.
    distances : pd.DataFrame
        Pairwise path-distance table with ``initial``, ``final``, and
        ``distance`` columns, as returned by :func:`shortest_paths`.
    center : numpy.ndarray, shape (3,)
        Blob centre in x, y, and z coordinates.
    radius : float
        Spatial hit-selection radius, in the same units as the coordinates.
    extreme_id : int
        Seed voxel ID for the blob.
    voxel_size : numpy.ndarray, shape (3,)
        Voxel dimensions; the voxel diagonal expands the path-distance
        candidate threshold.

    Returns
    -------
    pd.DataFrame
        Rows from ``hits`` selected for this blob.
    """
    distances = distances.set_index("initial")
    diag      = np.linalg.norm(voxel_size)

    # voxels that might have hits within the required radius
    # Note that loc uses a bracketed number. This is to force pandas to return a
    # DataFrame and not a Series.
    within_radius    = lambda df: df.distance < radius + diag
    candidate_voxels = distances.loc[[extreme_id]].loc[within_radius].final.values
    within_r         = np.linalg.norm(hits[_XYZ].values - center, axis=1) < radius

    # some hits might fall within the radius, but their distance **along the
    # track** (established by the voxel they belong to) might be longer. We want
    # hits from voxels that are connected to the extreme
    sel = hits.voxel_id.isin(candidate_voxels).values & within_r
    return hits.loc[sel]


def assign_blob_labels_inplace(hits     : pd.DataFrame,
                               voxels   : pd.DataFrame,
                               blob_low : Blob,
                               blob_high: Blob
                              ) -> None:
    """Label hit and voxel rows according to blob membership.

    Rows belonging to both blobs receive the ``"highlow"`` label. The input
    tables are modified in place; rows outside the blobs are left unchanged.

    Parameters
    ----------
    hits : pd.DataFrame
        Hit table indexed by hit ID, with a ``blob`` column.
    voxels : pd.DataFrame
        Voxel table indexed by voxel ID, with a ``blob`` column.
    blob_low, blob_high : Blob
        Lower- and higher-energy blobs whose hit and voxel IDs are assigned.

    Returns
    -------
    None
    """
    shared = list(set(blob_low.hit_ids).intersection(set(blob_high.hit_ids)))
    hits.loc[blob_low .hit_ids, "blob"] = "low"
    hits.loc[blob_high.hit_ids, "blob"] = "high"
    hits.loc[           shared, "blob"] = "highlow"

    shared = list(set(blob_low.voxel_ids).intersection(set(blob_high.voxel_ids)))
    voxels.loc[blob_low .voxel_ids, "blob"] = "low"
    voxels.loc[blob_high.voxel_ids, "blob"] = "high"
    voxels.loc[             shared, "blob"] = "highlow"


def make_tracks(hits        : pd.DataFrame,
                voxels      : pd.DataFrame,
                voxel_size  : np.ndarray,
                blob_radius : float,
                scan_radius : float | None,
                contiguity  : Contiguity = Contiguity.CORNER,
                energy_type : HitEnergy  = HitEnergy.E
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Assign track IDs to hits and voxels, blob labels to hits and voxels, and
    summarize each track.

    Disconnected voxel components are ordered by decreasing ``e`` energy and
    numbered from zero. Blob membership is assigned to hits and voxels. Each
    track summary's length is the path distance between its most distant
    voxels. The ``extreme1_*`` and ``extreme2_*`` fields contain the seed
    positions of the higher- and lower-energy blobs, respectively. The
    ``blob1_*`` and ``blob2_*`` fields contain the corresponding blob centres
    in the same order. The ``r_ave`` field is the radius of the average x-y
    position, not the average of the hits' radii.

    Parameters
    ----------
    hits : pd.DataFrame
        Non-empty hit table containing an ``event`` column, coordinates,
        ``voxel_id``, and the selected energy column.
    voxels : pd.DataFrame
        Voxel table indexed by voxel ID, with coordinates and summed ``e`` for
        ``energy_type``.
    voxel_size : numpy.ndarray, shape (3,)
        Voxel dimensions along the x, y, and z axes.
    blob_radius : float
        Spatial radius used to select hits belonging to each blob.
    scan_radius : float or None
        Optional path-distance search radius for choosing blob centres.
        ``None`` uses the track endpoints.
    contiguity : Contiguity, optional
        Neighbor criterion used to build tracks. Defaults to
        ``Contiguity.CORNER``.
    energy_type : HitEnergy, optional
        Hit-energy column used for track and blob energy calculations and
        weighted positions. Defaults to ``HitEnergy.E``.

    Returns
    -------
    hits : pd.DataFrame
        Copy of the hit table with ``track`` and ``blob`` columns. Blob labels
        are ``"low"``, ``"high"``, ``"highlow"``, or ``"none"``.
    voxels : pd.DataFrame
        Copy of the voxel table with ``track`` and ``blob`` columns. Blob
        labels are ``"low"``, ``"high"``, ``"highlow"``, or ``"none"``.
    tracks : pd.DataFrame
        One summary row per track, with columns defined by
        ``tracks_type``.
    """
    # generate empty dataframe
    track_df = pd.DataFrame(columns = list(tracks_type.keys()))

    # generate tracks and sort by energy
    track_graphs = make_track_graphs(voxels, voxel_size, contiguity)
    track_graphs = sorted(track_graphs,
                          key=partial(get_track_energy, voxels = voxels),
                          reverse=True)

    event  = int(hits.event.iloc[0])
    hits   =   hits.assign(track=-1, blob="none") # IMPORTANT: this is a copy
    voxels = voxels.assign(track=-1, blob="none") # IMPORTANT: this is a copy

    for track_no, track in enumerate(track_graphs):
        track_voxels = voxels.loc[list(track.nodes())]
        track_hits   =   hits.loc[hits.voxel_id.isin(track_voxels.index)]
        track_hits_r = np.sqrt(track_hits.X**2 + track_hits.Y**2)

        voxels.loc[track_voxels.index, "track"] = track_no
        hits  .loc[track_hits  .index, "track"] = track_no

        distances            = shortest_paths(track)
        extreme_1, extreme_2 = find_extrema(distances)
        blob_1, blob_2       = define_blobs(track_hits, track_voxels, distances,
                                            blob_radius, scan_radius,
                                            extreme_1, extreme_2,
                                            voxel_size, energy_type)
        blob_low, blob_high  = sort_blobs(blob_1, blob_2)
        assign_blob_labels_inplace(hits, voxels, blob_low, blob_high)

        # Track summary
        numb_of_voxels = len(track_voxels)
        numb_of_hits   = len(track_hits)
        numb_of_tracks = len(track_graphs)
        track_energy   = track_voxels.e.sum()
        track_length   = get_track_length(distances, extreme_1, extreme_2)

        vox_pos_low  = voxels.loc[blob_low .seed, _xyz]
        vox_pos_high = voxels.loc[blob_high.seed, _xyz]
        ave_pos      = hits_ave_pos(track_hits, energy_type)
        ave_r        = np.sum(ave_pos[:2]**2)**0.5

        # energy shared among blobs
        overlap = hits.loc[hits.index.isin(track_hits.index) & (hits.blob == "highlow"), energy_type.value].sum()

        # generate general tracking table
        list_of_vars = [event, track_no, track_energy, track_length,
                        numb_of_voxels, numb_of_hits, numb_of_tracks,
                        *track_hits[_XYZ].min(), track_hits_r.min(),
                        *track_hits[_XYZ].max(), track_hits_r.max(),
                        *ave_pos, ave_r,
                        *vox_pos_high, *vox_pos_low,
                        *blob_high.position, *blob_low.position,
                        blob_high.energy, blob_low.energy, overlap,
                        *voxel_size]
        track_df.loc[track_no] = list_of_vars

    # modify column dtype to match variable type
    track_df = track_df.astype(tracks_type)
    return hits, voxels, track_df


def pop_voxel_inplace(voxels: pd.DataFrame, vox_id: int) -> pd.Series:
    """Remove and return one voxel row, modifying ``voxels`` in place.

    Parameters
    ----------
    voxels : pd.DataFrame
        Voxel table indexed by voxel ID.
    vox_id : int
        ID of the voxel to remove.

    Returns
    -------
    pandas.Series
        Removed voxel row, with its index label set as the Series name.

    Raises
    ------
    KeyError
        If ``vox_id`` is not present in ``voxels``.
    """
    popped = voxels.loc[vox_id]
    voxels.drop(vox_id, inplace=True)
    return popped


def drop_voxel_inplace( hits       : pd.DataFrame
                      , voxels     : pd.DataFrame
                      , voxel_size : np.ndarray
                      , vox_id     : int
                      , e_type     : HitEnergy
                      , contiguity : Contiguity = Contiguity.CORNER
                      ) -> pd.Series:
    """
    Drop one voxel and redistribute its energy among the closest neighboring
    hits.

    The voxel's selected energy is added to the selected energies of the
    neighboring hits at the minimum distance from the energy-weighted
    barycenter of the voxel's hits. The added energy is divided among those
    hits in proportion to their selected energies. The input hit and voxel
    tables are modified in place.

    Parameters
    ----------
    hits : pd.DataFrame
        Hit table containing coordinates, ``voxel_id``, and the selected energy
        column.
    voxels : pd.DataFrame
        Voxel table indexed by voxel ID, with coordinates and summed ``e``.
    voxel_size : numpy.ndarray, shape (3,)
        Voxel dimensions along the x, y, and z axes.
    vox_id : int
        ID of the voxel to drop.
    e_type : HitEnergy
        Hit-energy column to redistribute.
    contiguity : Contiguity, optional
        Neighbor criterion. Defaults to ``Contiguity.CORNER``.

    Returns
    -------
    pandas.Series
        Removed voxel row with its ``e`` value set to NaN. Its hits have their
        selected energy set to NaN and ``voxel_id`` set to zero.
    """
    popped           = pop_voxel_inplace(voxels, vox_id)
    is_neighbour     = [neighbours(popped, voxel, voxel_size, contiguity) for _, voxel in voxels.iterrows()]
    neighbour_voxels = voxels.loc[is_neighbour]
    hits_in_voxel    = hits.loc[hits.voxel_id == vox_id]
    bary_pos = np.average(           hits_in_voxel[_XYZ]
                         , weights = hits_in_voxel[e_type.value]
                         , axis    =  0)

    neighbour_hits = hits.loc[hits.voxel_id.isin(neighbour_voxels.index)]
    distances      = np.linalg.norm(neighbour_hits[_XYZ] - bary_pos, axis=1)
    closest_hits   = neighbour_hits.loc[np.isclose(distances, distances.min())]


    # the energy of the dropped voxel is assigned to the closest hit.
    # However, several hits can be at exactly the same distance (this
    # happens when hits are distributed in a regular pattern). We generalize
    # this behaviour by determining all hits in neighbouring voxels within a
    # minimum distance from the main voxels barycentre position and share
    # the voxel energy among them, proportionally to each hit's energy
    e_type          = e_type.value
    total_closest_e = closest_hits[e_type].sum()
    new_hit_energy  = closest_hits[e_type] * (1 + popped.e/total_closest_e)

    # avoid warnings by creating explicit mask and using iloc
    mask = hits.index.isin(closest_hits.index)
    hits.iloc[mask, hits.columns.get_loc(e_type)] = new_hit_energy.values

    mask = hits.voxel_id == vox_id
    hits.loc[mask.values,     e_type] = np.nan
    hits.loc[mask.values, 'voxel_id'] = 0

    new_vox_energy = hits.groupby("voxel_id")[e_type].sum()
    # remove the hit energy from the popped voxel's hits
    new_vox_energy = new_vox_energy.drop(0)
    voxels.loc[new_vox_energy.index, 'e'] = new_vox_energy

    # set popped voxel energy to nan (we can copy here as there is no in-place requirements)
    popped = popped.copy()
    popped['e'] = np.nan

    return popped


def drop_voxels(hits            : pd.DataFrame,
                voxels          : pd.DataFrame,
                energy_threshold: float,
                voxel_size      : np.ndarray,
                e_type          : HitEnergy,
                min_vxls        : int = 3,
                contiguity      : Contiguity = Contiguity.CORNER
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Recursively drop low-energy endpoint voxels from each track.

    Tracks with fewer than ``min_vxls`` voxels are left unchanged. A candidate
    endpoint is dropped only when its energy is below ``energy_threshold`` and
    it has at least one other neighbor.

    Parameters
    ----------
    hits : pd.DataFrame
        Hit table containing coordinates, ``voxel_id``, and the selected energy
        column.
    voxels : pd.DataFrame
        Voxel table indexed by voxel ID, with coordinates and summed ``e``.
    energy_threshold : float
        Endpoint energy below which a voxel is eligible to be dropped.
    voxel_size : numpy.ndarray, shape (3,)
        Voxel dimensions along the x, y, and z axes.
    e_type : HitEnergy
        Hit-energy column used to identify and redistribute energy.
    min_vxls : int, optional
        Minimum track size for endpoint removal. Defaults to 3.
    contiguity : Contiguity, optional
        Neighbor criterion. Defaults to ``Contiguity.CORNER``.

    Returns
    -------
    modified_hits : pd.DataFrame
        A copy of the original hits where dropped hits are marked as voxel_id=0
        and have NaN selected energy. If no voxels are dropped, this is an
        unchanged copy of the input hits.
    modified_voxels : pd.DataFrame
        Remaining voxel table.
    dropped_voxels : pd.DataFrame
        Removed voxel rows, with their ``e`` values set to NaN. Empty if no
        voxels were dropped.
    """
    dropped         = []
    modified_hits   =   hits.copy()
    modified_voxels = voxels.copy()
    modified        = True
    while modified:
        modified = False
        trks = make_track_graphs(modified_voxels, voxel_size, contiguity)

        for t in trks:
            if len(t.nodes()) < min_vxls:
                continue

            distances = shortest_paths(t)
            for voxel_id in find_extrema(distances):
                extreme = modified_voxels.loc[voxel_id]
                if extreme.e < energy_threshold:
                    # be sure that the voxel to be eliminated has at least one neighbour
                    # beyond itself
                    n_neighbours = sum(neighbours(extreme, v, voxel_size, contiguity) for _, v in modified_voxels.iterrows())
                    if n_neighbours > 1:
                        dropped_voxel = drop_voxel_inplace(modified_hits, modified_voxels, voxel_size, voxel_id, e_type, contiguity)
                        dropped.append(dropped_voxel)
                        modified = True

    # preserve dataframes' column and type structure using a null slice
    dropped = pd.DataFrame(dropped) if dropped else modified_voxels.iloc[:0]

    return modified_hits, modified_voxels, dropped
