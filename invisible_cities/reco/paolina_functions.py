from itertools   import combinations
from functools   import partial

import numpy    as np
import pandas   as pd
import networkx as nx

from networkx import Graph

from .. core.exceptions import NoHits
from .. core.exceptions import NoVoxels
from .. types.symbols   import Contiguity
from .. types.symbols   import HitEnergy
from .. types.ic_types  import Blob
from .. types.ic_types  import types_dict_tracks
from .. types.ic_types  import NoneType

from typing import Sequence
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
    """
    voxel_graph = nx.Graph()
    voxel_graph.add_nodes_from(voxels.index)
    for i, j in combinations(voxels.index, 2):
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


def find_extrema_and_length(distances: pd.DataFrame) -> Tuple[int, int, float]:
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
    length : float
        Maximum shortest-path distance between the returned voxels.

    Raises
    ------
    NoVoxels
        If ``distances`` is empty.
    """
    if distances.empty:
        raise NoVoxels

    # pandas' indexing methods return series, which are homogeneous in the type
    # of their elements. If we pick up the three at the same time it casts both
    # integers to floats, so we pick them one by one instead
    idxmax = distances.distance.idxmax()
    v1     = distances.initial .loc[idxmax]
    v2     = distances.final   .loc[idxmax]
    length = distances.distance.loc[idxmax]

    return v1, v2, length


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
        Weighted mean position in x, y, and z order.
    """
    # catch cases with no weight
    if hits[etype.value].sum() == 0:
        return np.average(hits[_XYZ].values , axis = 0)

    return np.average( hits[_XYZ].values
                     , weights=hits[etype.value].values
                     , axis=0)


def find_blobs(hits        : pd.DataFrame,
               voxels      : pd.DataFrame,
               distances   : pd.DataFrame,
               blob_radius : float,
               scan_radius : float | None,
               extreme_id_1: int,
               extreme_id_2: int,
               voxel_size  : np.ndarray,
               energy_type : HitEnergy = HitEnergy.E
              ) -> Tuple[Blob, Blob]:
    """
    Identify the high- and low-energy blobs at the ends of a track.

    Blob centres are the energy-weighted hit positions at the endpoint voxels,
    or at the highest-encapsulating voxels when ``scan_radius`` is provided.
    Candidate hits must be within ``blob_radius`` of a centre and connected to
    its voxel by a sufficiently short path through the track.

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
        Endpoint voxel IDs, typically returned by
        :func:`find_extrema_and_length`.
    voxel_size : numpy.ndarray, shape (3,)
        Voxel dimensions along the x, y, and z axes. The voxel diagonal is used
        to conservatively select candidate voxels by path distance.
    energy_type : HitEnergy, optional
        Energy column used for blob weighting and energy totals. Defaults to
        ``HitEnergy.E``.

    Returns
    -------
    high_energy_blob : Blob
        Blob with the larger selected-energy sum.
    low_energy_blob : Blob
        Blob with the smaller selected-energy sum.

    Notes
    -----
    For a one-voxel track, both return values refer to the same blob and
    include all hits in that voxel.
    """
    if len(distances) == 1: # special case, one voxel
        blob = Blob(hits[energy_type.value].sum(),
                    hits_ave_pos(hits, energy_type),
                    hits.index.values)
        return blob, blob


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

    blob_pos_1 = hits_ave_pos(hits.loc[hits.voxel_id == blob_node_1], energy_type)
    blob_pos_2 = hits_ave_pos(hits.loc[hits.voxel_id == blob_node_2], energy_type)

    # voxels that might have been within the required radius
    distances     = distances.set_index("initial")
    diag          = np.linalg.norm(voxel_size)
    within_radius = lambda df: df.distance < blob_radius + diag
    candidate_voxels_1 = distances.loc[blob_node_1].loc[within_radius].final.values
    candidate_voxels_2 = distances.loc[blob_node_2].loc[within_radius].final.values

    within_r_1 = np.linalg.norm(hits[_XYZ].values - blob_pos_1, axis=1) < blob_radius
    within_r_2 = np.linalg.norm(hits[_XYZ].values - blob_pos_2, axis=1) < blob_radius

    # Some hits might be geometrically close to the blob centre but far from it
    # **along the track**. Keep only hits from voxels near the centre node.
    sel_1 = hits.voxel_id.isin(candidate_voxels_1).values & within_r_1
    sel_2 = hits.voxel_id.isin(candidate_voxels_2).values & within_r_2

    hits1 = hits.loc[sel_1]
    hits2 = hits.loc[sel_2]
    blob1 = Blob(hits1[energy_type.value].sum(), blob_pos_1, hits1.index.values)
    blob2 = Blob(hits2[energy_type.value].sum(), blob_pos_2, hits2.index.values)

    if blob1.energy > blob2.energy:
        return blob1, blob2
    else:
        return blob2, blob1


def find_highest_encapsulating_node(voxels       : pd.DataFrame,
                                    extrema_id   : int,
                                    distances    : pd.DataFrame,
                                    blob_radius  : float,
                                    scan_radius  : float) -> int:
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


def assign_blobs_inplace(track_graph : Graph,
                         hits        : pd.DataFrame,
                         voxels      : pd.DataFrame,
                         radius      : float,
                         extreme_id_1: int,
                         extreme_id_2: int,
                         voxel_size  : np.ndarray,
                        ) -> None:
    """
    Label hits and voxels according to their membership in the two blobs.

    The ``blob`` columns of ``hits`` and ``voxels`` are modified in place.
    Existing labels are left unchanged for rows outside either blob.

    Parameters
    ----------
    track_graph : networkx.Graph
        Track graph whose nodes are voxel IDs and whose edges have a
        ``distance`` weight.
    hits : pd.DataFrame
        Hits belonging to the track, including a ``voxel_id`` column.
    voxels : pd.DataFrame
        Voxel table indexed by voxel ID.
    radius : float
        Spatial radius used to select hits around each endpoint.
    extreme_id_1, extreme_id_2 : int
        Endpoint voxel IDs.
    voxel_size : numpy.ndarray, shape (3,)
        Voxel dimensions along the x, y, and z axes.

    Returns
    -------
    None

    Notes
    -----
    Blob labels are ``"low"`` and ``"high"`` for the lower- and higher-energy
    blobs, ``"highlow"`` for shared membership, and ``"none"`` for hits not
    assigned by this function. The caller is responsible for initializing
    labels to ``"none"`` if that is the desired default.
    """
    distances = distances.set_index("initial")
    if len(distances) == 1: # special case
        hits  .loc[:, "blob"] = "highlow"
        voxels.loc[:, "blob"] = "highlow"
        return
    diag      = np.linalg.norm(voxel_size)

    blob_pos_1 = hits_ave_pos(hits.loc[hits.voxel_id==extreme_id_1])
    blob_pos_2 = hits_ave_pos(hits.loc[hits.voxel_id==extreme_id_2])

    # voxels that might have within within the required radius
    within_radius = lambda df: df.distance < radius + diag
    candidate_voxels_1 = distances.loc[extreme_id_1].loc[within_radius].final.values
    candidate_voxels_2 = distances.loc[extreme_id_2].loc[within_radius].final.values

    within_r_1 = np.linalg.norm(hits[_XYZ].values - blob_pos_1, axis=1) < radius
    within_r_2 = np.linalg.norm(hits[_XYZ].values - blob_pos_2, axis=1) < radius

    # some hits might fall within the radius, but their distance **along the
    # track** (established by the voxel they belong to) might be longer. We want
    # hits from voxels that are connected to the extreme
    sel_1 = hits.voxel_id.isin(candidate_voxels_1).values & within_r_1
    sel_2 = hits.voxel_id.isin(candidate_voxels_2).values & within_r_2
    sel_both = sel_1 & sel_2
    e_1 = hits.loc[sel_1, "E"].sum()
    e_2 = hits.loc[sel_2, "E"].sum()

    label_1 = "low"  if e_1 <= e_2 else "high"
    label_2 = "high" if e_1 <= e_2 else "low"
    hits.loc[sel_1   , "blob"] = label_1
    hits.loc[sel_2   , "blob"] = label_2
    hits.loc[sel_both, "blob"] = "highlow"

    # not all of the original voxel selection have hits within the radius. We
    # want to keep only those that do.
    voxel_ids_1    = hits.voxel_id.loc[sel_1].unique()
    voxel_ids_2    = hits.voxel_id.loc[sel_2].unique()
    voxel_ids_both = list(set(voxel_ids_1).intersection(set(voxel_ids_2)))
    voxels.loc[voxel_ids_1   , "blob"] = label_1
    voxels.loc[voxel_ids_2   , "blob"] = label_2
    voxels.loc[voxel_ids_both, "blob"] = "highlow"


def make_tracks(hits        : pd.DataFrame,
                voxels      : pd.DataFrame,
                voxel_size  : np.ndarray,
                blob_radius : float,
                scan_radius : float | None,
                contiguity  : Contiguity = Contiguity.CORNER,
                energy_type : HitEnergy  = HitEnergy.E
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Assign track IDs to hits and voxels, blob labels to hits, and summarize
    each track.

    Disconnected voxel components are ordered by decreasing ``e`` energy and
    numbered from zero. Blob membership is assigned to hits; the returned
    voxel table is tagged with track IDs.

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
        Copy of the voxel table with a ``track`` column.
    tracks : pd.DataFrame
        One summary row per track, with columns defined by
        ``types_dict_tracks``.
    """
    # generate empty dataframe
    track_df = pd.DataFrame(columns = list(types_dict_tracks.keys()))

    # generate tracks and sort by energy
    track_graphs = make_track_graphs(voxels, voxel_size, contiguity)
    track_graphs = sorted(track_graphs,
                          key=partial(get_track_energy, voxels = voxels),
                          reverse=True)

    event  = int(hits.event.iloc[0])
    hits   = hits.copy()
    voxels = voxels.copy()
    hits  .insert(hits  .shape[1], "track",  9999)
    voxels.insert(voxels.shape[1], "track",  9999)
    hits  .insert(hits  .shape[1],  "blob", "none")

    for track_no, track in enumerate(track_graphs):
        distances = shortest_paths(track)

        # collect relevant information
        track_voxels                 = voxels.loc[list(track.nodes())]
        # create hits with radial information
        track_hits                   = hits[hits.voxel_id.isin(track_voxels.index)].assign(R = lambda df: np.sqrt(df.X**2 + df.Y**2))
        numb_of_voxels               = len(track_voxels)
        numb_of_hits                 = len(track_hits)
        numb_of_tracks               = len(track_graphs)
        energy                       = track_voxels.e.sum()
        extreme_1, extreme_2, length = find_extrema_and_length(distances)
        pos_1                        = voxels.loc[extreme_1]
        pos_2                        = voxels.loc[extreme_2]
        ave_pos                      = hits_ave_pos(track_hits, energy_type)
        ave_r                        = np.average(track_hits.R,
                                                  weights = track_hits[energy_type.value],
                                                  axis = 0)

        # blob information
        blob_high, blob_low = find_blobs(track_hits, voxels, distances,
                                         blob_radius, scan_radius,
                                         extreme_1, extreme_2,
                                         voxel_size, energy_type)

        # mark hits as being in low or high blob
        in_high = hits.index.isin(blob_high.hit_ids)
        in_low  = hits.index.isin(blob_low .hit_ids)

        hits.loc[in_high, 'blob']          = 'high'
        hits.loc[in_low , 'blob']          = 'low'
        hits.loc[in_high & in_low, 'blob'] = 'highlow'

        # energy shared among blobs
        overlap = hits.loc[in_high & in_low, energy_type.value].sum()

        # generate general tracking table
        list_of_vars = [event, track_no, energy, length,
                        numb_of_voxels, numb_of_hits, numb_of_tracks,
                        track_hits.X.min(), track_hits.Y.min(), track_hits.Z.min(), track_hits.R.min(),
                        track_hits.X.max(), track_hits.Y.max(), track_hits.Z.max(), track_hits.R.max(),
                        *ave_pos, ave_r, *pos_1[_xyz].tolist(), *pos_2[_xyz].tolist(),
                        *blob_high.position, *blob_low.position, blob_high.energy, blob_low.energy, overlap,
                        *voxel_size]
        track_df.loc[track_no] = list_of_vars

        hits  .loc[hits  .index.isin(track_hits  .index), "track"] = track_no
        voxels.loc[voxels.index.isin(track_voxels.index), "track"] = track_no

    # modify column dtype to match variable type
    track_df = track_df.apply(lambda x: x.astype(types_dict_tracks[x.name]))
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
    Drop one voxel and redistribute its energy among the closest neighbor hits.

    The voxel's energy is shared among all neighboring hits at the minimum
    distance from the energy-weighted barycenter of the voxel's hits, in
    proportion to those hits' selected energies. The input hit and voxel tables
    are modified in place.

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
        Removed voxel row with its ``e`` value set to NaN.
    """
    popped           = pop_voxel_inplace(voxels, vox_id)
    is_neighbour     = [neighbours(popped, voxel, voxel_size, contiguity) for _, voxel in voxels.iterrows()]
    neighbour_voxels = voxels.loc[is_neighbour]
    bary_pos = np.average( hits.loc[hits.voxel_id == vox_id, _XYZ]
                         , weights = hits.loc[hits.voxel_id == vox_id, e_type.value]
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
    hits.iloc[mask.values, hits.columns.get_loc(e_type)]     = np.nan
    hits.iloc[mask.values, hits.columns.get_loc('voxel_id')] = 0

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
    it has at least one other neighbor. The input hit and voxel tables are
    modified in place.

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
    dropped_hits : pd.DataFrame
        Hits belonging to dropped voxels, marked with ``voxel_id == 0`` and
        NaN selected energy. Empty if no voxels were dropped.
    voxels : pd.DataFrame
        Remaining voxel table.
    dropped_voxels : pd.DataFrame
        Removed voxel rows, with their ``e`` values set to NaN. Empty if no
        voxels were dropped.
    """

    dropped  = []
    #hits     =   hits.copy() # this isn't modified anywhere, so no need to copy
    modified = True
    while modified:
        modified = False
        trks = make_track_graphs(voxels, voxel_size, contiguity)

        for t in trks:
            if len(t.nodes()) < min_vxls:
                continue

            distances = shortest_paths(t)
            for voxel_id in find_extrema_and_length(distances)[:2]: # skip length
                extreme = voxels.loc[voxel_id]
                if extreme.e < energy_threshold:
                    # be sure that the voxel to be eliminated has at least one neighbour
                    # beyond itself
                    n_neighbours = sum(neighbours(extreme, v, voxel_size, contiguity) for _, v in voxels.iterrows())
                    if n_neighbours > 1:
                        dropped_voxel = drop_voxel_inplace(hits, voxels, voxel_size, voxel_id, e_type, contiguity)
                        dropped.append(dropped_voxel)
                        modified = True

    dropped = pd.DataFrame(dropped) if dropped else pd.DataFrame()
    if not dropped.empty:
        dropped_hits = hits[hits.voxel_id == 0]
    else:
        dropped_hits = pd.DataFrame([])

    return dropped_hits, voxels, dropped
