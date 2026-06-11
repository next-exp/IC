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
from .. types.ic_types  import types_dict_tracks

from .. types.ic_types  import NoneType

from typing import Sequence


def round_hits_positions_in_place(hits, decimals):
    """
    Rounds the hits positions to `decimals` decimals to avoid floating point
    comparison issues. The operation is performed inplace to avoid an
    unnecessary copy.
    """
    xyz = "X Y Z".split()
    hits.loc[:, xyz] = np.round(hits.loc[:, xyz], decimals)


def get_track_energy(track, voxels):
    return sum([voxels.loc[vox].e for vox in track.nodes()])


def extract_track_voxels(track  : Graph,
                         voxels : pd.DataFrame):
    '''
    Extract all voxels from track nodes for tracking table

    Parameters
    ----------
    track       :  track graph
    voxels      :  voxels

    Returns
    -------
    dataframe of voxels in the track
    '''
    return voxels.loc[list(track.nodes())]


def voxelize_hits( hits       : pd.DataFrame
                 , voxel_size : np.ndarray
                 , energy_type: HitEnergy = HitEnergy.E
                 ) -> (pd.DataFrame, pd.DataFrame): # hits, voxels
    """
    Assign each hit a voxel by discretizing the 3D space.
    """
    if hits.empty:
        raise NoHits

    energy_type = energy_type.value

    hits  = hits.copy()
    xyz   = hits["X Y Z".split()].values
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
    xyz = list("xyz")
    return np.linalg.norm((va.loc[xyz].values - vb.loc[xyz].values) / size) < contiguity.value


def make_track_graphs( voxels    : pd.DataFrame
                     , voxel_size: np.ndarray
                     , contiguity: Contiguity = Contiguity.CORNER
                     ) -> Sequence[Graph]:
    """
    Create a graph where the voxels are the nodes and the edges are any pair of
    neighbour voxel. Two voxels are considered to be neighbours if their
    distance normalized to their size is smaller than a contiguity factor.
    """
    xyz = list("xyz")

    voxel_graph = nx.Graph()
    voxel_graph.add_nodes_from(voxels.index)
    for i, j in combinations(voxels.index, 2):
        vi = voxels.loc[i]
        vj = voxels.loc[j]
        if neighbours(vi, vj, voxel_size, contiguity):
            voxel_graph.add_edge(i, j, distance = np.linalg.norm(vi[xyz] - vj[xyz]))

    return tuple( voxel_graph.subgraph(c).copy()
                  for c in nx.connected_components(voxel_graph)
                )


def shortest_paths(track_graph: Graph) -> pd.DataFrame:
    """
    Compute shortest path lengths between all nodes in a weighted graph.
    """
    distances = dict(nx.all_pairs_dijkstra_path_length(track_graph, weight='distance'))
    distances = ((v1, v2, d) for v1, dmap in distances.items() for v2, d in dmap.items())
    distances = pd.DataFrame(distances, columns="initial final distance".split())
    return distances


def find_extrema_and_length( track_graph: Graph
                           , voxels     : pd.DataFrame
                           ) -> (int, int, float): # extreme1, extreme2, length
    """
    Find the extrema and the length of a track. The extrema are sorted by energy
    """
    distances = shortest_paths(track_graph)
    if distances.empty:
        raise NoVoxels

    # pandas' indexing methods return series, which are homogeneous in the type
    # of their elements. If we pick up the three at the same time it casts both
    # integers to floats, so we pick them one by one instead
    idxmax = distances.distance.idxmax()
    v1     = distances.iloc[idxmax, 0]
    v2     = distances.iloc[idxmax, 1]
    length = distances.iloc[idxmax, 2]

    return v1, v2, length


def hits_ave_pos(hits  : pd.DataFrame,
                 etype : HitEnergy = HitEnergy.E) -> np.ndarray:
    """
    Calculate the energy-weighted average position of a set of hits
    """
    return np.average( hits[list("XYZ")].values
                     , weights=hits[etype.value].values
                     , axis=0)


def blob_energies_hits_and_centres(track_graph : Graph,
                         hits        : pd.DataFrame,
                         voxels      : pd.DataFrame,
                         radius      : float,
                         extreme_id_1: int,
                         extreme_id_2: int,
                         voxel_size  : np.ndarray,):
    '''
    Extract relevant blob information
    '''

    distances = shortest_paths(track_graph).set_index("initial")
    if len(distances) == 1: # special case, single voxel

        hits_oi = hits[hits.voxel_id == distances.index[0]]

        e_1 = e_2 = hits_oi.E.sum()
        blob_pos_1 = blob_pos_2 =  hits_ave_pos(hits_oi)
        return e_1, e_2, hits_oi, hits_oi, blob_pos_1, blob_pos_2

    diag = np.linalg.norm(voxel_size)

    blob_pos_1 = hits_ave_pos(hits.loc[hits.voxel_id==extreme_id_1])
    blob_pos_2 = hits_ave_pos(hits.loc[hits.voxel_id==extreme_id_2])

    # voxels that might have within within the required radius
    within_radius = lambda df: df.distance < radius + diag
    candidate_voxels_1 = distances.loc[extreme_id_1].loc[within_radius].final.values
    candidate_voxels_2 = distances.loc[extreme_id_2].loc[within_radius].final.values

    within_r_1 = np.linalg.norm(hits[list("XYZ")].values - blob_pos_1, axis=1) < radius
    within_r_2 = np.linalg.norm(hits[list("XYZ")].values - blob_pos_2, axis=1) < radius

    # some hits might fall within the radius, but their distance **along the
    # track** (established by the voxel they belong to) might be longer. We want
    # hits from voxels that are connected to the extreme
    sel_1 = hits.voxel_id.isin(candidate_voxels_1).values & within_r_1
    sel_2 = hits.voxel_id.isin(candidate_voxels_2).values & within_r_2
    sel_both = sel_1 & sel_2
    e_1 = hits.loc[sel_1, "E"].sum()
    e_2 = hits.loc[sel_2, "E"].sum()

    if e_1 > e_2:
        return e_1, e_2, hits.loc[sel_1], hits.loc[sel_2], blob_pos_1, blob_pos_2
    else:
        return e_2, e_1, hits.loc[sel_2], hits.loc[sel_1], blob_pos_2, blob_pos_1



def assign_blobs_inplace(track_graph : Graph,
                         hits        : pd.DataFrame,
                         voxels      : pd.DataFrame,
                         radius      : float,
                         extreme_id_1: int,
                         extreme_id_2: int,
                         voxel_size  : np.ndarray,
                        ):
    """
    Assigns each hit and voxel a label that links them to a blob. The code is:
    - "low"  for the lower energy blob
    - "high" for the higher energy blob
    - "lowhigh" if a hit belongs to both blobs
    - "none" otherwise (not set within this function)
    """
    distances = shortest_paths(track_graph).set_index("initial")
    if len(distances) == 1: # special case
        hits  .loc[:, "blob"] = "lowhigh"
        voxels.loc[:, "blob"] = "lowhigh"
        return
    diag      = np.linalg.norm(voxel_size)

    blob_pos_1 = hits_ave_pos(hits.loc[hits.voxel_id==extreme_id_1])
    blob_pos_2 = hits_ave_pos(hits.loc[hits.voxel_id==extreme_id_2])

    # voxels that might have within within the required radius
    within_radius = lambda df: df.distance < radius + diag
    candidate_voxels_1 = distances.loc[extreme_id_1].loc[within_radius].final.values
    candidate_voxels_2 = distances.loc[extreme_id_2].loc[within_radius].final.values

    within_r_1 = np.linalg.norm(hits[list("XYZ")].values - blob_pos_1, axis=1) < radius
    within_r_2 = np.linalg.norm(hits[list("XYZ")].values - blob_pos_2, axis=1) < radius

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
    hits.loc[sel_both, "blob"] = "lowhigh"

    # not all of the original voxel selection have hits within the radius. We
    # want to keep only those that do.
    voxel_ids_1    = hits.voxel_id.loc[sel_1].unique()
    voxel_ids_2    = hits.voxel_id.loc[sel_2].unique()
    voxel_ids_both = list(set(voxel_ids_1).intersection(set(voxel_ids_2)))
    voxels.loc[voxel_ids_1   , "blob"] = label_1
    voxels.loc[voxel_ids_2   , "blob"] = label_2
    voxels.loc[voxel_ids_both, "blob"] = "lowhigh"


def make_tracks(hits       : pd.DataFrame,
                voxels     : pd.DataFrame,
                voxel_size : np.ndarray,
                blob_radius: float,
                contiguity : Contiguity = Contiguity.CORNER,
                energy_type: HitEnergy  = HitEnergy.E
               ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame): # hits, voxels
    """
    Assign each hit and voxel a track and a blob. Tracks are simply enumerated
    according to the graph algorithm. The code for blob association is
    - "low"  for the lower energy blob
    - "high" for the higher energy blob
    - "none" otherwise
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

    for track_no, track in enumerate(track_graphs):

        # collect relevant information
        track_voxels                      = extract_track_voxels(track, voxels)
        # create hits with radial information
        track_hits                        = hits[hits.voxel_id.isin(track_voxels.index)].assign(R = lambda df: np.sqrt(df.X**2 + df.Y**2))
        numb_of_voxels                    = len(track_voxels)
        numb_of_hits                      = len(track_hits)
        numb_of_tracks                    = len(track_graphs)
        energy                            = track_voxels.e.sum()
        extreme_low, extreme_high, length = find_extrema_and_length(track, voxels)
        extreme_pos1                      = voxels.loc[extreme_low]
        extreme_pos2                      = voxels.loc[extreme_high]
        ave_pos                           = hits_ave_pos(track_hits, energy_type)
        ave_r                             = np.average(track_hits.R,             weights = track_hits.Ep, axis = 0)


        # blob information

        eblob1, eblob2, hits_blob1, hits_blob2, blob_pos1, blob_pos2 = blob_energies_hits_and_centres(track, hits, voxels, blob_radius, extreme_low, extreme_high, voxel_size)

        # calculate overlap
        common_hits = hits_blob1.merge(hits_blob2, how="inner")
        overlap     = common_hits.Ep.sum()


        # generate general tracking table
        list_of_vars = [event, track_no, energy, length,
                        numb_of_voxels, numb_of_hits, numb_of_tracks,
                        track_hits.X.min(), track_hits.Y.min(), track_hits.Z.min(), track_hits.R.min(),
                        track_hits.X.max(), track_hits.Y.max(), track_hits.Z.max(), track_hits.R.max(),
                        *ave_pos, ave_r, *extreme_pos1[['x', 'y', 'z']].tolist(), *extreme_pos2[['x', 'y', 'z']].tolist(),
                        *blob_pos1, *blob_pos2, eblob1, eblob2, overlap,
                        *voxel_size]
        track_df.loc[track_no] = list_of_vars

        hits  .loc[hits.index.isin(track_hits.index),     "track"] = track_no
        voxels.loc[voxels.index.isin(track_voxels.index), "track"] = track_no

    # modify column dtype to match variable type
    track_df = track_df.apply(lambda x: x.astype(types_dict_tracks[x.name]))
    return hits, voxels, track_df


def pop_voxel_inplace(voxels: pd.DataFrame, vox_id: int) -> pd.Series:
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
    Eliminate an individual voxel from a set of voxels and give its energy to
    the hits closest to the barycenter of the eliminated voxel's hits, provided
    that it belongs to a neighbour voxel. The dropped voxel is returned.
    """
    popped           = pop_voxel_inplace(voxels, vox_id)
    is_neighbour     = [neighbours(popped, voxel, voxel_size, contiguity) for _, voxel in voxels.iterrows()]
    neighbour_voxels = voxels.loc[is_neighbour]
    bary_pos = np.average( hits.loc[hits.voxel_id == vox_id][list("XYZ")]
                         , weights = hits.loc[hits.voxel_id == vox_id, e_type.value]
                         , axis    =  0)

    neighbour_hits = hits.loc[hits.voxel_id.isin(neighbour_voxels.index)]
    distances      = np.linalg.norm(neighbour_hits[list("XYZ")] - bary_pos, axis=1)
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
    hits.loc[closest_hits.index, e_type] = new_hit_energy
    hits.loc[hits.voxel_id == vox_id, e_type] = np.nan
    hits.loc[hits.voxel_id == vox_id, 'voxel_id'] = 0

    new_vox_energy = hits.groupby("voxel_id")[e_type].sum()
    # remove the hit energy from the popped voxel's hits
    new_vox_energy = new_vox_energy.drop(0)
    voxels.loc[new_vox_energy.index, "e"] = new_vox_energy.values

    return popped


def drop_voxels(hits            : pd.DataFrame,
                voxels          : pd.DataFrame,
                energy_threshold: float,
                voxel_size      : np.ndarray,
                e_type          : HitEnergy,
                min_vxls        : int = 3,
                contiguity      : Contiguity = Contiguity.CORNER
               ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame): # hits, voxels, dropped voxels
    """
    Find voxels at the end-points of a track and tag them, recursively, if their
    energy is lower than a threshold.
    """

    dropped  = []
    #hits     =   hits.copy() # this isn't modified anywhere, so no need to copy
    voxels   = voxels.copy()
    modified = True
    while modified:
        modified = False
        trks = make_track_graphs(voxels, voxel_size, contiguity)

        for t in trks:
            if len(t.nodes()) < min_vxls:
                continue

            for voxel_id in find_extrema_and_length(t, voxels)[:2]: # skip length
                extreme = voxels.loc[voxel_id]
                if extreme.e < energy_threshold:
                    # be sure that the voxel to be eliminated has at least one neighbour
                    # beyond itself
                    n_neighbours = sum(neighbours(extreme, v, voxel_size, contiguity) for _, v in voxels.iterrows())
                    if n_neighbours > 1:
                        dropped_voxel = drop_voxel_inplace(hits, voxels, voxel_size, voxel_id, e_type, contiguity)
                        dropped.append(dropped_voxel)
                        modified = True

    dropped = pd.concat(dropped) if dropped else pd.DataFrame()
    return hits, voxels, dropped
