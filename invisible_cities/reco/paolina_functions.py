"""Functions for Paolina analysis
JR April 2017
"""
from itertools import count

import numpy as np
import networkx as nx

#vol_min = np.array([-250, -250, -100],dtype=np.int16)  # volume minimum (x,y,z)
#vol_max = np.array([250, 250, 1100],dtype=np.int16)  # volume maximum (x,y,z)
#vox_size = np.array([10, 10, 5],dtype=np.int16)    # voxel size
#blob_radius = 15.                    # blob radius in mm

# voxel object
class Voxel:
    def __init__(self, ID, pos, E, ix, tID):
        self.ID   = ID
        self.pos  = pos
        self.ix   = ix
        self.E    = E
        self.tID  = tID

    def __repr__(self):
        return '<{} {}>'.format(self.__class__.__name__, self.ix.tolist())
    __str__ = __repr__

def build_voxels(hitc, vol_min, vol_max, vox_size):
    """Builds a list of voxels from the specified hit collection.
    """
    #print("Building voxels...")

    # calculate the size of the volume dimensions
    vdim = ((vol_max - vol_min) / vox_size).astype(np.int16)

    # create the voxel array
    voxel_energy = np.zeros(vdim)

    # add the energy of all hits to the voxels
    for hh in hitc:
        dimensions = np.array([hh.X, hh.Y, hh.Z])
        i,j,k = np.clip((dimensions - vol_min) / vox_size,
                        0,
                        vdim - 1).astype(int)

        voxel_energy[i][j][k] += hh.E

    # get lists of the nonzero x,y,z indices and E values
    nzx, nzy, nzz =      np.nonzero(voxel_energy)
    nze =   voxel_energy[np.nonzero(voxel_energy)]

    vid = count()
    return [ Voxel(next(vid),
                   vol_min + np.array(ix) * vox_size,
                   E,        np.array(ix),
                   -1)
             for *ix, E in zip(nzx,nzy,nzz,nze) ]

def distance(v1, v2):
    return np.linalg.norm(v1.pos - v2.pos)

def calc_adj_matrix(voxelc):
    """Creates the adjacency matrix.
        -1         --> self
        0          --> not a neighbor
        (distance) --> voxels are neighbors
    """
    #print("Calculating adj matrix...")

    # use the voxels: determine neighboring voxels by face, edge, or corner connections
    adj_mat = np.zeros([len(voxelc), len(voxelc)])

    # iterate through all voxels, and for each one find the neighboring voxels
    for vv1 in voxelc:
        for vv2 in voxelc:
            maxdelta = max(abs(vv1.ix - vv2.ix))
            if   maxdelta == 0: adj_mat[vv1.ID][vv2.ID] = -1
            elif maxdelta == 1: adj_mat[vv1.ID][vv2.ID] = distance(vv1, vv2)

    return adj_mat

def construct_tracks(voxelc, adj_mat):
    """Constructs all independent tracks given the list of voxels and adjacency matrix.
        Note: assumes the rows and columns of the adjacency matrix correspond
            to the voxels in the order that they are placed in voxelc
    """
    int_to_voxel = dict(zip(count(), voxelc))
    int_graph = nx.from_numpy_matrix(np.clip(adj_mat, 0, None))
    vox_graph = nx.relabel_nodes(int_graph, int_to_voxel)

    # find all independent tracks
    trks = []
    while vox_graph:

        # add all nodes with a path from node 0 to a single track
        tnodes = []
        tid    = 0
        gnodes = vox_graph.nodes()
        for nn in gnodes:
            if nx.has_path(vox_graph, gnodes[0], nn):
                nn.tID = tid
                tnodes.append(nn)
                tid += 1
        tgraph = nx.Graph()
        tgraph.add_nodes_from(tnodes)
        tgraph.add_weighted_edges_from(vox_graph.edges(tnodes, data='weight'))
        trks.append(tgraph)

        # remove these nodes from the original graph and start again
        vox_graph.remove_nodes_from(tnodes)

    # find the largest independent track
    etrk = np.zeros(len(trks))
    for itk,trk in enumerate(trks):
        ee = sum(vv.E for vv in trk)
        etrk[itk] = ee
    itmax = np.argmax(etrk)
    #itmax = np.argmax([trk.number_of_nodes() for trk in trks])
    #print("Found {0} tracks with max having {1} nodes".format(len(trks),trks[itmax].number_of_nodes()))

    return itmax, trks

def calc_dist_mat(tgraph):
    """Calculates the distance matrix and longest shortest path.
    """

    # initialize the distance matrix
    tvoxelc = tgraph.nodes()
    dist_mat = np.zeros([len(tvoxelc), len(tvoxelc)])
    dmax  = -1
    v1max = None
    v2max = None

    # compute the matrix, using only nodes in the specified track
    for n1,vv1 in enumerate(tvoxelc):
        for vv2 in tvoxelc[0:n1]:

            # calculate the length of the shortest path between these two voxels
            dist = nx.astar_path_length(tgraph, vv1, vv2)
            #print("--- Adding dist of {0}".format(dist))
            dist_mat[vv1.tID][vv2.tID] = dist_mat[vv2.tID][vv1.tID] = dist
            if dist > dmax or dmax < 0:
                dmax  = dist
                v1max = vv1
                v2max = vv2

    # compute one longest shortest path
    #print("Longest shortest path is of length {0}".format(dist_mat[v1max.tID][v2max.tID]))
    spath = nx.astar_path(tgraph, v1max, v2max)
    return dist_mat, spath

def construct_blobs(tgraph, dist_mat, spath, blob_radius):
    """Construct the blobs.
    """
    #print("Constructing blobs...")

    tvoxelc = tgraph.nodes()
    Eblob1 = Eblob2 = 0
    ext1 = spath[ 0]
    ext2 = spath[-1]
    #print("found ext1 {0} and ext2 {1}".format(ext1,ext2))

    # add the energies of voxels within 1 blob radius of each extreme
    for vv in tvoxelc:
        dist1 = dist_mat[ext1.tID][vv.tID]
        if dist1 < blob_radius:
            Eblob1 += vv.E
        dist2 = dist_mat[ext2.tID][vv.tID]
        if dist2 < blob_radius:
            Eblob2 += vv.E

    return Eblob1, Eblob2
