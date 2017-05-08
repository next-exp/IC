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
    def __init__(self, ID, X, Y, Z, E, ix, iy, iz, size_x, size_y, size_z, tID):
        self.ID     = ID
        self.X      = X
        self.Y      = Y
        self.Z      = Z
        self.E      = E
        self.ix     = ix
        self.iy     = iy
        self.iz     = iz
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.tID    = tID


def build_voxels(hitc, vol_min, vol_max, vox_size):
    """Builds a list of voxels from the specified hit collection.
    """
    #print("Building voxels...")

    # calculate the size of the volume dimensions
    vdim = ((vol_max - vol_min) / vox_size).astype(np.int16)

    # create the voxel array
    varr = np.zeros(vdim)

    # add the energy of all hits to the voxels
    for hh in hitc:
        ivox = int((hh.X - vol_min[0]) / vox_size[0])
        jvox = int((hh.Y - vol_min[1]) / vox_size[1])
        kvox = int((hh.Z - vol_min[2]) / vox_size[2])

        ivox = np.clip(ivox, 0, vdim[0] - 1)
        jvox = np.clip(jvox, 0, vdim[1] - 1)
        kvox = np.clip(kvox, 0, vdim[2] - 1)

        varr[ivox][jvox][kvox] += hh.E

    # get lists of the nonzero x,y,z indices and E values
    nzx,nzy,nzz = np.nonzero(varr)
    nze = varr[np.nonzero(varr)]

    vid = count()
    return [ Voxel(next(vid),
                   vol_min[0] + ix * vox_size[0],
                   vol_min[1] + iy * vox_size[1],
                   vol_min[2] + iz * vox_size[2],
                   ee, ix, iy, iz,
                   vox_size[0], vox_size[1], vox_size[2],
                   -1)
             for ix,iy,iz,ee in zip(nzx,nzy,nzz,nze) ]

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
            if(vv1.ix == vv2.ix and vv1.iy == vv2.iy and vv1.iz == vv2.iz):
                adj_mat[vv1.ID][vv2.ID] = -1
            elif ((vv1.ix == vv2.ix+1 or vv1.ix == vv2.ix-1 or vv1.ix == vv2.ix) and
                  (vv1.iy == vv2.iy+1 or vv1.iy == vv2.iy-1 or vv1.iy == vv2.iy) and
                  (vv1.iz == vv2.iz+1 or vv1.iz == vv2.iz-1 or vv1.iz == vv2.iz)):
                adj_mat[vv1.ID][vv2.ID] = (np.sqrt((vv2.X - vv1.X) ** 2 +
                                                   (vv2.Y - vv1.Y) ** 2 +
                                                   (vv2.Z - vv1.Z) ** 2 ))

    return adj_mat

def construct_tracks(voxelc, adj_mat):
    """Constructs all independent tracks given the list of voxels and adjacency matrix.
        Note: assumes the rows and columns of the adjacency matrix correspond
            to the voxels in the order that they are placed in voxelc
    """
    #print("Constructing tracks...")

    # add all voxels as nodes to a Graph
    pgraph = nx.Graph()
    pgraph.add_nodes_from(voxelc)

    # add edges connecting each node to its neighbor nodes based on the values in the adjacency matrix
    for nA in voxelc:
        for nB in voxelc:
            ndist = adj_mat[nA.ID][nB.ID]
            if ndist > 0:
                #print("-- Adding edge from {0} to {1} with weighting of {2}".format(nA.ID,nB.ID,ndist))
                pgraph.add_edge(nA, nB, weight=ndist)

    # find all independent tracks
    trks = []
    while pgraph.number_of_nodes() > 0:

        # add all nodes with a path from node 0 to a single track
        tnodes = []
        tid    = 0
        gnodes = pgraph.nodes()
        for nn in gnodes:
            if nx.has_path(pgraph, gnodes[0], nn):
                nn.tID = tid
                tnodes.append(nn)
                tid += 1
        tgraph = nx.Graph()
        tgraph.add_nodes_from(tnodes)
        tgraph.add_weighted_edges_from(pgraph.edges(tnodes, data='weight'))
        trks.append(tgraph)

        # remove these nodes from the original graph and start again
        pgraph.remove_nodes_from(tnodes)

    # find the largest independent track
    etrk = np.zeros(len(trks))
    for itk,trk in enumerate(trks):
        ee = np.sum([vv.E for vv in trk])
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
