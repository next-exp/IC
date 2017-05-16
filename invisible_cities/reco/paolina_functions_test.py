from collections import namedtuple
import numpy as np

from pytest import fixture

from . dst_io            import Hit, HitCollection
from . paolina_functions import build_voxels
from . paolina_functions import calc_adj_matrix
from . paolina_functions import construct_tracks
from . paolina_functions import calc_dist_mat
from . paolina_functions import construct_blobs


Data = namedtuple('TestData',
                  'vol_min, vol_max, vox_size, blob_radius, hitc')

@fixture(scope='session')
def data():
    vol_min  = np.array([-250, -250, -100], dtype=np.int16) # volume minimum (x,y,z)
    vol_max  = np.array([ 250,  250, 1100], dtype=np.int16) # volume maximum (x,y,z)
    vox_size = np.array([  10,   10,   10], dtype=np.int16) # voxel size
    blob_radius = 15

    # create a test track
    hitc = HitCollection()
    hitc.evt   = 0
    hitc.time  = 0

    ttrk_x = [-45, -35, -25, -15, -5,  5, 15, 25, 35, 45, 80]
    ttrk_y = [  0,   0,   0,   0,  0,  0,  0,  0,  0,  0, 30]
    ttrk_z = [-45, -35, -25, -15, -5,  5, 15, 25, 35, 45, 80]
    ttrk_q = [ 10,  10,  10,  10, 10, 10, 10, 10, 10, 10, 20]

    for xx, yy, zz, qq in zip(ttrk_x, ttrk_y, ttrk_z, ttrk_q):
        hit       = Hit()
        hit.Npeak = 0
        hit.X     = xx
        hit.Y     = yy
        hit.R     = (xx**2 + yy**2) ** 0.5
        hit.Phi   = np.arctan2(yy, xx)
        hit.Z     = zz
        hit.Q     = qq
        hit.E     = qq
        hit.Ecorr = qq
        hit.Nsipm = 10
        hitc.append(hit)

    return Data(vol_min, vol_max, vox_size, blob_radius, hitc)

####### Fixtures corresponding to function outputs #############################

@fixture(scope='session')
def voxelc(data):
    d = data
    return build_voxels(d.hitc, d.vol_min, d.vol_max, d.vox_size)

@fixture(scope='session')
def adj_mat(voxelc):
    return calc_adj_matrix(voxelc)

@fixture(scope='session')
def tracks(voxelc, adj_mat):
    return construct_tracks(voxelc, adj_mat)

@fixture(scope='session')
def c_dist_mat(tracks):
    itmax, trks = tracks
    return calc_dist_mat(trks[itmax])

@fixture(scope='session')
def blobs(tracks, c_dist_mat, data):
    itmax, trks     = tracks
    dist_mat, spath = c_dist_mat
    return construct_blobs(trks[itmax], dist_mat, spath, data.blob_radius)

##### Tests ####################################################################

def test_number_of_voxels(voxelc, data):
    assert len(voxelc) == len(data.hitc)

def test_number_of_tracks(tracks):
    itmax, trks = tracks
    assert len(trks) == 2

def test_adjacency_matrix_diagonal(adj_mat):
    assert (np.diagonal(adj_mat) == -1).all()

def test_adjacency_matrix_off_diagonal(adj_mat):
    # neighbors should all be 10*sqrt(2) apart
    assert abs(adj_mat[0][1] - 14.142135) < 1.0e-3
    assert (adj_mat[0][1] == adj_mat[1][2] ==
            adj_mat[2][3] == adj_mat[3][4] ==
            adj_mat[4][5] == adj_mat[5][6] ==
            adj_mat[6][7] == adj_mat[7][8] ==
            adj_mat[8][9] )

def test_adjacency_matrix_is_symmetric(adj_mat):
    assert (adj_mat == adj_mat.T).all()

def test_track_lengths(tracks):
    itmax, trks = tracks
    itmin = 0
    if itmin == itmax: itmin =  1
    assert len(trks[itmax]) == 10
    assert len(trks[itmin]) ==  1

def test_blob_energies(blobs):
    Eblob1, Eblob2 = blobs
    assert Eblob1 == 20
    assert Eblob2 == 20

def test_blob_extreme_locations(c_dist_mat):
    dist_mat, spath = c_dist_mat

    e1 =  0
    e2 = -1
    if spath[e1].pos[0] > spath[e2].pos[0]:
        e1 = -1
        e2 =  0
    assert (np.array([-50, 0, -50]) == spath[e1].pos).all()
    assert (np.array([ 40, 0,  40]) == spath[e2].pos).all()

def test_dist_mat_properties(c_dist_mat):
    dist_mat, spath = c_dist_mat

    for ii in range(len(dist_mat)):
        assert dist_mat[ii][ii] == 0
    assert abs(np.max(dist_mat) - 127.27922061) < 1-0e-3
