"""hvt = Hits Voxels and Tracks plotting functions.

"""
import numpy  as np
import pandas as pd
import tables as tb
import matplotlib.pyplot as plt

from matplotlib import colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.pyplot import figure, show

from .  mpl_functions     import circles
from .  mpl_functions     import cube
from .  mpl_functions     import plt_scatter3d
from .  mpl_functions     import plt_scatter2d
from .  mpl_functions     import make_color_map
from .  mpl_functions     import set_plot_labels
from .. core.system_of_units_c import units
from .. core.core_functions import loc_elem_1d

from .. database               import load_db

def distance(va, vb):
    return np.linalg.norm(va.pos - vb.pos)

def print_tracks(tc):
    for trk_no, trk in enumerate(tc.tracks):
        print('trk no = {}, number of voxels = {}, energy = {}'.
              format(trk_no, trk.number_of_voxels, trk.E))
        vE = [v.E for v in trk.voxels]
        print('voxel energies = {}'.format(vE))
        ba, bb = trk.blobs
        print('----------------------------')
        print('blob a: number of voxels = {}, seed = {}, energy = {}'.
              format(ba.number_of_voxels, ba.seed, ba.E))
        print('blob b: number of voxels = {}, seed = {}, energy = {}'.
              format(bb.number_of_voxels, bb.seed, bb.E))

        blobs_sa = set(ba.voxels)
        blobs_sb = set(bb.voxels)
        blobs_i = blobs_sa.intersection(blobs_sb)

        print('intersection blobs a and blob b = {}'.
              format(len(blobs_i)))


        print('\n')

def print_distance_to_track(tc, trkm):
    trkm_ba, trkm_bb = trkm.blobs
    for trk_no, trk in enumerate(tc.tracks):
        print('trk no = {}, number of voxels = {}, energy = {}'.
              format(trk_no, trk.number_of_voxels, trk.E))
        ba, bb = trk.blobs
        print('----------------------------')
        print('blob a: number of voxels = {}, seed = {}, energy = {}'.
              format(ba.number_of_voxels, ba.seed, ba.E))
        print('blob b: number of voxels = {}, seed = {}, energy = {}'.
              format(bb.number_of_voxels, bb.seed, bb.E))

        print("""distances to reference track:
            a : a = {}        a : b = {}
            b : a = {}        b : b = {}
    """.format(distance(trkm_ba.seed, ba.seed),
               distance(trkm_ba.seed, bb.seed),
               distance(trkm_bb.seed, ba.seed),
               distance(trkm_bb.seed, bb.seed)))


def get_hits(hits, norm=True):
    x, y, z, q = [], [], [], []
    for hit in hits:
        x.append(hit.X)
        y.append(hit.Y)
        z.append(hit.Z)
        q.append(hit.E)
    if norm:
        return np.array(x), np.array(y), np.array(z), np.array(q)/np.amax(q)
    else:
        return np.array(x), np.array(y), np.array(z), np.array(q)


def set_xyz_limit(ax, xview, yview, zview):
    ax.set_xlim3d(xview)
    ax.set_ylim3d(yview)
    ax.set_zlim3d(zview)


def set_xy_limit(ax, xview, yview):
    ax.set_xlim(xview)
    ax.set_ylim(yview)


def plot_hits_3D(hits, norm=True, xview=(-198, 198), yview=(-198, 198), zview=(0, 500),
                 xsc = 10, ysc = 10,  zsc = 10,
                 s = 30, alpha=0.3, figsize=(12, 12)):

    x, y, z, q = get_hits(hits, norm = norm)

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(221, projection='3d')
    set_xyz_limit(ax1, xview, yview, zview)

    ax2 = fig.add_subplot(222, projection='3d')
    set_xyz_limit(ax2, xview, yview, (np.amin(z) - zsc, np.amax(z) + zsc))

    ax3 = fig.add_subplot(223)
    set_xy_limit(ax3, xview, yview)

    ax4 = fig.add_subplot(224)
    set_xy_limit(ax4, (np.amin(x) -xsc, np.amax(x) + xsc),
                      (np.amin(y) -ysc, np.amax(y) + ysc))

    plt_scatter3d(ax1, x, y, z, q, s, alpha)
    plt_scatter3d(ax2, x, y, z, q, s, alpha)
    plt_scatter2d(ax3, x, y, q, s, alpha)
    plt_scatter2d(ax4, x, y, q, s, alpha)

def make_cube(v, voxel_size, col, axes):
    quads = cube(origin=(v.X, v.Y, v.Z),
                 width=voxel_size.X,
                 height=voxel_size.Y,
                 depth=voxel_size.Z)
    collection = Poly3DCollection(quads)
    collection.set_color(col)
    axes.add_collection3d(collection)

def set_color_map(fig, axes, voxels, alpha, colormap='Blues'):
    scplot = axes.scatter([], [], c=[])
    fig.colorbar(scplot, ax=axes)
    voxel_energy = sorted([v.E for v in voxels])
    RGBA, _ = make_color_map(voxel_energy, alpha=alpha, colormap=colormap)
    return RGBA, voxel_energy


def draw_voxels(voxels, voxel_size, xview=(-198, 198), yview=(-198, 198), zview=(0, 500),
                figsize=(10, 8), alpha=0.5, colormap='Blues'):

    canvas = figure()
    canvas.set_size_inches(figsize)
    axes = Axes3D(canvas)

    set_xyz_limit(axes, xview, yview, zview)
    RGBA, voxel_energy = set_color_map(canvas, axes, voxels, alpha, colormap=colormap)
    #print(RGBA)
    for v in voxels:
        c_index = loc_elem_1d(voxel_energy, v.E)
        make_cube(v, voxel_size, RGBA[c_index], axes)
    show()


def draw_voxels2(voxels, voxel_size, xview=(-198, 198), yview=(-198, 198), zview=(0, 500),
                 alpha=0.5, figsize=(10, 8), colormap='Blues'):

    fig = plt.figure(figsize=figsize)
    axes = fig.add_subplot(111, projection='3d', aspect='equal')

    set_xyz_limit(axes, xview, yview, zview)
    RGBA, voxel_energy = set_color_map(canvas, axes, voxels, alpha, colormap=colormap)

    for v in voxels:
        c_index = loc_elem_1d(RGBA, v.E)
        make_cube(v, voxel_size, RGBA[c_index], axes)
    plt.show()

def draw_tracks(track_container, voxel_size,
                xview=(-198, 198), yview=(-198, 198), zview=(0, 500),
                alpha=0.5, figsize=(10, 8)):

    fig = plt.figure(figsize=figsize)
    axes = fig.add_subplot(111, projection='3d', aspect='equal')
    set_xyz_limit(axes, xview, yview, zview)
    RGBA = [(1.0, 0.0, 0.0, 0.5), (0.0, 0.0, 1.0, 0.5) ]

    for trk in track_container.tracks:
        ba, bb = trk.blobs
        for v in trk.voxels:
            make_cube(v, voxel_size, RGBA[0], axes)
        for v in ba.voxels:
            make_cube(v, voxel_size, RGBA[1], axes)
        for v in bb.voxels:
            make_cube(v, voxel_size, RGBA[1], axes)

    plt.show()
