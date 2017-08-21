"""A utility module for plots with matplotlib"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)


from matplotlib import colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.pyplot import figure, show

from .. core.system_of_units_c import units
from .. core.core_functions import define_window
from .. database     import load_db


# matplotlib.style.use("ggplot")
#matplotlib.rc('animation', html='html5')

# histograms, signals and shortcuts


def plot_vector(v, figsize=(10,10)):
    """Plot vector v and return figure """
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot(1, 1, 1)
    plt.plot(v)


def fplot_xy(x, y, figsize=(10,10)):
    """Plot y vs x and return figure"""
    fig = Figure(figsize=figsize)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(x,y)
    return fig


def plot_xy(x, y, figsize=(10,10)):
    """Plot y vs x and return figure"""
    plt.figure(figsize=figsize)
    ax1 = plt.subplot(1, 1, 1)
    plt.plot(x,y)


def plot_sipm_list(sipmrwf, sipm_list, x=4):
    """Plot a list of SiPMs."""
    plt.figure(figsize=(12, 12))
    nmax = len(sipm_list)
    y = int(nmax / x) + 1
    for i in range(0, nmax):
        plt.subplot(x, y, i+1)
        plt.plot(sipmrwf[sipm_list[i]])


def plot_sensor_list_ene_map(run_number, swf, slist, stype='PMT', cmap='Blues'):
        """Plot a map of the energies of sensors in list."""
        DataSensor = load_db.DataPMT(run_number)
        radius = 10
        if stype == 'SiPM':
            DataSensor = load_db.DataSiPM(run_number)
            radius = 2
        xs = DataSensor.X.values
        ys = DataSensor.Y.values
        r = np.ones(len(xs)) * radius
        col = np.zeros(len(xs))
        for i in slist:
            col[i] = np.sum(swf[i])

        plt.figure(figsize=(8, 8))
        plt.subplot(aspect="equal")
        circles(xs, ys, r, c=col, alpha=0.5, ec="none", cmap=cmap)
        plt.colorbar()

        plt.xlim(-198, 198)
        plt.ylim(-198, 198)


def draw_pmt_map(run_number):
    """Draws a map with the channel_id number in the positions of the PMTs.
        channel_id = elecID (electronic ID) for PMTs.
        xpmt       = x pos
    """
    DataPMT = load_db.DataPMT(run_number)
    xpmt = DataPMT.X.values
    ypmt = DataPMT.Y.values
    channel_id = DataPMT.ChannelID.values
    cid = ['{}'.format(c) for c in channel_id]
    fig, ax = plt.subplots()
    plt.plot(xpmt, ypmt, 'o')

    for c, x,y in zip(cid, xpmt,ypmt):
        xy = (x,y)
        offsetbox = TextArea(c, minimumdescent=False)
        ab = AnnotationBbox(offsetbox, xy)
        ax.add_artist(ab)

    plt.show()


def plt_scatter3d(ax, x, y, z, q, s = 30, alpha=0.3):
    ax.scatter(x, y, z, c=q, s=s, alpha=alpha)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")


def plt_scatter2d(ax, x, y, q, s = 30, alpha=0.3):
    ax.scatter(x, y, c=q, s=s, alpha=alpha)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")


def make_tracking_plane_movie(slices, thrs=0.1):
    """Create a video made of consecutive frames showing the response of
    the tracking plane.

    Parameters
    ----------
    slices : 2-dim np.ndarray
        The signal of each SiPM (axis 1) for each time sample (axis 0).
    thrs : float, optional
        Default cut value to be applied to each slice. Defaults to 0.1.

    Returns
    -------
    mov : matplotlib.animation
        The movie.
    """

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8)
    DataSensor = load_db.DataSiPM(0)
    X = DataSensor.X.values
    Y = DataSensor.Y.values

    xmin, xmax = np.nanmin(X), np.nanmax(X)
    ymin, ymax = np.nanmin(Y), np.nanmax(Y)
    def init():
        global cbar, scplot
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
        scplot = ax.scatter([], [], c=[])
        cbar = fig.colorbar(scplot, ax=ax)
        cbar.set_label("Charge (pes)")
        return (scplot,)

    def animate(i):
        global cbar, scplot
        slice_ = slices[i]
        selection = slice_ > np.nanmax(slice_) * thrs
        x, y, q = X[selection], Y[selection], slice_[selection]
        cbar.remove()
        fig.clear()
        ax = plt.gca()
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
        scplot = ax.scatter(x, y, c=q, marker="s", vmin=0,
                            vmax=np.nanmax(slices))
        cbar = fig.colorbar(scplot, ax=ax,
                            boundaries=np.linspace(0, np.nanmax(slices), 100))
        cbar.set_label("Charge (pes)")
        return (scplot,)

    anim = matplotlib.animation.FuncAnimation(fig, animate, init_func=init,
                                              frames=len(slices), interval=200,
                                              blit=False)
    return anim


def set_plot_labels(xlabel="", ylabel="", grid=True):
    """Short cut to set labels in plots."""
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if grid is True:
        plt.grid(which="both", axis="both")


# Circles!
def circles(x, y, s, c="a", vmin=None, vmax=None, **kwargs):
    """Make a scatter of circles plot of x vs y, where x and y are
    sequence like objects of the same lengths. The size of circles are
    in data scale.

    Parameters
    ----------
    x,y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, )
        Radius of circle in data unit.
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`

    Examples
    --------
    a = np.arange(11)
    circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')
    plt.colorbar()

    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)

    """

    if np.isscalar(c):
        kwargs.setdefault("color", c)
        c = None
    if "fc" in kwargs:
        kwargs.setdefault("facecolor", kwargs.pop("fc"))
    if "ec" in kwargs:
        kwargs.setdefault("edgecolor", kwargs.pop("ec"))
    if "ls" in kwargs:
        kwargs.setdefault("linestyle", kwargs.pop("ls"))
    if "lw" in kwargs:
        kwargs.setdefault("linewidth", kwargs.pop("lw"))

    patches = [Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(x, y, s)]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        collection.set_array(np.asarray(c))
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    if c is not None:
        plt.sci(collection)
    return collection

"""
Defines geometry plotting utilities objects:
-   :func:`quad`
-   :func:`grid`
-   :func:`cube`
"""


__all__ = ['quad', 'grid', 'cube']


def quad(plane='xy', origin=None, width=1, height=1, depth=0):
    """
    Returns the vertices of a quad geometric element in counter-clockwise
    order.
    Parameters
    ----------
    plane : array_like, optional
        **{'xy', 'xz', 'yz'}**,
        Construction plane of the quad.
    origin: array_like, optional
        Quad origin on the construction plane.
    width: numeric, optional
        Quad width.
    height: numeric, optional
        Quad height.
    depth: numeric, optional
        Quad depth.
    Returns
    -------
    ndarray
        Quad vertices.
    Examples
    --------
    >>> quad()
    array([[0, 0, 0],
           [1, 0, 0],
           [1, 1, 0],
           [0, 1, 0]])
    """

    u, v = (0, 0) if origin is None else origin

    plane = plane.lower()
    if plane == 'xy':
        vertices = ((u, v, depth), (u + width, v, depth),
                    (u + width, v + height, depth), (u, v + height, depth))
    elif plane == 'xz':
        vertices = ((u, depth, v), (u + width, depth, v),
                    (u + width, depth, v + height), (u, depth, v + height))
    elif plane == 'yz':
        vertices = ((depth, u, v), (depth, u + width, v),
                    (depth, u + width, v + height), (depth, u, v + height))
    else:
        raise ValueError('"{0}" is not a supported plane!'.format(plane))

    return np.array(vertices)


def grid(plane='xy',
         origin=None,
         width=1,
         height=1,
         depth=0,
         width_segments=1,
         height_segments=1):
    """
    Returns the vertices of a grid made of quads.
    Parameters
    ----------
    plane : array_like, optional
        **{'xy', 'xz', 'yz'}**,
        Construction plane of the grid.
    origin: array_like, optional
        Grid origin on the construction plane.
    width: numeric, optional
        Grid width.
    height: numeric, optional
        Grid height.
    depth: numeric, optional
        Grid depth.
    width_segments: int, optional
        Grid segments, quad counts along the width.
    height_segments: int, optional
        Grid segments, quad counts along the height.
    Returns
    -------
    ndarray
        Grid vertices.
    Examples
    --------
    >>> grid(width_segments=2, height_segments=2)
    array([[[ 0. ,  0. ,  0. ],
            [ 0.5,  0. ,  0. ],
            [ 0.5,  0.5,  0. ],
            [ 0. ,  0.5,  0. ]],
    <BLANKLINE>
           [[ 0. ,  0.5,  0. ],
            [ 0.5,  0.5,  0. ],
            [ 0.5,  1. ,  0. ],
            [ 0. ,  1. ,  0. ]],
    <BLANKLINE>
           [[ 0.5,  0. ,  0. ],
            [ 1. ,  0. ,  0. ],
            [ 1. ,  0.5,  0. ],
            [ 0.5,  0.5,  0. ]],
    <BLANKLINE>
           [[ 0.5,  0.5,  0. ],
            [ 1. ,  0.5,  0. ],
            [ 1. ,  1. ,  0. ],
            [ 0.5,  1. ,  0. ]]])
    """

    u, v = (0, 0) if origin is None else origin

    w_x, h_y = width / width_segments, height / height_segments

    quads = []
    for i in range(width_segments):
        for j in range(height_segments):
            quads.append(
                quad(plane, (i * w_x + u, j * h_y + v), w_x, h_y, depth))

    return np.array(quads)


def cube(plane=None,
         origin=None,
         width=1,
         height=1,
         depth=1,
         width_segments=1,
         height_segments=1,
         depth_segments=1):
    """
    Returns the vertices of a cube made of grids.
    Parameters
    ----------
    plane : array_like, optional
        Any combination of **{'+x', '-x', '+y', '-y', '+z', '-z'}**,
        Included grids in the cube construction.
    origin: array_like, optional
        Cube origin.
    width: numeric, optional
        Cube width.
    height: numeric, optional
        Cube height.
    depth: numeric, optional
        Cube depth.
    width_segments: int, optional
        Cube segments, quad counts along the width.
    height_segments: int, optional
        Cube segments, quad counts along the height.
    depth_segments: int, optional
        Cube segments, quad counts along the depth.
    Returns
    -------
    ndarray
        Cube vertices.
    Examples
    --------
    >>> cube()
    array([[[ 0.,  0.,  0.],
            [ 1.,  0.,  0.],
            [ 1.,  1.,  0.],
            [ 0.,  1.,  0.]],
    <BLANKLINE>
           [[ 0.,  0.,  1.],
            [ 1.,  0.,  1.],
            [ 1.,  1.,  1.],
            [ 0.,  1.,  1.]],
    <BLANKLINE>
           [[ 0.,  0.,  0.],
            [ 1.,  0.,  0.],
            [ 1.,  0.,  1.],
            [ 0.,  0.,  1.]],
    <BLANKLINE>
           [[ 0.,  1.,  0.],
            [ 1.,  1.,  0.],
            [ 1.,  1.,  1.],
            [ 0.,  1.,  1.]],
    <BLANKLINE>
           [[ 0.,  0.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  1.,  1.],
            [ 0.,  0.,  1.]],
    <BLANKLINE>
           [[ 1.,  0.,  0.],
            [ 1.,  1.,  0.],
            [ 1.,  1.,  1.],
            [ 1.,  0.,  1.]]])
    """

    plane = (('+x', '-x', '+y', '-y', '+z', '-z')
             if plane is None else [p.lower() for p in plane])
    u, v, w = (0, 0, 0) if origin is None else origin

    w_s, h_s, d_s = width_segments, height_segments, depth_segments

    grids = []
    if '-z' in plane:
        grids.extend(grid('xy', (u, w), width, depth, v, w_s, d_s))
    if '+z' in plane:
        grids.extend(grid('xy', (u, w), width, depth, v + height, w_s, d_s))

    if '-y' in plane:
        grids.extend(grid('xz', (u, v), width, height, w, w_s, h_s))
    if '+y' in plane:
        grids.extend(grid('xz', (u, v), width, height, w + depth, w_s, h_s))

    if '-x' in plane:
        grids.extend(grid('yz', (w, v), depth, height, u, d_s, h_s))
    if '+x' in plane:
        grids.extend(grid('yz', (w, v), depth, height, u + width, d_s, h_s))

    return np.array(grids)

def make_color_map(lst, alpha=0.5, colormap='inferno'):
    """lst is the sequence to map to colors"""
    minima = min(lst)
    maxima = max(lst)

    norm = mcolors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=colormap)

    RGBA = []
    for v in lst:
        RGBA.append(mapper.to_rgba(v, alpha))

    return RGBA, mapper
