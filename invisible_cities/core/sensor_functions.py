"""Functions manipulating sensors (PMTs and SiPMs)
JJGC January 2017
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation

from   invisible_cities.database import load_db
import invisible_cities.core.system_of_units as units
from   invisible_cities.core.mpl_functions import circles


def weighted_sum(CWF, w_vector):
    """Return the weighted sum of CWF (weights defined in w_vector)."""

    NPMT = len(CWF)
    NWF = len(CWF[0])
    assert len(w_vector) == NPMT

    csum = np.zeros(NWF, dtype=np.double)
    for j in range(NPMT):
        csum += CWF[j] * 1 / w_vector[j]
    return csum


def sipm_with_signal(sipmrwf, thr=1):
    """Find the SiPMs with signal in this event."""
    SIPML = []
    for i in range(sipmrwf.shape[0]):
        if np.sum(sipmrwf[i] > thr):
            SIPML.append(i)
    return np.array(SIPML)


def plot_sipm_list(sipmrwf, sipm_list, x=4):
    """Plot a list of SiPMs."""
    plt.figure(figsize=(12, 12))
    nmax = len(sipm_list)
    y = int(nmax / x) + 1
    for i in range(0, nmax):
        plt.subplot(x, y, i+1)
        plt.plot(sipmrwf[sipm_list[i]])


def plot_sensor_list_ene_map(swf, slist, stype='PMT', cmap='Blues'):
        """Plot a map of the energies of sensors in list."""
        DataSensor = load_db.DataPMT(0)
        radius = 10
        if stype == 'SiPM':
            DataSensor = load_db.DataSiPM(0)
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


# def plot_event_3D(pmap, sipmdf, outputfile=None, thrs=0.):
#     """Create a 3D+1 representation of the event based on the SiPMs signal
#     for each slice.
#
#     Parameters
#     ----------
#     pmap : Bridges.PMap
#         The pmap of some event.
#     sipmdf : pd.DataFrame
#         Contains the X, Y info.
#     outputfile : string, optional
#         Name of the outputfile. If given, the plot is saved with this name.
#         It is not saved by default.
#     thrs : float, optional
#         Relative cut to be applied per slice. Defaults to 0.
#
#     """
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     x, y, z, q = [], [], [], []
#     for peak in pmap.get("S2"):
#         for time, sl in zip(peak.times, peak.anode):
#             selection = sl > thrs * np.nanmax(sl)
#             x.extend(sipmdf["X"].values[selection])
#             y.extend(sipmdf["Y"].values[selection])
#             z.extend(np.ones_like(sl[selection]) * time)
#             q.extend(sl[selection])
#
#     ax.scatter(x, z, y, c=q, s=[2*qi for qi in q], alpha=0.3)
#     ax.set_xlabel("x (mm)")
#     ax.set_ylabel("z (mm)")
#     ax.set_zlabel("y (mm)")
#     fig.set_size_inches(10,8)
#     if outputfile is not None:
#         fig.savefig(outputfile)
