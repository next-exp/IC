"""A utility module for plots with matplotlib"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
# from mpl_toolkits.mplot3d import Axes3D
# from IPython.display import HTML

from .. core.system_of_units_c import units
from .. core.core_functions import define_window
from .. database     import load_db


# matplotlib.style.use("ggplot")
#matplotlib.rc('animation', html='html5')

# histograms, signals and shortcuts
def hbins(x, nsigma=5, nbins=10):
    """Given an array x, hbins returns the number of bins
    in an interval of  [<x> - nsigma*std(x), <x> + nsigma*std(x)]
    """
    xmin = np.average(x) - nsigma*np.std(x)
    xmax = np.average(x) + nsigma*np.std(x)
    bins = np.linspace(xmin, xmax, nbins)
    return bins


def histo(x, nbins, title="hsimple", xlabel="", ylabel="Frequency"):
    """histograms"""

    plt.hist(x, nbins, histtype="bar", alpha=0.75)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def plts(signal, signal_start=0, signal_end=1e+4, offset=5):
    """Plot a signal in a give interval, control offset by hand."""
    ax1 = plt.subplot(1, 1, 1)
    ymin = np.amin(signal[signal_start:signal_end]) - offset
    ymax = np.amax(signal[signal_start:signal_end]) + offset
    ax1.set_xlim([signal_start, signal_end])
    ax1.set_ylim([ymin, ymax])
    plt.plot(signal)


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


def plot_signal(signal_t, signal, title="signal",
                signal_start=0, signal_end=1e+4,
                ymax=200, t_units="", units=""):
    """Given a series signal (t, signal), plot the signal."""

    ax1 = plt.subplot(1, 1, 1)
    ax1.set_xlim([signal_start, signal_end])
    ax1.set_ylim([0, ymax])
    set_plot_labels(xlabel="t ({})".format(t_units),
                  ylabel="signal ({})".format(units))
    plt.title(title)
    plt.plot(signal_t, signal)
    # plt.show()


def plot_signal_vs_time_mus(signal,
                            t_min      =    0,
                            t_max      = 1200,
                            signal_min =    0,
                            signal_max =  200,
                            figsize=(6,6)):
    """Plot signal versus time in mus (tmin, tmax in mus). """
    plt.figure(figsize=figsize)
    tstep = 25 # in ns
    PMTWL = signal.shape[0]
    signal_t = np.arange(0., PMTWL * tstep, tstep)/units.mus
    ax1 = plt.subplot(1, 1, 1)
    ax1.set_xlim([t_min, t_max])
    ax1.set_ylim([signal_min, signal_max])
    set_plot_labels(xlabel = "t (mus)",
                    ylabel = "signal (pes/adc)")
    plt.plot(signal_t, signal)


def plot_waveform(pmtwf, zoom=False, window_size=800):
    """Take as input a vector a single waveform and plot it"""

    first, last = 0, len(pmtwf)
    if zoom:
        first, last = define_window(pmtwf, window_size)

    mpl.set_plot_labels(xlabel="samples", ylabel="adc")
    plt.plot(pmtwf[first:last])


def plot_waveforms_overlap(wfs, zoom=False, window_size=800):
    """Draw all waveforms together. If zoom is True, plot is zoomed
    around peak.
    """
    first, last = 0, wfs.shape[1]
    if zoom:
        first, last = define_window(wfs[0], window_size)
    for wf in wfs:
        plt.plot(wf[first:last])


def plot_wfa_wfb(wfa, wfb, zoom=False, window_size=800):
    """Plot together wfa and wfb, where wfa and wfb can be
    RWF, CWF, BLR.
    """
    plt.figure(figsize=(12, 12))
    for i in range(len(wfa)):
        first, last = 0, len(wfa[i])
        if zoom:
            first, last = define_window(wfa[i], window_size)
        plt.subplot(3, 4, i+1)
        # ax1.set_xlim([0, len_pmt])
        mpl.set_plot_labels(xlabel="samples", ylabel="adc")
        plt.plot(wfa[i][first:last], label= 'WFA')
        plt.plot(wfb[i][first:last], label= 'WFB')
        legend = plt.legend(loc='upper right')
        for label in legend.get_texts():
            label.set_fontsize('small')


def plot_pmt_waveforms(pmtwfdf, zoom=False, window_size=800, figsize=(10,10)):
    """plot PMT wf and return figure"""
    plt.figure(figsize=figsize)
    for i in range(len(pmtwfdf)):
        first, last = 0, len(pmtwfdf[i])
        if zoom:
            first, last = define_window(pmtwfdf[i], window_size)

        ax = plt.subplot(3, 4, i+1)
        set_plot_labels(xlabel="samples", ylabel="adc")
        plt.plot(pmtwfdf[i][first:last])


def plot_pmt_signals_vs_time_mus(pmt_signals,
                                 pmt_active,
                                 t_min      =    0,
                                 t_max      = 1200,
                                 signal_min =    0,
                                 signal_max =  200,
                                 figsize=(10,10)):
    """Plot PMT signals versus time in mus  and return figure."""

    tstep = 25
    PMTWL = pmt_signals[0].shape[0]
    signal_t = np.arange(0., PMTWL * tstep, tstep)/units.mus
    plt.figure(figsize=figsize)

    for j, i in enumerate(pmt_active):
        ax1 = plt.subplot(3, 4, j+1)
        ax1.set_xlim([t_min, t_max])
        ax1.set_ylim([signal_min, signal_max])
        set_plot_labels(xlabel = "t (mus)",
                        ylabel = "signal (pes/adc)")

        plt.plot(signal_t, pmt_signals[i])


def plot_calibrated_sum_in_mus(CSUM,
                               tmin=0, tmax=1200,
                               signal_min=-5, signal_max=200,
                               csum=True, csum_mau=False):
    """Plots calibrated sums in mus (notice units)"""

    if csum:
        plot_signal_vs_time_mus(CSUM.csum,
                                t_min=tmin, t_max=tmax,
                                signal_min=signal_min, signal_max=signal_max,
                                label='CSUM')
    if csum_mau:
        plot_signal_vs_time_mus(CSUM.csum_mau,
                                t_min=tmin, t_max=tmax,
                                signal_min=signal_min, signal_max=signal_max,
                                label='CSUM_MAU')



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



def plot_event_3D(pmap, sipmdf, outputfile=None, thrs=0.):
    """Create a 3D+1 representation of the event based on the SiPMs signal
    for each slice.

    Parameters
    ----------
    pmap : Bridges.PMap
        The pmap of some event.
    sipmdf : pd.DataFrame
        Contains the X, Y info.
    outputfile : string, optional
        Name of the outputfile. If given, the plot is saved with this name.
        It is not saved by default.
    thrs : float, optional
        Relative cut to be applied per slice. Defaults to 0.

    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x, y, z, q = [], [], [], []
    for peak in pmap.get("S2"):
        for time, sl in zip(peak.times, peak.anode):
            selection = sl > thrs * np.nanmax(sl)
            x.extend(sipmdf["X"].values[selection])
            y.extend(sipmdf["Y"].values[selection])
            z.extend(np.ones_like(sl[selection]) * time)
            q.extend(sl[selection])

    ax.scatter(x, z, y, c=q, s=[2*qi for qi in q], alpha=0.3)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_zlabel("y (mm)")
    fig.set_size_inches(10,8)
    if outputfile is not None:
        fig.savefig(outputfile)

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
