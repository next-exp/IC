"""A utility module for plots with matplotlib"""
from __future__ import print_function, division, absolute_import

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
# from mpl_toolkits.mplot3d import Axes3D
# from IPython.display import HTML

import invisible_cities.core.core_functions as cf
import invisible_cities.core.system_of_units as units
import invisible_cities.core.wfm_functions as wfm
import invisible_cities.core.tbl_functions as tbl


# matplotlib.style.use("ggplot")
matplotlib.rc('animation', html='html5')

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
    plt.show()


def set_plot_labels(xlabel="", ylabel="", grid=True):
    """Short cut to set labels in plots."""
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if grid is True:
        plt.grid(which="both", axis="both")


# Circles!
def circles(x, y, s, c="b", vmin=None, vmax=None, **kwargs):
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


def plot_waveforms(pmtwfdf, maxlen=0, zoom=False, window_size=800):
    """Take as input a df storing the PMT wf and plot the 12 PMT WF."""
    plt.figure(figsize=(12, 12))
    for i in range(12):
        first, last = 0, len(pmtwfdf[i])
        if zoom:
            first, last = define_window(pmtwfdf[i], window_size)
        plt.subplot(3, 4, i+1)
        # ax1.set_xlim([0, len_pmt])
        set_plot_labels(xlabel="samples", ylabel="adc")
        plt.plot(pmtwfdf[i][first:last])

    plt.show()


def scan_waveforms(pmtea, list_of_events=(0,)):
    """Takes the earray pmtea and a list of events and scan the waveforms."""
    for event in list_of_events:
        plot_waveforms(wfm.get_waveforms(pmtea, event_number=event))
        cf.wait()


def define_window(wf, window_size):
    """Define a window based on a peak. Takes max plus/minus *window_size*."""
    peak = np.argmax(abs(wf - np.mean(wf)))
    return max(0, peak - window_size), min(len(wf), peak + window_size)


def overlap_waveforms(wfset, event, zoom=True, window_size=800):
    """Draw all waveforms together. If zoom is True, plot is zoomed around peak."""
    wfs = wfset[event]
    first, last = 0, wfs.shape[1]
    if zoom:
        first, last = define_window(wfs[0], window_size)
    for wf in wfs:
        plt.plot(wf[first:last])


def compare_raw_blr(pmtrwf, pmtblr, evt=0, zoom=True, window_size=800):
    """Compare PMT RWF and BLR WF. Option zoom takes a window around the
    peak of size window_size.

    """
    plt.figure(figsize=(12, 12))
    for i, (raw, blr) in enumerate(zip(pmtrwf[evt], pmtblr[evt])):
        first, last = 0, pmtrwf.shape[2]
        if zoom:
            first, last = define_window(raw, window_size)
        plt.subplot(3, 4, i+1)
        plt.plot(raw[first:last])
        plt.plot(blr[first:last])

def plot_blr_cwf(pmtblr, cwf, maxlen=0, zoom=False, window_size=800):
    """Take as input a df storing the PMT wf and plot the 12 PMT WF."""
    plt.figure(figsize=(12, 12))
    for i in range(12):
        first, last = 0, len(pmtblr[i])
        if zoom:
            first, last = define_window(pmtblr[i], window_size)
        plt.subplot(3, 4, i+1)
        # ax1.set_xlim([0, len_pmt])
        set_plot_labels(xlabel="samples", ylabel="adc")
        plt.plot(pmtblr[i][first:last])
        plt.plot(cwf[i][first:last])

    plt.show()

def compare_corr_raw(pmtcwf, pmtblr, evt=0, zoom=True, window_size=800):
    """Compare PMT CWF and RWF (or BLR). Option zoom takes a window around
    the peak of size *window_size*.
    """
    pmtblr = map(lambda wf: 2500 - wf, pmtblr)
    plt.figure(figsize=(12, 12))
    for i, (raw, blr) in enumerate(zip(pmtcwf[evt], pmtblr[evt])):
        first, last = 0, pmtcwf.shape[2]
        first, last = define_window(raw, window_size)
        plt.subplot(3, 4, i+1)
        plt.plot(raw[first:last])
        plt.plot(blr[first:last])


def plot_pmtwf(PMTWF):
    """Plot pmtwf."""
    pmtwf = PMTWF[0]
    plt.plot(pmtwf["time_mus"] / units.mus, pmtwf["ene_pes"])
    ene = pmtwf["ene_pes"].values / 12
    time = pmtwf["time_mus"].values / units.mus
    plt.xlabel("t ($\mu$s)")
    plt.ylabel("E (pes)")
    plt.show()
    plt.figure(figsize=(12, 12))
    for i in range(1, len(PMTWF)):
        plt.subplot(3, 4, i)
        pmtwf = PMTWF[i]
        plt.plot(pmtwf["time_mus"] / units.mus, pmtwf["ene_pes"])
        plt.plot(time, ene)

    plt.show()


def plot_sensor(geom_df, sensor_df, energy_df, event=0, radius=10):
    """Plot the energy of the sensors."""
    x = sensor_df["x"].values
    y = sensor_df["y"].values
    r = np.ones(len(sensor_df["x"].values)) * radius
#    col = energy_df[event].values ### BUG! we were taking columns

#    col = energy_df.iloc[[event]].values.flatten() JMB fix
    col = energy_df.ix[event].values  # another fix more concise

    plt.figure(figsize=(10, 10))
    plt.subplot(aspect="equal")
    circles(x, y, r, c=col, alpha=0.5, ec="none")
    plt.colorbar()
    # xlim(-198,198)  #one should use geom info
    # ylim(-198,198)
    plt.xlim(geom_df["xdet_min"], geom_df["xdet_max"])
    plt.ylim(geom_df["ydet_min"], geom_df["ydet_max"])
    return col


def plot_ene_pmt(geom_df, sensor_df, epmt, event_number=0, radius=10):
    """Plots the reconstructed energy of the PMTs energy_se is a series
    describing the reconstructed energy in each PMT.
    """
    x = sensor_df["x"].values
    y = sensor_df["y"].values
    r = np.ones(len(sensor_df["x"].values)) * radius
    col = epmt[event_number]

    plt.figure(figsize=(10, 10))
    plt.subplot(aspect="equal")
    circles(x, y, r, c=col, alpha=0.5, ec="none")
    plt.colorbar()
    # xlim(-198,198)  #one should use geom info
    # ylim(-198,198)
    plt.xlim(geom_df["xdet_min"], geom_df["xdet_max"])
    plt.ylim(geom_df["ydet_min"], geom_df["ydet_max"])
    return col


def plot_sipm(sipm, nmin=0, nmax=16, x=4, y=4):
    """Take as input a wf storing the SiPMs and plot nsipm."""
    plt.figure(figsize=(12, 12))

    for i in range(nmin, nmax):
        plt.subplot(y, y, i+1)
        plt.plot(sipm[i])

    plt.show()

def plot_best(sipmrwf, sipmtwf, sipmdf, evt=0):
    """Plot the noisy waveform of the SiPM with greatest charge and
    superimpose the true waveform.
    """
    plt.figure(figsize=(10, 8))
    # Find SiPM with greatest peak
    maxsipm = np.unravel_index(sipmrwf[evt].argmax(), sipmrwf[evt].shape)[0]
    print("SiPM with greatest peak is at "
          "index {} with ID {}".format(maxsipm, sipmdf.ix[maxsipm].sensorID))

    # Plot noisy waveform in red and noiseless waveform in blue
    true_times, true_amps = tbl.read_sensor_wf(sipmtwf, evt, maxsipm)
    plt.plot(sipmrwf[evt, maxsipm, :])
    plt.plot(true_times, np.array(true_amps) * sipmdf["adc_to_pes"][maxsipm])
    plt.xlabel("time ($\mu$s)")
    plt.ylabel("Energy (adc)")


def plot_best_group(sipmrwf, sipmtwf, sipmdf, evt=0, nsipms=9, ncols=3):
    """Plot the noisy (red) and true (blue) waveforms of the *nsipms*
    SiPMs with greatest charge.
    """
    plt.figure(figsize=(10, 8))
    # Find SiPM with greatest peak
    sipms = enumerate(sipmrwf[evt])
    sipms = sorted(sipms, key=lambda x: max(x[1]), reverse=True)[:nsipms]

    nrows = int(math.ceil(nsipms * 1 / ncols))
    for i, (sipm_index, sipm_wf) in enumerate(sipms):
        true_times, true_amps = tbl.read_sensor_wf(sipmtwf, evt, sipm_index)
        if len(true_amps) == 0:
            true_times = np.arange(len(sipm_wf))
            true_amps = np.zeros(len(sipm_wf))
        plt.subplot(nrows, ncols, i+1)
        plt.plot(sipm_wf)
        plt.plot(true_times,
                 np.array(true_amps) * sipmdf["adc_to_pes"][sipm_index])
        plt.xlabel("time ($\mu$s)")
        plt.ylabel("Energy (adc)")
    plt.tight_layout()


def plot_pmap(pmap, legend=True, style="*-"):
    for i, peak in enumerate(pmap.peaks):
        plt.plot(peak.times, peak.cathode, style,
                 label="peak #{} type {}".format(i, peak.signal))

    if legend:
        plt.legend(loc="upper left")
    plt.xlabel("Time ($\mu$s)")
    plt.ylabel("Energy (pes)")


def plot_anode_slice(slice, sipmdf, threshold=0.1, cut_type="RELATIVE"):
    """Plot the anode for a single slice as a colored 2D plot.

    Parameters
    ----------
    slice : 1-dim np.ndarray
        The charge for each sensor.
    sipmdf : pd.DataFrame
        Contains the sensors' info.
    threshold : float, optional
        Cut level for SiPMs. Defaults to 0.1.
    cut_type : string or None, optional.
        Type of cut to be applied on the charge. Options are "ABSOLUTE",
        "RELATIVE" (default) or None. The parameter *threshold* should vary
        according to this flag. If None, it will be ignored.

    Raises
    ------
    ValueError : if cut_type does not match any of the available options.
    """
    fig = plt.figure()
    xmin, xmax = np.nanmin(sipmdf["X"].values), np.nanmax(sipmdf["X"].values)
    ymin, ymax = np.nanmin(sipmdf["Y"].values), np.nanmax(sipmdf["Y"].values)
    if cut_type is None:
        selection = np.ones(slice.size, dtype=bool)
    elif cut_type.upper() == "RELATIVE":
        selection = slice > np.nanmax(slice) * threshold
    elif cut_type.upper() == "ABSOLUTE":
        selection = slice > threshold
    else:
        raise ValueError("cut_type value not recognized")

    x, y = sipmdf["X"][selection], sipmdf["Y"][selection]
    q = slice[selection]

    plt.scatter(x, y, c=q)
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))
    plt.colorbar().set_label("Charge (pes)")


def plot_anode_sum(pmap, sipmdf, threshold=0.1, cut_type="RELATIVE"):
    """Shortcut for plotting a pmap as a z-collapsed slice.

    Parameters
    ----------
    pmap : Bridges.PMap
        A pmap of any event.

    Other parameters are the ones from plot_anode_slice.
    """
    slice = np.nansum(np.concatenate([peak.anode for peak in pmap]), axis=0)
    plot_anode_slice(slice, sipmdf, threshold, cut_type)


def plot_track(geom_df, mchits_df, vox_size=10, zoom=False):
    """Plot the hits of a mctrk. Adapted from JR plotting functions notice
    that geom_df and mchits_df are pandas objects defined above if
    zoom = True, the track is zoomed in (min/max dimensions are taken
    from track). If zoom = False, detector dimensions are used

    """
    grdcol = 0.99

    varr_x = mchits_df["x"].values * vox_size
    varr_y = mchits_df["y"].values * vox_size
    varr_z = mchits_df["z"].values * vox_size
    varr_c = mchits_df["energy"].values / units.keV

    min_x = geom_df["xdet_min"] * vox_size
    max_x = geom_df["xdet_max"] * vox_size
    min_y = geom_df["ydet_min"] * vox_size
    max_y = geom_df["ydet_max"] * vox_size
    min_z = geom_df["zdet_min"] * vox_size
    max_z = geom_df["zdet_max"] * vox_size
    emin = 0
    emax = np.max(varr_c)

    if zoom is True:
        min_x = np.min(varr_x)
        max_x = np.max(varr_x)
        min_y = np.min(varr_y)
        max_y = np.max(varr_y)
        min_z = np.min(varr_z)
        max_z = np.max(varr_z)
        emin  = np.min(varr_c)

    # Plot the 3D voxelized track.
    fig = plt.figure(1)
    fig.set_figheight(6)
    fig.set_figwidth(8)

    ax1 = fig.add_subplot(111, projection="3d")
    s1 = ax1.scatter(varr_x, varr_y, varr_z, marker="s", linewidth=0.5,
                     s=2*vox_size, c=varr_c, cmap=plt.get_cmap("rainbow"),
                     vmin=emin, vmax=emax)

    # this disables automatic setting of alpha relative of distance to camera
    s1.set_edgecolors = s1.set_facecolors = lambda *args: None

    print(" min_x ={} max_x ={}".format(min_x, max_x))
    print(" min_y ={} max_y ={}".format(min_y, max_y))
    print(" min_z ={} max_z ={}".format(min_z, max_z))
    print("min_e ={} max_e ={}".format(emin, emax))

    ax1.set_xlim([min_x, max_x])
    ax1.set_ylim([min_y, max_y])
    ax1.set_zlim([min_z, max_z])

    #    ax1.set_xlim([0, 2 * vox_ext]);
    #    ax1.set_ylim([0, 2 * vox_ext]);
    #    ax1.set_zlim([0, 2 * vox_ext]);
    ax1.set_xlabel("x (mm)")
    ax1.set_ylabel("y (mm)")
    ax1.set_zlabel("z (mm)")
    ax1.set_title("")

    lb_x = ax1.get_xticklabels()
    lb_y = ax1.get_yticklabels()
    lb_z = ax1.get_zticklabels()
    for lb in (lb_x + lb_y + lb_z):
        lb.set_fontsize(8)

    ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    for axis in [ax1.w_xaxis, ax1.w_yaxis, ax1.w_zaxis]:
        axis._axinfo.update({"grid": {"color": (grdcol, grdcol, grdcol, 1)}})

    cb1 = plt.colorbar(s1)
    cb1.set_label("Energy (keV)")

    plt.show()


def plot_track_projections(geom_df, mchits_df, vox_size=10, zoom=False):
    """Plot the projections of an MC track. Adapted from function
    plot_track above notice that geom_df and mchits_df are pandas
    objects defined above if zoom = True, the track is zoomed in
    (min/max dimensions are taken from track). If zoom = False,
    detector dimensions are used.

    For now, it is assumed that vox_sizeX = vox_sizeY = vox_sizeZ
    """
    vox_sizeX = vox_size
    vox_sizeY = vox_size
    vox_sizeZ = vox_size

    varr_x = mchits_df["x"].values * vox_size
    varr_y = mchits_df["y"].values * vox_size
    varr_z = mchits_df["z"].values * vox_size
    varr_c = mchits_df["energy"].values/units.keV

    min_x = geom_df["xdet_min"] * vox_size
    max_x = geom_df["xdet_max"] * vox_size
    min_y = geom_df["ydet_min"] * vox_size
    max_y = geom_df["ydet_max"] * vox_size
    min_z = geom_df["zdet_min"] * vox_size
    max_z = geom_df["zdet_max"] * vox_size

    if zoom is True:
        min_x = np.min(varr_x)
        max_x = np.max(varr_x)
        min_y = np.min(varr_y)
        max_y = np.max(varr_y)
        min_z = np.min(varr_z)
        max_z = np.max(varr_z)

    # Plot the 2D projections.
    fig = plt.figure(1)
    fig.set_figheight(5.)
    fig.set_figwidth(20.)

    # Create the x-y projection.
    ax1 = fig.add_subplot(131)
    hxy, xxy, yxy = np.histogram2d(varr_y, varr_x,
                                   weights=varr_c, normed=False,
                                   bins= ((1.05 * max_y - 0.95 * min_y) / vox_sizeY,
                                          (1.05 * max_x - 0.95 * min_x) / vox_sizeX),
                                   range=([0.95 * min_y,  1.05 * max_y],
                                          [0.95 * min_x,  1.05 * max_x]))

    extent1 = [yxy[0], yxy[-1], xxy[0], xxy[-1]]
    sp1 = ax1.imshow(hxy, extent=extent1, interpolation="none",
                     aspect="auto", origin="lower")
    ax1.set_xlabel("x (mm)")
    ax1.set_ylabel("y (mm)")
    cbp1 = plt.colorbar(sp1)
    cbp1.set_label("Energy (keV)")

    # Create the y-z projection.
    ax2 = fig.add_subplot(132)
    hyz, xyz, yyz = np.histogram2d(varr_z, varr_y,
                                   weights=varr_c, normed=False,
                                   bins= ((1.05 * max_z - 0.95 * min_z) / vox_sizeZ,
                                          (1.05 * max_y - 0.95 * min_y) / vox_sizeY),
                                   range=([0.95 * min_z,  1.05 * max_z],
                                          [0.95 * min_y,  1.05 * max_y]))

    extent2 = [yyz[0], yyz[-1], xyz[0], xyz[-1]]
    sp2 = ax2.imshow(hyz, extent=extent2, interpolation="none",
                     aspect="auto", origin="lower")
    ax2.set_xlabel("y (mm)")
    ax2.set_ylabel("z (mm)")
    cbp2 = plt.colorbar(sp2)
    cbp2.set_label("Energy (keV)")

    # Create the x-z projection.
    ax3 = fig.add_subplot(133)
    hxz, xxz, yxz = np.histogram2d(varr_z, varr_x,
                                   weights=varr_c, normed=False,
                                   bins= ((1.05 * max_z - 0.95 * min_z) / vox_sizeZ,
                                          (1.05 * max_x - 0.95 * min_x) / vox_sizeX),
                                   range=([0.95 * min_z,  1.05 * max_z],
                                          [0.95 * min_x,  1.05 * max_x]))

    extent3 = [yxz[0], yxz[-1], xxz[0], xxz[-1]]
    sp3 = ax3.imshow(hxz, extent=extent3, interpolation="none",
                     aspect="auto", origin="lower")
    ax3.set_xlabel("x (mm)")
    ax3.set_ylabel("z (mm)")
    cbp3 = plt.colorbar(sp3)
    cbp3.set_label("Energy (keV)")

    plt.show()


def make_movie(slices, sipmdf, thrs=0.1):
    """Create a video made of consecutive frames showing the response of
    the tracking plane.

    Parameters
    ----------
    slices : 2-dim np.ndarray
        The signal of each SiPM (axis 1) for each time sample (axis 0).
    sipmdf : pd.DataFrame
        Contains the sensors information.
    thrs : float, optional
        Default cut value to be applied to each slice. Defaults to 0.1.

    Returns
    -------
    mov : matplotlib.animation
        The movie.
    """

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8)
    X, Y = sipmdf["X"].values, sipmdf["Y"].values
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
        scplot = ax.scatter(x, y, c=q, marker="s", vmin=0, vmax=np.nanmax(slices))
        cbar = fig.colorbar(scplot, ax=ax, boundaries=np.linspace(0, np.nanmax(slices), 100))
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
