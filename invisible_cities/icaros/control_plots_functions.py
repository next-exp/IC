#----------------------------------------
#
# Functions for Kr analysis of NEXT100
#
#----------------------------------------

import numpy             as np
import pandas            as pd
from   scipy.optimize     import curve_fit
from typing import Callable, Union, Tuple

import matplotlib.pyplot as plt


from invisible_cities.core.core_functions import in_range, shift_to_bin_centers
from invisible_cities.core.fit_functions import fit, profileX, expo
from scipy import stats


#
#---- plotting
#

dtbins     = np.linspace(0, 1800, 101)
dtrmsbins  = np.linspace(0, 10, 101)
dtrms2bins = np.linspace(0, 55, 101)
ebins      = np.linspace(0, 15e3, 101)

freq = lambda : plt.ylabel("frequency")



def monitor_S1(df         : pd.DataFrame,
               run_number : int,
               ebins      : np.array,
               ns1bins    : np.array,
               s1hbins    : np.array,
               s1wbins    : np.array):
    """
    Plots distributions for nS1, S1e, S1h, S1w.
    -To get the actual number of S1 and not the times that it is repeated, we use the first
    entry after doing groupby event s2_peak to then groupby event again and .count().
    -Why are we plotting the mean values? I think its ok but think why
    Parameters
    ----------
    df : pd.DataFrame
      Input dataframe (either the kdst before or after applying selections)
    run_number : int
      Number of the run being analyzed
    ebins : np.array
      To set energy limits and range
    ns1bins : np.array
      To set limits and range on nS1
    s1hbins : np.array
      To set limits and range on S1 height
    s1wbins : np.array
      To set limits and range on S1 width
    Returns
    -------
    Histograms of the variables
    """

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    nevents = len(df['event'].unique())

    df1 = df.groupby("event s2_peak".split()).first()
    df1_ = df1.groupby('event').count()
    df1__ = df1.groupby('event').mean()

    axs[0, 0].hist(df1_.nS1, ns1bins,
                   density=True, histtype='step', label=
                   f'run: {run_number}\n'
                   f'events: {nevents}\n'
                   f'mean: {df1_.nS1.mean():.2f}\n'
                   f'std: {df1_.nS1.std():.2f}')
    axs[0, 0].set_xlabel('Number of S1')
    axs[0, 0].set_title('nS1 distribution')
    axs[0, 0].grid(True)
    axs[0, 0].legend()


    axs[0, 1].hist(df1__.S1e, ebins,
                   density=True, histtype='step', label=
                   f'run: {run_number}\n'
                   f'events: {nevents}\n'
                   f'mean: {df1__.S1e.mean():.2f}\n'
                   f'std: {df1__.S1e.std():.2f}')
    axs[0, 1].set_xlabel('S1e (pe)')
    axs[0, 1].set_title('S1e distribution')
    axs[0, 1].grid(True)
    axs[0, 1].legend()


    axs[1, 0].hist(df1__.S1h, s1hbins,
                   density=True, histtype='step', label=
                   f'run: {run_number}\n'
                   f'events: {nevents}\n'
                   f'mean: {df1__.S1h.mean():.2f}\n'
                   f'std: {df1__.S1h.std():.2f}')
    axs[1, 0].set_xlabel('S1h (pe)')
    axs[1, 0].set_title('S1h distribution')
    axs[1, 0].grid(True)
    axs[1, 0].legend()


    axs[1, 1].hist(df1__.S1w, s1wbins,
                   density=True, histtype='step', label=
                   f'run: {run_number}\n'
                   f'events: {nevents}\n'
                   f'mean: {df1__.S1w.mean():.2f}\n'
                   f'std: {df1__.S1w.std():.2f}')
    axs[1, 1].set_xlabel('S1w (pe)')
    axs[1, 1].set_title('S1w distribution')
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    fig.tight_layout();



def monitor_S2(df          : pd.DataFrame,
               run_number  : int,
               ebins       : np.array,
               ns2bins     : np.array,
               s2hbins     : np.array,
               s2qbins     : np.array,
               qmaxbins    : np.array,
               s2wbins     : np.array):
    """
    Plots distributions for nS2, S2e, S2h, S2q, qmax and S2w.
    -To get the actual number of S2 and not the times that it is repeated, we use the first
    entry after doing groupby event s1_peak to then groupby event again and .count().
    -Why are we plotting the mean values? I think its ok but think why
    Parameters
    ----------
    df : pd.DataFrame
      Input dataframe (either the kdst before or after applying selections).
    run_number : int
      Number of the run being analyzed.
    ebins : np.array
      To set energy limits and range.
    ns2bins : np.array
      To set limits and range on nS2.
    s2hbins : np.array
      To set limits and range on S2 height.
    s2qbins : np.array
      To set limits and range on S2 charge.
    qmaxbins : np.array
      To set limits and range on Qmax.
    s2wbins : np.array
      To set limits and range on S2 width.
    Returns
    -------
    Histograms of the variables

    """


    fig, axs = plt.subplots(3, 2, figsize=(15, 12))

    nevents = len(df['event'].unique())

    df_ = df.groupby("event s1_peak".split()).first()
    df2 = df_.groupby('event').count()
    df2_ = df_.groupby('event').mean()

    axs[0, 0].hist(df2.nS2, ns2bins, histtype = 'step', density=True, label =
                   f'run: {run_number}\n'
                   f'events: {nevents}\n'
                   f'mean: {df2.qmax.mean():.2f}\n'
                   f'std: {df2.qmax.std():.2f}')
    axs[0, 0].set_xlabel('Number of S2')
    axs[0, 0].set_title('nS2 distribution')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    axs[0, 1].hist(df2_.S2e, ebins,
                   density=True, histtype='step', label=
                   f'run: {run_number}\n'
                   f'events: {nevents}\n'
                   f'mean: {df2_.S2e.mean():.2f}\n'
                   f'std: {df2_.S2e.std():.2f}')
    axs[0, 1].set_xlabel('S2e (pe)')
    axs[0, 1].set_title('S2e distribution')
    axs[0, 1].grid(True)
    axs[0, 1].legend()


    axs[1, 0].hist(df2_.S2h, s2hbins,
                   density=True, histtype='step', label=
                   f'run: {run_number}\n'
                   f'events: {nevents}\n'
                   f'mean: {df2_.S2h.mean():.2f}\n'
                   f'std: {df2_.S2h.std():.2f}')
    axs[1, 0].set_xlabel('S2h (pe)')
    axs[1, 0].set_title('S2h distribution')
    axs[1, 0].grid(True)
    axs[1, 0].legend()


    axs[1, 1].hist(df2_.S2q, s2qbins, histtype = 'step', density=True, label =
                   f'run: {run_number}\n'
                   f'events: {nevents}\n'
                   f'mean: {df2_.S2q.mean():.2f}\n'
                   f'std: {df2_.S2q.std():.2f}')
    axs[1, 1].set_xlabel('S2q (pe)')
    axs[1, 1].set_title('S2q distribution')
    axs[1, 1].grid(True)
    axs[1, 1].legend()


    axs[2, 0].hist(df2_.qmax, qmaxbins, histtype = 'step', density=True, label =
                   f'run: {run_number}\n'
                   f'events: {nevents}\n'
                   f'mean: {df2_.qmax.mean():.2f}\n'
                   f'std: {df2_.qmax.std():.2f}')
    axs[2, 0].set_xlabel('qmax (pe)')
    axs[2, 0].set_title('Q max distribution')
    axs[2, 0].grid(True)
    axs[2, 0].legend()


    axs[2, 1].hist(df2_.S2w, s2wbins , histtype = 'step', density=True, label =
                   f'run: {run_number}\n'
                   f'events: {nevents}\n'
                   f'mean: {df2_.qmax.mean():.2f}\n'
                   f'std: {df2_.qmax.std():.2f}')
    axs[2, 1].set_xlabel('S2w (pe)')
    axs[2, 1].set_title('S2w distribution')
    axs[2, 1].grid(True)
    axs[2, 1].legend()

    fig.tight_layout();




def monitor_dtime(df          : pd.DataFrame,
                  dtrms2_low  : Callable,
                  dtrms2_upp  : Callable,
                  dtrms2_cen  : Callable):
    """
    Plots 2D histograms of DTrms and square DTrms as a function of drift time
    and the drift time and squared drift time distributions.
    Parameters
    ----------
    df : pd.DataFrame
      Dataframe containing the drift time that we want to monitor.
    dtrms2_low : Callable
      Function of drift time that defines the lower limit of the diffusion band.
    dtrms2_upp : Callable
      Function of drift time that defines the upper limit of the diffusion band.
    dtrms2_cen : Callabla
      Function of drift time that define the center of the diffusion band.
    """

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs[0,0].hist2d(df.DT, df.Zrms, (dtbins, dtrmsbins), cmin = 1);
    axs[0,0].set_xlabel("Drift time ($\mu$s)"); axs[0,0].set_ylabel("DT$_{rms}$ ($\mu$s)"); axs[0,0].set_xlim(0, 1300)

    axs[0,1].hist2d(df.DT, df.Zrms**2, (dtbins, dtrms2bins));
    axs[0,1].plot(df.DT, dtrms2_low(df.DT), ".r", ms=2);
    axs[0,1].plot(df.DT, dtrms2_upp(df.DT), ".r", ms=2);
    axs[0,1].plot(df.DT, dtrms2_cen(df.DT), '.g', ms = 2);
    axs[0,1].set_xlabel("Drift time ($\mu$s)"); axs[0,1].set_ylabel("DT$_{rms}^2$ ($\mu$s)"); axs[0,1].set_xlim(0, 1300)

    axs[1,0].hist(df.DT, dtbins, histtype = 'step');
    axs[1,0].set_xlabel("Drift time ($\mu$s)");
    axs[1,0].grid(True)

    axs[1,1].hist(df.Zrms**2, 100, (0, 40), histtype = 'step');
    axs[1,1].set_xlabel("DT$_{rms}^2$ ($\mu$s)");
    axs[1,1].grid(True)

    fig.tight_layout();


def monitor_lifetime(df        : pd.DataFrame,
                     S2erange  : np.array,
                     dtbins    : np.array):
    """
    Plots a 2D histogram of S2e vs drift time.
    """

    plt.hist2d(df.DT, df.S2e, (dtbins, ebins));
    plt.xlabel(r"DT ($\mu$s)");
    plt.ylabel("S2e (pe)");
    plt.xlim(0, 1500)

    fig.tight_layout();



def monitor_kr_distribution(df        : pd.DataFrame,
                            bins      : int,
                            dtr2_bins : tuple):
    """
    Plots the square radial distribution and a 2D distribution of the
    square radius as a function of drift time.
    """

    sel = in_range(df.S2e, 7.5e3, 9.5e3) & in_range(df.DT, 20, 1350)


    DT = (df.DT[sel]).dropna()
    R2 = (df.X[sel]**2 + df.Y[sel]**2).dropna()


    fig, axs = plt.subplots(1, 2, figsize = (15, 5))

    axs[0].hist(R2, bins, histtype = 'step', density = True);
    axs[0].set_xlabel("R$^2$ (mm$^2$)"); freq();
    axs[0].grid(True)


    axs[1].hist2d(DT, R2, dtr2_bins);
    axs[1].set_xlabel("DT ($\mu$s)");
    axs[1].set_ylabel("R$^2$ (mm$^2$)");




def hist2D(df        : pd.DataFrame,
           run       : int,
           statistic : str):
    """
    Plots a 2D histogram or "map" in X,Y where each bin contains the specified
    statistic (mean, counts...) of the S2e (pe)
    """

    df = df.dropna(subset=['X', 'Y'])

    bins  = 100

    xrange = (df.X.min(), df.X.max())
    yrange = (df.Y.min(), df.Y.max())


    values, ebins, _  = stats.binned_statistic_dd((
                      df.X, df.Y), df.S2e,
                      bins=[np.linspace(*xrange, bins), np.linspace(*yrange, bins)], statistic = statistic
                      )

    bin_centers = [0.5 * (b[1:] + b[:-1]) for b in ebins]
    mesh = np.meshgrid(*bin_centers)
    x_grid = mesh[0].ravel()
    y_grid = mesh[1].ravel()
    weight = values.T.ravel()

    plt.hist2d(x_grid, y_grid, bins = ebins, weights = weight, cmin = 0.01);
    c = plt.colorbar();
    if statistic == 'mean':
        c.ax.set_ylabel('Mean S2e (pe)');

    if statistic == 'counts':
        c.ax.set_ylabel('Average number of events');

    plt.xlim(-500, 500);
    plt.ylim(-500, 500);
    plt.xlabel("X (mm)"); plt.ylabel("Y (mm)"); plt.title(f'Run {run}')


    plt.tight_layout();


def plot_Ec(Ec_1  : pd.core.series.Series,
            Ec_2  : pd.core.series.Series) -> [Tuple[float, float, float], Tuple[float, float, float]]:
    """
    This function is for comparing specifically the distributions of corrected energy
    from applying the preliminary map (Ec) vs the "final" corrected energy
    (from applying the self map and all the correction chain, Ec_2)
    """
    mean_Ec2 = Ec_2.mean()
    stdEc2 = Ec_2.std()
    umeanEc2 = Ec_2.std()/np.sqrt(len(Ec_2))

    mean_Ec =Ec_1.mean()
    stdEc = Ec_1.std()
    umeanEc = Ec_1.std()/np.sqrt(len(Ec_2))


    plt.hist(Ec_1, 100, range = (20, 60), histtype = 'step', color = 'black', density = True, label = f'mean Ec: {mean_Ec:.2f}keV')
    plt.hist(Ec_2, 100, range = (20, 60), histtype = 'step', color = 'red', density = True, label = f'mean Ec_2: {mean_Ec2:.2f}keV');
    plt.xlabel('Ec (keV)'); freq();
    plt.grid();
    plt.legend();


    plt.tight_layout();

    return (mean_Ec2, stdEc2, umeanEc2), (mean_Ec, stdEc, umeanEc)


def plot_lifetime_fit(df : pd.DataFrame) -> [[float, float], Tuple[float, float]]:

    """
    Plots a 2D histogram of DT vs S2e.
    Computes a fit to the lifetime and calculates and plots its profile.
    -df should be kdst_in_region (from applying select_lifetime_region() function)
    or the fit won't work.
    """

    f  = fit(expo, df.DT, df. S2e, seed = [8000, -30000]);
    magnitudes = f.values
    uncertainties = (f[2][0], f[2][1])
    plt.xlabel(r"DT ($\mu$s)"); plt.ylabel("S2e (pe)");
    plt.hist2d(df.DT, df.S2e,(50, 50), cmin = 1e-3);

    const = magnitudes[0]
    lifetime = - magnitudes[1]
    dt, e, se = profileX(df.DT, df.S2e, std = False, nbins = 20)

    plt.plot(dt, const*np.exp(-dt/lifetime), color = 'red');
    plt.errorbar(dt, e, yerr = se, fmt = '.');
    plt.ylim(6000, 10000);

    return const, lifetime, (uncertainties)



def krmap_ratio_hist2D(df1        : pd.DataFrame,
                       df2        : pd.DataFrame,
                       run1_name  : Union[str, int],
                       run2_name  : Union[str, int],
                       bins       : int,
                       statistic  : str):
    """
    Plots ratio of the specified statistic (mean, counts...) on a 2D histogram
    for 2 different runs, ideally one run and the reference run used in the preliminary map.
    """

    xrange = (-500, 500)
    yrange = (-500, 500)

    n1 = len(df1.X)
    n2 = len(df2.X)
    df1 = df1.dropna(subset=['X', 'Y'])
    df2 = df2.dropna(subset=['X', 'Y'])

    values1, bins, _ = stats.binned_statistic_dd(
        (df1.X, df1.Y), df1.S2e,
        bins=[np.linspace(*xrange, bins), np.linspace(*yrange, bins)],
        statistic = statistic
    )

    values2, bins, _ = stats.binned_statistic_dd(
        (df2.X, df2.Y), df2.S2e,
        bins=ebins,
        statistic= statistic
    )


    # Calculates ratio avoiding dividing by 0
    with np.errstate(divide='ignore', invalid='ignore'):
        counts1_norm = counts1/n1
        counts2_norm = counts2/n2
        ratio = np.true_divide(counts1_norm, counts2_norm)
        ratio[~np.isfinite(ratio)] = np.nan  # writes NaN instead of inf

    bin_centers = [0.5 * (b[1:] + b[:-1]) for b in ebins]
    mesh = np.meshgrid(*bin_centers)
    x_grid = mesh[0].ravel()
    y_grid = mesh[1].ravel()
    ratio_flat = ratio.T.ravel()  # Transpose bc mean is in [Xbins, Ybins]

    plt.figure(figsize=(8,6))
    plt.hist2d(x_grid, y_grid, bins=ebins, weights=ratio_flat, cmin=1e-7, cmax = 2)
    cbar = plt.colorbar()
    cbar.set_label(f'Ratio {statistic} {run1_name} / {run2_name}')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.title(f'Ratio {statistic}: {run1_name} / {run2_name}')
    plt.xlim(*xrange)
    plt.ylim(*yrange)
    plt.tight_layout()


    ratio = ratio[~np.isnan(ratio)]
    #return ratio



def plot_sigmoid(df : pd.DataFrame):

    """
    Plots sigmoid fit to drift time using kdst_in_region
    (from applying select_lifetime_region() function to the kdst)
    """

    counts, bins, _ = np.histogram(df.DT, bins = 20, range = (1200, 1500))
    bin_centers = shift_to_bin_centers(bins)
    f = fit(sigmoid, bin_centers, counts, seed = [1000, 1400, 0, 0.1])

    plt.plot(bin_centers, counts, 'o', color = 'black', markersize = 5, label = 'DT mean')
    plt.plot(bin_centers, sigmoid(bin_centers, *f.values), color = 'red', label = f'Sigmoid_fit')
    plt.xlabel(r'DT($\mu$s)');
    plt.ylabel('Event distribution');
    plt.xlim(1200, 1500);
    plt.grid(True);
    plt.legend()
