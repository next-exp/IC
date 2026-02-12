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

from invisible_cities.types.symbols import SelRegionMethod, NormMethod
from invisible_cities.core.core_functions import in_range, shift_to_bin_centers
from invisible_cities.core.fit_functions import fit, profileX, expo, sigmoid
from scipy import stats

from invisible_cities.icaros.lifetime_vdrift_functions import select_lifetime_region


#
#---- plotting
#

dtbins     = np.linspace(0, 1800, 101)
dtrmsbins  = np.linspace(0, 10, 101)
dtrms2bins = np.linspace(0, 55, 101)
ebins      = np.linspace(0, 15e3, 101)

freq = lambda : plt.ylabel("frequency")



def monitor_S1(df         : pd.DataFrame,
               df2        : pd.DataFrame,
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

    df1 = df.groupby("event s1_peak".split()).first()
    df1_ = df1.groupby('event').count()
    df1__ = df1.groupby('event').mean()

    df2 = df2.groupby("event s1_peak".split()).first()
    df2_ = df2.groupby('event').count()
    df2__ = df2.groupby('event').mean()

    axs[0, 0].hist(df1_.nS1, ns1bins,
                   histtype='step', color = 'mediumpurple', lw = 2, label=
                   f'run: {run_number}\n'
                   f'events: {nevents}\n'
                   f'mean: {df1_.nS1.mean():.2f}\n'
                   f'std: {df1_.nS1.std():.2f}')
    axs[0, 0].hist(df2_.nS1, ns1bins,
                    histtype='step', color = 'black', lw = 2, label=
                   f'run: {run_number} after selection\n'
                   f'events: {nevents}\n'
                   f'mean: {df2_.nS1.mean():.2f}\n'
                   f'std: {df2_.nS1.std():.2f}')
    axs[0, 0].set_xlabel('Number of S1')
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_title('nS1 distribution')
    axs[0, 0].grid(True)
    axs[0, 0].legend()


    axs[0, 1].hist(df1.S1e, ebins,
                   histtype='step',color = 'mediumpurple', lw = 2, label=
                   f'run: {run_number}\n'
                   f'events: {nevents}\n'
                   f'mean: {df1.S1e.mean():.2f}\n'
                   f'std: {df1.S1e.std():.2f}')
    axs[0, 1].hist(df2__.S1e, ebins,
                   histtype='step',color = 'black',lw = 2, label=
                   f'run: {run_number} after selection\n'
                   f'events: {nevents}\n'
                   f'mean: {df2__.S1e.mean():.2f}\n'
                   f'std: {df2__.S1e.std():.2f}')
    axs[0, 1].set_xlabel('S1e (pe)')
    axs[0, 1].set_title('S1e distribution')
    axs[0, 1].grid(True)
    axs[0, 1].legend()


    axs[1, 0].hist(df1.S1h, s1hbins,
                    histtype='step', color = 'mediumpurple',lw = 2, label=
                   f'run: {run_number}\n'
                   f'events: {nevents}\n'
                   f'mean: {df1.S1h.mean():.2f}\n'
                   f'std: {df1.S1h.std():.2f}')
    axs[1, 0].hist(df2__.S1h, s1hbins,
                   histtype='step',color = 'black', lw = 2, label=
                   f'run: {run_number} after selection\n'
                   f'events: {nevents}\n'
                   f'mean: {df2__.S1h.mean():.2f}\n'
                   f'std: {df2__.S1h.std():.2f}')
    axs[1, 0].set_xlabel('S1h (pe)')
    axs[1, 0].set_title('S1h distribution')
    axs[1, 0].grid(True)
    axs[1, 0].legend()


    axs[1, 1].hist(df1.S1w, s1wbins,
                   histtype='step',color = 'mediumpurple', lw = 2, label=
                   f'run: {run_number}\n'
                   f'events: {nevents}\n'
                   f'mean: {df1.S1w.mean():.2f}\n'
                   f'std: {df1.S1w.std():.2f}')
    axs[1, 1].hist(df2__.S1w, s1wbins,
                   histtype='step', color = 'black', lw = 2, label=
                   f'run: {run_number} after selection\n'
                   f'events: {nevents}\n'
                   f'mean: {df2__.S1w.mean():.2f}\n'
                   f'std: {df2__.S1w.std():.2f}')
    axs[1, 1].set_xlabel('S1w (pe)')
    axs[1, 1].set_title('S1w distribution')
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    fig.tight_layout();



def monitor_S2(df          : pd.DataFrame,
               df2         : pd.DataFrame,
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

    df_ = df.groupby("event s2_peak".split()).first()
    df1 = df_.groupby('event').count()
    df1_ = df_.groupby('event').mean()

    df2 = df2.groupby("event s2_peak".split()).first()
    df2_ = df2.groupby('event').count()
    df2__ = df2_.groupby('event').mean()

    axs[0, 0].hist(df1.nS2, ns2bins, histtype = 'step', color = 'mediumpurple',lw = 2, label =
                   f'run: {run_number}\n'
                   f'events: {nevents}\n'
                   f'mean: {df1.qmax.mean():.2f}\n'
                   f'std: {df1.qmax.std():.2f}')
    axs[0, 0].hist(df2_.nS2, ns2bins, histtype = 'step', color = 'black', lw = 2, label =
                   f'run: {run_number} after selection\n'
                   f'events: {nevents}\n'
                   f'mean: {df2_.qmax.mean():.2f}\n'
                   f'std: {df2_.qmax.std():.2f}')
    axs[0, 0].set_xlabel('Number of S2')
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_title('nS2 distribution')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    axs[0, 1].hist(df_.S2e, ebins,
                   histtype='step', color = 'mediumpurple', lw = 2, label=
                   f'run: {run_number}\n'
                   f'events: {nevents}\n'
                   f'mean: {df_.S2e.mean():.2f}\n'
                   f'std: {df_.S2e.std():.2f}')
    axs[0, 1].hist(df2.S2e, ebins,
                   histtype='step', color = 'black', lw = 2, label=
                   f'run: {run_number} after selection\n'
                   f'events: {nevents}\n'
                   f'mean: {df2.S2e.mean():.2f}\n'
                   f'std: {df2.S2e.std():.2f}')
    axs[0, 1].set_xlabel('S2e (pe)')
    axs[0, 1].set_title('S2e distribution')
    axs[0, 1].grid(True)
    axs[0, 1].legend()


    axs[1, 0].hist(df_.S2h, s2hbins,
                   histtype='step',color = 'mediumpurple', lw = 2, label=
                   f'run: {run_number}\n'
                   f'events: {nevents}\n'
                   f'mean: {df_.S2h.mean():.2f}\n'
                   f'std: {df_.S2h.std():.2f}')

    axs[1, 0].hist(df2.S2h, s2hbins,
                   histtype='step', color = 'black', lw = 2, label=
                   f'run: {run_number} after selection\n'
                   f'events: {nevents}\n'
                   f'mean: {df2.S2h.mean():.2f}\n'
                   f'std: {df2.S2h.std():.2f}')
    axs[1, 0].set_xlabel('S2h (pe)')
    axs[1, 0].set_title('S2h distribution')
    axs[1, 0].grid(True)
    axs[1, 0].legend()


    axs[1, 1].hist(df_.S2q, s2qbins, histtype = 'step', color = 'mediumpurple', lw = 2,
                   label =
                   f'run: {run_number}\n'
                   f'events: {nevents}\n'
                   f'mean: {df_.S2q.mean():.2f}\n'
                   f'std: {df_.S2q.std():.2f}')
    axs[1, 1].hist(df2.S2q, s2qbins, histtype = 'step', color = 'black', lw = 2,
                   label =
                   f'run: {run_number} after selection\n'
                   f'events: {nevents}\n'
                   f'mean: {df2.S2q.mean():.2f}\n'
                   f'std: {df2.S2q.std():.2f}')
    axs[1, 1].set_xlabel('S2q (pe)')
    axs[1, 1].set_title('S2q distribution')
    axs[1, 1].grid(True)
    axs[1, 1].legend()


    axs[2, 0].hist(df_.qmax, qmaxbins, histtype = 'step', color = 'mediumpurple', lw = 2,
                   label =
                   f'run: {run_number}\n'
                   f'events: {nevents}\n'
                   f'mean: {df_.qmax.mean():.2f}\n'
                   f'std: {df_.qmax.std():.2f}')
    axs[2, 0].hist(df2.qmax, qmaxbins, histtype = 'step', color = 'black', lw = 2,
                   label =
                   f'run: {run_number} after selection\n'
                   f'events: {nevents}\n'
                   f'mean: {df2.qmax.mean():.2f}\n'
                   f'std: {df2.qmax.std():.2f}')
    axs[2, 0].set_xlabel('qmax (pe)')
    axs[2, 0].set_title('Q max distribution')
    axs[2, 0].grid(True)
    axs[2, 0].legend()


    axs[2, 1].hist(df_.S2w, s2wbins , histtype = 'step', color = 'mediumpurple', lw = 2,
                   label =
                   f'run: {run_number}\n'
                   f'events: {nevents}\n'
                   f'mean: {df_.qmax.mean():.2f}\n'
                   f'std: {df_.qmax.std():.2f}')
    axs[2, 1].hist(df2.S2w, s2wbins , histtype = 'step', color = 'black', lw = 2,
                   label =
                   f'run: {run_number} after selection\n'
                   f'events: {nevents}\n'
                   f'mean: {df2.qmax.mean():.2f}\n'
                   f'std: {df2.qmax.std():.2f}')
    axs[2, 1].set_xlabel('S2w (pe)')
    axs[2, 1].set_title('S2w distribution')
    axs[2, 1].grid(True)
    axs[2, 1].legend()

    fig.tight_layout();




def monitor_dtime(df          : pd.DataFrame,
                  df2         : pd.DataFrame,
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
    df1 = df.groupby('event s2_peak'.split()).first().reset_index()
    df1_ = df1.groupby('event').mean()

    axs[0,0].hist2d(df1.DT, df1.Zrms**2, (dtbins, dtrms2bins));
    axs[0,0].plot(df1.DT, dtrms2_low(df1.DT), ".r", ms=2);
    axs[0,0].plot(df1.DT, dtrms2_upp(df1.DT), ".r", ms=2);
    axs[0,0].plot(df1.DT, dtrms2_cen(df1.DT), '.g', ms = 2);
    axs[0,0].set_xlabel("Drift time ($\mu$s)"); axs[0,0].set_ylabel("DT$_{rms}^2$ ($\mu$s)"); axs[0,0].set_xlim(0, 1300)
    axs[0, 0].set_title('Before selection')


    axs[0,1].hist2d(df2.DT, df2.Zrms**2, (dtbins, dtrms2bins));
    axs[0,1].plot(df2.DT, dtrms2_low(df2.DT), ".r", ms=2);
    axs[0,1].plot(df2.DT, dtrms2_upp(df2.DT), ".r", ms=2);
    axs[0,1].plot(df2.DT, dtrms2_cen(df2.DT), '.g', ms = 2);
    axs[0,1].set_xlabel("Drift time ($\mu$s)"); axs[0,1].set_ylabel("DT$_{rms}^2$ ($\mu$s)"); axs[0,1].set_xlim(0, 1300)
    axs[0,1].set_title('After selection')

    axs[1,0].hist(df1_.DT, dtbins, histtype = 'step', color = 'mediumpurple',lw = 2,
                  label = 'before selection');
    axs[1,0].hist(df2.DT, dtbins, histtype = 'step', color = 'black', lw = 2,
                  label = 'after selection');
    axs[1,0].legend();
    axs[1,0].set_xlabel("Drift time ($\mu$s)");
    axs[1,0].grid(True)

    axs[1,1].hist(df1_.Zrms**2, 100, (0, 40), histtype = 'step',color = 'mediumpurple',lw = 2,
                  label = 'before selection');
    axs[1,1].hist(df2.Zrms**2, 100, (0, 40), histtype = 'step', color = 'black',lw = 2,
                  label = 'after selection');
    axs[1,1].legend();
    axs[1,1].set_xlabel("DT$_{rms}^2$ ($\mu$s)");
    axs[1,1].grid(True)

    fig.tight_layout();


def monitor_lifetime(df        : pd.DataFrame,
                     ebins    : np.array,
                     dtbins     : np.array):
    """
    Plots a 2D histogram of S2e vs drift time.
    """
    fig, axs = plt.subplots(1, 1)
    axs.hist2d(df.DT, df.S2e, (dtbins, ebins));
    axs.set_xlabel(r"DT ($\mu$s)");
    axs.set_ylabel("S2e (pe)");
    #axs.set_xlim(0, 1500)

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


    fig, axs = plt.subplots(1, 2, figsize = (21, 7))

    axs[0].hist(R2, bins, histtype = 'step', color = 'mediumpurple', lw = 2);
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

    fig, axs = plt.subplots(1, 1)


    h, xedges, yedges, im = axs.hist2d(
        x_grid,
        y_grid,
        bins=ebins,
        weights=weight,
        cmin=6000,
        cmax = 10000
    )


    c = fig.colorbar(im, ax=axs)

    if statistic == 'mean':
        c.ax.set_ylabel('Mean S2e (pe)')

    if statistic == 'counts':
        c.ax.set_ylabel('Average number of events')

    axs.set_xlim(-500, 500)
    axs.set_ylim(-500, 500)
    axs.set_xlabel("X (mm)")
    axs.set_ylabel("Y (mm)")
    axs.set_title(f'Run {run}')

    fig.tight_layout()



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
    median_Ec2 = Ec_2.median()

    mean_Ec =Ec_1.mean()
    stdEc = Ec_1.std()
    umeanEc = Ec_1.std()/np.sqrt(len(Ec_2))
    median_Ec = Ec_1.median()

    fig, axs = plt.subplots(1, 1)
    axs.hist(Ec_1, 100, range = (25, 60), histtype = 'step', color = 'black', lw = 2,
      label = f'mean Ec: {mean_Ec:.2f}keV\n'
      f'median Ec: {median_Ec:.2f}keV\n'
      f'std Ec: {stdEc:.2f}keV\n'
      f'umean Ec: {umeanEc:.2f}keV')

    axs.hist(Ec_2, 100, range = (25, 60), histtype = 'step', color = 'mediumpurple',lw = 2,
      label = f'mean Ec_2: {mean_Ec2:.2f}keV\n'
      f'median Ec_2: {median_Ec2:.2f}keV\n'
      f'std Ec_2: {stdEc2:.2f}keV\n'
      f'umean Ec_2: {umeanEc2:.2f}keV')
    axs.set_xlabel('Ec (keV)'); freq();
    axs.grid();
    axs.legend();


    fig.tight_layout();


def plot_lifetime_fit(df         : pd.DataFrame,
                      x0         : float,
                      y0         : float,
                      shape      : SelRegionMethod.circle,
                      shape_size : float,
                      dtbins     : np.array,
                      ebins      : np.array):

    """
    Plots a 2D histogram of DT vs S2e.
    Computes a fit to the lifetime and calculates and plots its profile.
    -df should be df after selections, to then apply select_lifetime_region
     so the fit works properly.
    """

    df_in_region = select_lifetime_region(df, x0, y0, shape, shape_size)

    f  = fit(expo, df_in_region.DT, df_in_region.S2e, seed = [8000, -30000]);
    magnitudes = f.values
    uncertainties = (f[2][0], f[2][1])

    fig, axs = plt.subplots(1, 1)
    axs.hist2d(df_in_region.DT, df_in_region.S2e, (dtbins, ebins), cmin = 0.01);
    axs.set_xlabel(r"DT ($\mu$s)");
    axs.set_ylabel("S2e (pe)");
    axs.set_xlim(0, 1400)

    const = magnitudes[0]
    lifetime = - magnitudes[1]
    dt, e, se = profileX(df_in_region.DT, df_in_region.S2e, std = False, nbins = 20)

    axs.plot(dt, const*np.exp(-dt/lifetime), color = 'red',
             label =f'{const:.2f}'r'$ \cdot e^{(-dt/'f'{lifetime:.2f}''})$\n'
                    f'u_const : {uncertainties[0]:.2f}\n'
                    f'u_lifetime : {uncertainties[1]:.2f}');
    axs.errorbar(dt, e, yerr = se, fmt = '.');
    axs.set_ylim(6000, 10000);
    axs.legend();



def plot_sigmoid(df         : pd.DataFrame,
                 x0         : float,
                 y0         : float,
                 shape      : SelRegionMethod.circle,
                 shape_size : float):

    """
    Plots sigmoid fit to drift time using selected dataframe
    (we apply select_lifetime_region() function to selected dataframe)
    """

    df_in_region = select_lifetime_region(df, x0, y0, shape, shape_size)

    counts, bins = np.histogram(df_in_region.DT, bins = 20, range = (1200, 1500))
    bin_centers = shift_to_bin_centers(bins)
    f = fit(sigmoid, bin_centers, counts, seed = [1000, 1400, 0, 0.1])

    fig, axs = plt.subplots(1, 1)

    axs.plot(bin_centers, counts, 'o', color = 'black', markersize = 5, label = 'DT mean')
    axs.plot(bin_centers, sigmoid(bin_centers, *f.values), color = 'red', label = f'Sigmoid_fit')
    axs.set_xlabel(r'DT($\mu$s)');
    axs.set_ylabel('Event distribution');
    axs.set_xlim(1200, 1500);
    axs.grid(True);
    axs.legend()


def plot_XY_distributions(df         : pd.DataFrame,
                          df2        : pd.DataFrame,
                          run_number : int,
                          xy_range   : np.array):
    """
    Plots histograms for X and Y before and after selections.
    Parameters
    ----------
    df : pd.DataFrame.
      Initial dataframe, Sophronia output.
    df2 : pd.DataFrame.
      Dataframe after performing selections.
    xy_range : np.array.
      Ideally a linspace withing the x,y limits (-500, 500) and specifying the number of bins
    run_number : int.
      Run number.
    Returns
    -------
    Histograms for X and Y of both dataframes.
    """

    fig, axs = plt.subplots(1, 2, figsize=(21, 7))
    df_ = df.groupby('event s2_peak'.split()).first().reset_index()

    axs[0].hist(df.X.values, xy_range, histtype = 'step', color = 'mediumpurple',lw = 2, label = 'before selection');
    axs[0].hist(df2.X.values, xy_range, histtype = 'step', color = 'black',lw = 2, label = 'after selection');
    axs[0].set_xlabel('X (mm)');
    axs[0].legend();
    axs[0].grid();
    axs[0].set_title(f'{run_number}')


    axs[1].hist(df.Y.values, xy_range, histtype = 'step', color = 'mediumpurple',lw = 2, label = 'before selection');
    axs[1].hist(df2.Y.values, xy_range, histtype = 'step', color = 'black', lw = 2, label = 'after selection');
    axs[1].set_xlabel('Y (mm)');
    axs[1].legend();
    axs[1].grid();
    axs[1].set_title(f'{run_number}')




def make_control_plots(df          : pd.DataFrame,
                       df_sel      : pd.DataFrame,
                       df_corr     : pd.DataFrame,
                       run_number  : int,
                       ebins1      : np.array,
                       ns1bins     : np.array,
                       s1hbins     : np.array,
                       s1wbins     : np.array,
                       ebins2      : np.array,
                       ns2bins     : np.array,
                       s2hbins     : np.array,
                       s2qbins     : np.array,
                       qmaxbins    : np.array,
                       s2wbins     : np.array,
                       dtrms2_low  : Callable,
                       dtrms2_upp  : Callable,
                       dtrms2_cen  : Callable,
                       dtbins2     : np.array,
                       bins        : int,
                       dtr2_bins   : tuple,
                       col_name1   : str,
                       col_name2   : str,
                       statistic   : str,
                       x0          : float,
                       y0          : float,
                       shape       : SelRegionMethod,
                       shape_size  : float,
                       xy_range    : np.array):

    monitor_S1(df, df_sel, run_number, ebins1, ns1bins, s1hbins, s1wbins)

    monitor_S2(df, df_sel, run_number, ebins2, ns2bins, s2hbins, s2qbins, qmaxbins, s2wbins)

    monitor_dtime(df, df_sel, dtrms2_low, dtrms2_upp, dtrms2_cen)

    monitor_lifetime(df, ebins2, dtbins2)

    monitor_kr_distribution(df, bins, dtr2_bins)

    plot_Ec(df_corr[col_name1], df_corr[col_name2])

    hist2D(df_sel, run_number, statistic)

    plot_lifetime_fit(df_sel, x0, y0, shape, shape_size, dtbins2, ebins2)

    plot_sigmoid(df_sel, x0, y0, shape, shape_size)

    plot_XY_distributions(df, df_sel, run_number, xy_range)
