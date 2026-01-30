#----------------------------------------
#
# Functions for Kr analysis of NEXT100
#
#----------------------------------------

import numpy             as np
import pandas            as pd
from   scipy.optimize     import curve_fit

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


def monitor_S1S2(df, run_number):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    nevents = len(df['event'].unique())

    df2 = df.groupby('event s1_peak'.split()).first().reset_index()
    df2 = df2.groupby('event').count()

    axs[0, 0].hist(df2.nS1, bins=10, range=(0, 20),
                   density=True, histtype='step', label=
                   f'run: {run_number}\n'
                   f'events: {nevents}\n'
                   f'mean: {df2_.nS1.mean():.2f}\n'
                   f'std: {df2_.nS1.std():.2f}')
    axs[0, 0].set_xlabel('Number of S1')
    axs[0, 0].set_title('nS1 distribution')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    df1 = df.groupby('event s2_peak'.split()).first().reset_index()
    df1 = df1.groupby('event').count()


    axs[0, 1].hist(df1.nS2, bins=20, range=(0, 20),
                   density=True, histtype='step', label=
                   f'run: {run_number}\n'
                   f'events: {nevents}\n'
                   f'mean: {df1_.nS2.mean():.2f}\n'
                   f'std: {df1_.nS2.std():.2f}')
    axs[0, 1].set_xlabel('Number of S2')
    axs[0, 1].set_title('nS2 distribution')
    axs[0, 1].grid(True)
    axs[0, 1].legend()


    axs[1, 0].hist(df2.S1t, bins=100, histtype = 'step', density=True, label =
                   f'run: {run_number}\n'
                   f'events: {nevents}\n'
                   f'mean: {df.S1t.mean():.2f}\n'
                   f'std: {df.S1t.std():.2f}')
    axs[1, 0].set_xlabel('S1 waveform time')
    axs[1, 0].set_title('S1 time distribution')
    axs[1, 0].grid(True)
    axs[1, 0].legend()


    axs[1, 1].hist(df1.S2t, bins=100, histtype = 'step', density=True, label =
                   f'run: {run_number}\n'
                   f'events: {nevents}\n'
                   f'mean: {df.S2t.mean():.2f}\n'
                   f'std: {df.S2t.std():.2f}')
    axs[1, 1].set_xlabel('S2 waveform time')
    axs[1, 1].set_title('S2 time distribution')
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    fig.tight_layout()



def monitor_S1(df, ebins, ns1bins, s1hbins, s1wbins):
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



def monitor_S2(df, run_number, ebins, ns2bins, s2hbins, s2qbins, qmaxbins, s2wbins):
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




def monitor_dtime(df, dtrms2_low, dtrms2_upp, dtrms2_cen):

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


def monitor_lifetime(df, S2erange, dtbins, ebins):

    fig, axs = plt.subplots(1, 2, figsize = (15, 5))

    axs[0].hist2d(df.DT, df.S2e, (dtbins, ebins));
    axs[0].set_xlabel(r"DT ($\mu$s)"); axs[0].set_ylabel("S2e (pe)"); axs[0].set_xlim(0, 1500)


    axs[1].scatter(df.DT, df.S2e, alpha = 0.01); axs[1].set_xlim(0, 1500);
    axs[1].set_ylim(*S2erange);
    axs[1].set_xlabel(r"DT ($\mu$s)");
    axs[1].set_ylabel("S2e (pe)");

    fig.tight_layout();



def monitor_kr_distribution(df, bins, dtr2_bins):

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




def hist2D(df, run, statistic):

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
    return values, bin_centers


def plot_Ec(df): 

    mean_Ec2 =df.Ec_2.mean()
    stdEc2 = df.Ec_2.std()
    umeanEc2 = df.Ec_2.std()/np.sqrt(len(df.Ec_2))
    
    mean_Ec =df.Ec.mean()
    stdEc = df.Ec.std()
    umeanEc = df.Ec.std()/np.sqrt(len(df.Ec))
    
    
    plt.hist(df.Ec, 100, range = (20, 60), histtype = 'step', color = 'black', density = True, label = f'mean Ec: {mean_Ec:.2f}keV')
    plt.hist(df.Ec_2, 100, range = (20, 60), histtype = 'step', color = 'red', density = True, label = f'mean Ec_2: {mean_Ec2:.2f}keV'); 
    plt.xlabel('Ec (keV)'); freq();
    plt.grid();
    plt.legend();
    
    
    plt.tight_layout();

    return (mean_Ec2, stdEc2, umeanEc2), (mean_Ec, stdEc, umeanEc)
    

def plot_lifetime_fit(df): #df should be kdst_in_region or the fit won't work 
    
    magnitudes, uncertainties = LT_fit(df.DT, df. S2e, p0 = [8000, -30000]);
    plt.xlabel(r"DT ($\mu$s)"); plt.ylabel("S2e (pe)");
    plt.hist2d(df.DT, df.S2e,(50, 50), cmin = 1e-3);

    const = magnitudes[0]
    lifetime = - magnitudes[1]
    dt, e, se = profileX(df.DT, df.S2e, std = False, nbins = 20)
    
    plt.plot(dt, const*np.exp(-dt/lifetime), color = 'red');
    plt.errorbar(dt, e, yerr = se, fmt = '.');
    plt.ylim(6000, 10000);
     
    return const, lifetime, (uncertainties)


    
def krmap_ratio_hist2D(df1, df2, run1_name, run2_name, bins, statistic):

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
    return ratio



def plot_sigmoid(df): #use kdst_in_region
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
    
    
 
    