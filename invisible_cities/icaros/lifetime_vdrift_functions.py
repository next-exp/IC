import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.patches import Circle, Rectangle
from typing import Tuple
from invisible_cities.core.stat_functions import poisson_sigma
import invisible_cities.core.fit_functions as fit



def select_lifetime_region(df, bins, x0, y0, r, half_size, circle=True):

    df = df.dropna(subset=['X','Y','S2e', 'DT'])

    xmin, xmax = df.X.min(), df.X.max()
    ymin, ymax = df.Y.min(), df.Y.max()

    mean, ebins, _  = stats.binned_statistic_dd(
        (df.X, df.Y), df.S2e,
        bins=[np.linspace(xmin, xmax, bins),
              np.linspace(ymin, ymax, bins)],
        statistic="mean"
    )

    x_centers = shift_to_bin_centers(ebins[0])
    y_centers = shift_to_bin_centers(ebins[1])

    Xc,Yc = np.meshgrid(x_centers, y_centers)


    if circle == False:

        x1, y1 = x0-half_size, y0-half_size
        mask = ((Xc >= x1) & (Xc <= x1+2*half_size) & (Yc >= y1) & (Yc <= y1+2*half_size))
        mean_values = mean.T[mask]


        mask_df = ((df.X >= x1) & (df.X <= x1+2*half_size) & (df.Y >= y1) &
                   (df.Y <= y1+2*half_size))

        df_region = df[mask_df]

    else:
        mask = (Xc - x0)**2 + (Yc - y0)**2 <= r**2
        mean_values = mean.T[mask]

        mask_df = (df.X - x0)**2 + (df.Y - y0)**2 <= r**2
        df_region = df[mask_df]


    return df_region, mean_values



def LT_fit(DT, s2e, p0):

    def exponential(t, e, tau):
        return e*np.exp(-t/tau)

    popt, pcov = curve_fit(exponential, DT, s2e, p0 = p0)
    magnitudes = (popt[0], popt[1])
    uncertainties = (np.sqrt(pcov[0,0]), np.sqrt(pcov[1,1]))
    return magnitudes, uncertainties



def sigmoid(x          : np.array,
            scale      : float,
            inflection : float,
            slope      : float,
            offset     : float)->np.array:
    '''
    Sigmoid function, it computes the sigmoid of the input array x using the specified
    parameters for scaling, inflection point, slope, and offset.
    Parameters
    ----------
    x : np.array
        The input array.
    scale : float
        The scaling factor determining the maximum value of the sigmoid function.
    inflection : float
        The x-value of the sigmoid's inflection point (where the function value is half of the scale).
    slope : float
        The slope parameter that controls the steepness of the sigmoid curve.
    offset : float
        The vertical offset added to the sigmoid function.
    Returns
    -------
    np.array
        Array of computed sigmoid values for x array.
    '''
    sigmoid = scale / (1 + np.exp(-slope * (x - inflection))) + offset
    return sigmoid


def compute_drift_v(dtdata   : np.array,
                    nbins    : int,
                    dtrange  : Tuple[float, float],
                    seed     : Tuple[float, float, float, float]):
    '''
    Computes the drift velocity for a given distribution
    using the sigmoid function to get the cathode edge.
    Parameters
    ----------
    dtdata: array_like
        Values of DT coordinate.
    nbins: int (optional)
        The number of bins in the z coordinate for the binned fit
    dtrange: length-2 tuple (optional)
        Fix the range in DT.
    seed: length-4 tuple (optional)
        Seed for the fit.
    detector: string (optional)
        Used to get the cathode position from DB.
    Returns
    -------
    dv: float
        Drift velocity.
    dvu: float
        Drift velocity uncertainty.
    '''
    y, x = np.histogram(dtdata, nbins, dtrange)
    x    = shift_to_bin_centers(x)
    if seed is None: seed = np.max(y), np.mean(dtrange), 0.5, np.min(y) # CHANGE: dtrange should be established from db
    # At the moment there is not NEXT-100 DB so this won't work for that geometry
    z_cathode = 1187.37
    print(seed)
    try:
        f   = fit.fit(sigmoid, x, y, seed, sigma = poisson_sigma(y), fit_range = dtrange)
        par = f.values
        err = f.errors
        dv  = (z_cathode + 10/2)/par[1]
        dvu = dv/par[1] * err[1]
    except RuntimeError:
        print("WARNING: Sigmoid fit for dv computation fails. NaN value will be set in its place.")
        dv, dvu = np.nan, np.nan
    return dv, dvu
