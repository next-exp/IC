import numpy as np
from matplotlib.patches import Circle, Rectangle
from typing import Tuple
from invisible_cities.core.stat_functions import poisson_sigma
import invisible_cities.core.fit_functions as fit
from invisible_cities.core.core_functions import in_range, shift_to_bin_centers
from invisible_cities.types.symbols import SelRegionMethod



def select_lifetime_region(df, x0, y0, shape, shape_size):

    """
    shape_size is either the size of the radius or square side
    """

    df = df.dropna(subset=['X','Y'])

    if shape is SelRegionMethod.square:

        x1, y1 = x0-shape_size, y0-shape_size

        mask_df = in_range(df.X, x1, x1+2*shape_size) & in_range(df.Y, y1, y1+2*shape_size)

        df_region = df[mask_df]
        return df_region

    if shape is SelRegionMethod.circle:

        mask_df = (df.X - x0)**2 + (df.Y - y0)**2 <= shape_size**2
        df_region = df[mask_df]
        return df_region



def compute_drift_v(dtdata   : np.array,
                    dtbins   : np.array,
                    seed     : Tuple[float, float, float, float]):
    '''
    Computes the drift velocity for a given distribution
    using the sigmoid function to get the cathode edge.
    Parameters
    ----------
    dtdata: array_like
        Values of DT coordinate.
    dtbins: array_like
        Binning for drift velocity computation
    seed: length-4 tuple (optional)
        Seed for the fit.
    Returns
    -------
    dv: float
        Drift velocity.
    dvu: float
        Drift velocity uncertainty.
    '''
    y, x = np.histogram(dtdata, dtbins)
    x    = shift_to_bin_centers(x)

    if seed is None: seed = np.max(y), np.mean(dtbins), 0.5, np.min(y)
    z_cathode = 1187.37
    #print(seed)
    try:
        f   = fit.fit(fit.sigmoid, x, y, seed, sigma = poisson_sigma(y))
        par = f.values
        err = f.errors
        dv  = (z_cathode + 10/2)/par[1]
        dvu = dv/par[1] * err[1]
    except RuntimeError:
        print("WARNING: Sigmoid fit for dv computation fails. NaN value will be set in its place.")
        dv, dvu = np.nan, np.nan
    return dv, dvu
