import numpy  as np

from   typing               import List, Tuple, Callable, Optional
from   pandas               import DataFrame

from .. core.fit_functions  import fit, gauss
from .. core.core_functions import in_range, shift_to_bin_centers


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


def gauss_seed(x         : np.array,
               y         : np.array,
               sigma_rel : Optional[int] = 0.05):

    '''
    This function estimates the seed for a gaussian fit.

    Parameters
    ----------
    x: np.array
        Data to fit.
    y: int
        Number of bins for the histogram.
    sigma_rel (Optional): int
        Relative error, default 5%.

    Returns
    -------
    seed: Tuple
        Tuple with the seed estimation.
    '''

    y_max  = np.argmax(y)
    x_max  = x[y_max]
    sigma  = sigma_rel * x_max
    amp    = y_max * (2 * np.pi)**0.5 * sigma * np.diff(x)[0]
    seed   = amp, x_max, sigma

    return seed


def resolution(values : np.array,
               errors : np.array):

    '''
    Computes the resolution (FWHM) from the Gaussian parameters.

    Parameters
    ----------
    values: np.array
        Gaussian parameters: amplitude, center, and sigma.
    errors: np.array
        Uncertainties for the Gaussian parmeters.

    Returns
    -------
    res: float
        Resolution.
    ures: float
        Uncertainty of resolution.
    '''

    amp  ,   mu,   sigma = values
    u_amp, u_mu, u_sigma = errors

    res  = 235.48 * sigma/mu
    ures = res * (u_mu**2/mu**2 + u_sigma**2/sigma**2)**0.5

    return res, ures


def quick_gauss_fit(data : np.array,
                    bins : int):

    '''
    This function histograms input data and then fits it to a Gaussian.

    Parameters
    ----------
    data: np.array
        Data to fit.
    bins: int
        Number of bins for the histogram.

    Returns
    -------
    fit_output: FitFunction
        Object containing the fit results
    '''

    y, x  = np.histogram(data, bins)
    x     = shift_to_bin_centers(x)
    seed  = gauss_seed(x, y)

    fit_result = fit(gauss, x, y, seed)

    return fit_result


def get_number_of_time_bins(nStimeprofile : int,
                            tstart        : int,
                            tfinal        : int)->int:

    '''
    Computes the number of time bins to use for a given time step
    in seconds.

    Parameters
    ----------
    nStimeprofile: int
        Time step in seconds.
    tstart: int
        Initial timestamp for the dataset.
    tfinal: int
        Final timestamp for the dataset.

    Returns
    -------
    ntimebins: int
        Number of time bins.
    '''

    ntimebins = int(np.floor((tfinal - tstart) / nStimeprofile))
    ntimebins = np.max([ntimebins, 1])

    return ntimebins


def get_time_series_df(ntimebins  : int,
                       time_range : Tuple[float, float],
                       dst        : DataFrame)->Tuple[np.array, List[np.array]]:

    '''
    Given a dst this function returns a time series (ts) and a list of masks which are used to divide
    the event in time intervals.

    Parameters
    ----------
        ntimebins : int
            Number of time bins
        time_range : Tuple
            Time range
        dst : pd.DataFrame
            DataFrame

    Returns
    -------
        A Tuple with:
            np.array       : The time series
            List[np.array] : The list of masks to get the events for each time series.
    '''

    modified_right_limit = np.nextafter(time_range[-1], np.inf)
    time_bins            = np.linspace(time_range[0], modified_right_limit, ntimebins+1)
    masks                = np.array([in_range(dst['time'].to_numpy(), time_bins[i], time_bins[i + 1]) for i in range(ntimebins)])

    return shift_to_bin_centers(time_bins), masks
