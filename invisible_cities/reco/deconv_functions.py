import numpy  as np
import pandas as pd

from typing  import List
from typing  import Tuple
from typing  import Optional
from typing  import Callable

from scipy        import interpolate
from scipy.signal import fftconvolve, convolve

from ..core .core_functions import shift_to_bin_centers
from ..core .core_functions import in_range

def cut_and_redistribute_df(cut_condition : str,
                            variables     : List[str]=[]) -> Callable:
    '''
    Apply a cut condition to a dataframe and redistribute the cut out values
    of a given variable.

    Parameters
    ----------
    df      : dataframe to be cut

    Initialization parameters:
        cut_condition : String with the cut condition (example "Q > 10")
        variables     : List with variables to be redistributed.

    Returns
    ----------
    pass_df : dataframe after applying the cut and redistribution.
    '''
    def cut_and_redistribute(df : pd.DataFrame) -> pd.DataFrame:
        pass_df = df.query(cut_condition)
        pass_df._is_copy = False
        for redist_variable in variables:
            pass_df[redist_variable] = pass_df[redist_variable] * (df[redist_variable].sum() / pass_df[redist_variable].sum())
        return pass_df

    return cut_and_redistribute

def drop_isolated_sensors(distance  : List[float]=[10., 10.],
                          variables : List[str  ]=[        ]) -> Callable:
    """
    Drops rogue/isolated hits (SiPMs) from a groupedby dataframe.

    Parameters
    ----------
    df      : GroupBy ('event' and 'npeak') dataframe

    Initialization parameters:
        distance  : Distance to check for other sensros. Usually equal to sensor pitch.
        variables : List with variables to be redistributed.

    Returns
    ----------
    pass_df : hits after removing isolated hits
    """
    dist = np.sqrt(distance[0] ** 2 + distance[1] ** 2)

    def drop_isolated_sensors(df : pd.DataFrame) -> pd.DataFrame:
        x       = df.X.values
        y       = df.Y.values
        mask_xy = [False if np.ma.masked_equal(np.sqrt((x - xi)**2 + (y - yi)**2), 0.0, copy=False).min() > dist
                         else True
                         for xi, yi in zip(x, y)]
        pass_df = df[mask_xy]
        pass_df._is_copy = False
        for redist_variable in variables:
            pass_df[f'{redist_variable}'] = pass_df[redist_variable] * (df[redist_variable].sum() / pass_df[redist_variable].sum())
        return pass_df

    return drop_isolated_sensors


def deconvolutionInput(sampleWidth : List[float],
                       bin_size    : List[float],
                       interMethod : Optional[str]=None
                       ) -> Callable:
    """
    Prepares the given data for deconvolution. This involves interpolation of
    the data.

    Parameters
    ----------
    data        : Sensor (hits) position points.
    weight      : Sensor charge for each point.

    Initialization parameters:
        sampleWidth : Sampling size of the sensors.
        bin_size    : Size of the interpolated bins.
        interMethod : Interpolation method.

    Returns
    ----------
    Hs          : Charge input for deconvolution.
    interPoints : Coordinates of the deconvolution input.
    """

    def deconvolutionInput(data        : Tuple[np.ndarray, ...],
                           weight      : np.ndarray
                           ) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
        ranges  = [[data[i].min() - 1.5 * sw, data[i].max() + 1.5 * sw] for i, sw   in enumerate(sampleWidth)]
        allbins = [int(np.diff(rang)/sampleWidth[i])                    for i, rang in enumerate(     ranges)]
        nbin    = [int(np.diff(rang)/bin_size[i])                       for i, rang in enumerate(     ranges)]

        Hs, edges   = (np.histogramdd(data, bins=allbins, range=ranges, normed=False, weights=weight)
                       if interMethod is not None else
                       np.histogramdd(data, bins=nbin   , range=ranges, normed=False, weights=weight))

        interPoints = np.meshgrid(*(shift_to_bin_centers(edge) for edge in edges), indexing='ij')
        interPoints = tuple      (interP.flatten() for interP in interPoints)

        if interMethod is not None:
            Hs, interPoints    = interpolateSignal(Hs        , interPoints    , edges    , nbin    , interMethod)

        return Hs, interPoints

    return deconvolutionInput


def interpolateSignal(Hs          : np.ndarray,
                      interPoints : Tuple[np.ndarray, ...],
                      edges       : Tuple[np.ndarray, ...],
                      nbin        : List[int],
                      interMethod : Optional[str]=None
                      ) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Interpolates an n-dimensional distribution along N points. Interpolation
    has a lower limit equal to 0.

    Parameters
    ----------
    Hs          : Distribution weights to be interpolated.
    interPoints : Distribution coordinates to be interpolated.
    edges       : Edges of the coordinates.
    nbin        : Number of points to be interpolated in each dimension.
    interMethod : Interpolation method.

    Returns
    ----------
    H1        : Interpolated distribution weights.
    newPoints : Interpolated coordinates.
    """
    newPoints   = np.meshgrid(*(shift_to_bin_centers(np.linspace(np.min(edge), np.max(edge), nbin[i]+1)) for i, edge in enumerate(edges)), indexing='ij')
    newPoints   = tuple      (newP.flatten() for newP in newPoints)

    H1 = interpolate.griddata(interPoints, Hs.flatten(), newPoints, method=interMethod)
    H1 = np.nan_to_num       (H1.reshape(nbin))
    H1 = np.where            (H1<0, 0, H1)

    return H1, newPoints


def find_nearest(array : np.ndarray,
                 value : float
                 ) -> float :
    """
    Find nearest value to a given value in an array. Wrote by @unutbu.

    Parameters
    ----------
    array : Array to be searched.
    value : Value to be found.

    Returns
    ----------
    array[idx] : Input array value closest to the input value.
    """
    array =  np.asarray(array)
    idx   = (np.abs    (array - value)).argmin()
    return array[idx]


def deconvolve(iterationNumber : int,
               iterationThr    : float,
               sampleWidth     : List[float],
               bin_size        : List[float],
               interMethod     : Optional[str]=None
               ) -> Callable:
    """
    Deconvolves a given set of data (sensor position and its response)
    using Lucy-Richardson deconvolution.

    Parameters
    ----------
    data        : Sensor (hits) position points.
    weight      : Sensor charge for each point.
    psf         : Point-spread function.

    Initialization parameters:
        iterationNumber : Number of Lucy-Richardson iterations
        sampleWidth     : Sampling size of the sensors.
        bin_size        : Size of the interpolated bins.
        interMethod     : Interpolation method.

    Returns
    ----------
    deconv_image : Deconvolved image.
    interPos     : Coordinates of the deconvolved image.
    """
    varName = np.array(['xr', 'yr', 'zr'])
    deconvInput = deconvolutionInput(sampleWidth, bin_size, interMethod)

    def deconvolve(data            : Tuple[np.ndarray, ...],
                   weight          : np.ndarray,
                   psf             : pd.DataFrame
                  ) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:

        interSignal, interPos = deconvInput(data, weight)
        psf_deco      = psf.factor.values.reshape(tuple(psf[var].nunique() for var in varName[:len(data)]))
        deconv_image  = np.nan_to_num(richardson_lucy(interSignal, psf_deco,
                                                      iterationNumber, iterationThr))

        return deconv_image.flatten(), interPos

    return deconvolve

def richardson_lucy(image, psf, iterations=50, iter_thr=0.):
    """Richardson-Lucy deconvolution (modification from scikit-image package).
    Parameters
    ----------
    image : ndarray
       Input degraded image (can be N dimensional).
    psf : ndarray
       The point spread function.
    iterations : int, optional
       Number of iterations. This parameter plays the role of
       regularisation.
    iter_thr : float, optional
       Threshold on the relative difference between iterations to stop iterating.

    Returns
    -------
    im_deconv : ndarray
       The deconvolved image.
    Examples
    --------
    >>> from skimage import color, data, restoration
    >>> camera = color.rgb2gray(data.camera())
    >>> from scipy.signal import convolve2d
    >>> psf = np.ones((5, 5)) / 25
    >>> camera = convolve2d(camera, psf, 'same')
    >>> camera += 0.1 * camera.std() * np.random.standard_normal(camera.shape)
    >>> deconvolved = restoration.richardson_lucy(camera, psf, 5)
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    """
    # compute the times for direct convolution and the fft method. The fft is of
    # complexity O(N log(N)) for each dimension and the direct method does
    # straight arithmetic (and is O(n*k) to add n elements k times)
    direct_time = np.prod(image.shape + psf.shape)
    fft_time =  np.sum([n*np.log(n) for n in image.shape + psf.shape])

    # see whether the fourier transform convolution method or the direct
    # convolution method is faster (discussed in scikit-image PR #1792)
    time_ratio = 40.032 * fft_time / direct_time

    if time_ratio <= 1 or len(image.shape) > 2:
        convolve_method = fftconvolve
    else:
        convolve_method = convolve

    image = image.astype(np.float)
    psf = psf.astype(np.float)
    im_deconv = 0.5 * np.ones(image.shape)
    s = slice(None, None, -1)
    psf_mirror = psf[(s,) * psf.ndim]
    eps = np.finfo(image.dtype).eps
    ref_image = image/image.max()

    for i in range(iterations):
        x = convolve_method(im_deconv, psf, 'same')
        np.place(x, x==0, eps)
        relative_blur = image / x
        im_deconv *= convolve_method(relative_blur, psf_mirror, 'same')

        rel_diff = np.sum(np.nan_to_num(((im_deconv/im_deconv.max() - ref_image)**2)/ref_image))
        if rel_diff < iter_thr:
            break

        ref_image = im_deconv/im_deconv.max()

    return im_deconv
