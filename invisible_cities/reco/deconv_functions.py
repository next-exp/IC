import numpy  as np
import pandas as pd

from typing  import List
from typing  import Tuple
from typing  import Callable

from enum    import auto

from scipy        import interpolate
from scipy.signal import fftconvolve
from scipy.signal import convolve
from scipy.spatial.distance import cdist

from ..core .core_functions import shift_to_bin_centers

from .. types.ic_types      import AutoNameEnumBase


class InterpolationMethod(AutoNameEnumBase):
    nearest = auto()
    linear  = auto()
    cubic   = auto()
    none    = auto()


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
        pass_df = df.query(cut_condition).copy()
        if not len(pass_df): return pass_df

        with np.errstate(divide='ignore'):
            columns  =      pass_df.loc[:, variables]
            columns *= np.divide(df.loc[:, variables].sum().values, columns.sum())
            pass_df.loc[:, variables] = columns

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
        distance  : Distance to check for other sensors. Usually equal to sensor pitch.
        variables : List with variables to be redistributed.

    Returns
    ----------
    pass_df : hits after removing isolated hits
    """
    dist = np.sqrt(distance[0] ** 2 + distance[1] ** 2)

    def drop_isolated_sensors(df : pd.DataFrame) -> pd.DataFrame:
        x       = df.X.values
        y       = df.Y.values
        xy      = np.column_stack((x,y))
        dr2     = cdist(xy, xy) # compute all square distances

        if not np.any(dr2>0):
            return df.iloc[:0] # Empty dataframe

        closest = np.apply_along_axis(lambda d: d[d > 0].min(), 1, dr2) # find closest that it's not itself
        mask_xy = closest <= dist # take those with at least one neighbour
        pass_df = df.loc[mask_xy, :].copy()

        with np.errstate(divide='ignore'):
            columns  = pass_df.loc[:, variables]
            columns *= np.divide(df.loc[:, variables].sum().values, columns.sum())
            pass_df.loc[:, variables] = columns

        return pass_df

    return drop_isolated_sensors


def deconvolution_input(sample_width : List[float],
                        bin_size     : List[float],
                        inter_method : InterpolationMethod = InterpolationMethod.cubic
                       ) -> Callable:
    """
    Prepares the given data for deconvolution. This involves interpolation of
    the data.

    Parameters
    ----------
    data        : Sensor (hits) position points.
    weight      : Sensor charge for each point.

    Initialization parameters:
        sample_width : Sampling size of the sensors.
        bin_size     : Size of the interpolated bins.
        inter_method : Interpolation method.

    Returns
    ----------
    Hs          : Charge input for deconvolution.
    inter_points : Coordinates of the deconvolution input.
    """
    if inter_method not in InterpolationMethod:
        raise ValueError(f'inter_method {inter_method} is not a valid interpolation method.')

    def deconvolution_input(data        : Tuple[np.ndarray, ...],
                            weight      : np.ndarray
                           ) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:

        ranges = [[coord.min() - 1.5 * sw, coord.max() + 1.5 * sw]     for coord,   sw in zip(data      , sample_width)]
        nbin   = [np.ceil(np.diff(rang)/bs).astype('int')[0]           for bs   , rang in zip(bin_size  ,       ranges)]

        if inter_method in (InterpolationMethod.linear, InterpolationMethod.cubic, InterpolationMethod.nearest):
            allbins = [np.linspace(*rang, np.ceil(np.diff(rang)/sw)+1) for rang ,   sw in zip(ranges[:2], sample_width)]
            Hs, edges = np.histogramdd(data, bins=allbins, normed=False, weights=weight)
        elif inter_method is InterpolationMethod.none:
            Hs, edges = np.histogramdd(data, bins=nbin   , normed=False, weights=weight, range=ranges)
        else:
            raise ValueError(f'inter_method {inter_method} is not a valid interpolatin mode.')

        inter_points = np.meshgrid(*(shift_to_bin_centers(edge) for edge in edges), indexing='ij')
        inter_points = tuple      (inter_p.flatten() for inter_p in inter_points)

        if inter_method in (InterpolationMethod.linear, InterpolationMethod.cubic, InterpolationMethod.nearest):
            Hs, inter_points = interpolate_signal(Hs, inter_points, edges, nbin, inter_method)

        return Hs, inter_points

    return deconvolution_input


def interpolate_signal(Hs           : np.ndarray,
                       inter_points : Tuple[np.ndarray, ...],
                       edges        : Tuple[np.ndarray, ...],
                       nbin         : List[int],
                       inter_method : InterpolationMethod = InterpolationMethod.cubic
                       ) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Interpolates an n-dimensional distribution along N points. Interpolation
    has a lower limit equal to 0.

    Parameters
    ----------
    Hs           : Distribution weights to be interpolated.
    inter_points : Distribution coordinates to be interpolated.
    edges        : Edges of the coordinates.
    nbin         : Number of points to be interpolated in each dimension.
    inter_method : Interpolation method.

    Returns
    ----------
    H1         : Interpolated distribution weights.
    new_points : Interpolated coordinates.
    """
    coords = (shift_to_bin_centers(np.linspace(np.min(edge), np.max(edge), n + 1))
              for n, edge in zip(nbin, edges))
    new_points   = np.meshgrid(*coords, indexing='ij')
    new_points   = tuple      (new_p.flatten() for new_p in new_points)

    H1 = interpolate.griddata(inter_points, Hs.flatten(), new_points, method=inter_method.value)
    H1 = np.nan_to_num       (H1.reshape(nbin))
    H1 = np.clip             (H1, 0, None)

    return H1, new_points


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


def deconvolve(n_iterations  : int,
               iteration_tol : float,
               sample_width  : List[float],
               bin_size      : List[float],
               inter_method  : InterpolationMethod = InterpolationMethod.cubic
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
        n_iterations : Number of Lucy-Richardson iterations
        sample_width : Sampling size of the sensors.
        bin_size     : Size of the interpolated bins.
        inter_method : Interpolation method.

    Returns
    ----------
    deconv_image : Deconvolved image.
    inter_pos     : Coordinates of the deconvolved image.
    """
    var_name     = np.array(['xr', 'yr', 'zr'])
    deconv_input = deconvolution_input(sample_width, bin_size, inter_method)

    def deconvolve(data   : Tuple[np.ndarray, ...],
                   weight : np.ndarray,
                   psf    : pd.DataFrame
                  ) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:

        inter_signal, inter_pos = deconv_input(data, weight)
        columns       = var_name[:len(data)]
        psf_deco      = psf.factor.values.reshape(psf.loc[:, columns].nunique().values)
        deconv_image  = np.nan_to_num(richardson_lucy(inter_signal, psf_deco,
                                                      n_iterations, iteration_tol))

        return deconv_image, inter_pos

    return deconvolve

def richardson_lucy(image, psf, iterations=50, iter_thr=0.):
    """Richardson-Lucy deconvolution (modification from scikit-image package).

    The modification adds a value=0 protection, the possibility to stop iterating
    after reaching a given threshold and the generalization to n-dim of the
    PSF mirroring.

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
    fft_time    = np.sum([n*np.log(n) for n in image.shape + psf.shape])

    # see whether the fourier transform convolution method or the direct
    # convolution method is faster (discussed in scikit-image PR #1792)
    time_ratio = 40.032 * fft_time / direct_time

    if time_ratio <= 1 or len(image.shape) > 2:
        convolve_method = fftconvolve
    else:
        convolve_method = convolve

    image      = image.astype(np.float)
    psf        = psf.astype(np.float)
    im_deconv  = 0.5 * np.ones(image.shape)
    s          = slice(None, None, -1)
    psf_mirror = psf[(s,) * psf.ndim] ### Allow for n-dim mirroring.
    eps        = np.finfo(image.dtype).eps ### Protection against 0 value
    ref_image  = image/image.max()

    for i in range(iterations):
        x = convolve_method(im_deconv, psf, 'same')
        np.place(x, x==0, eps) ### Protection against 0 value
        relative_blur = image / x
        im_deconv *= convolve_method(relative_blur, psf_mirror, 'same')

        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = np.sum(np.divide(((im_deconv/im_deconv.max() - ref_image)**2), ref_image))
        if rel_diff < iter_thr: ### Break if a given threshold is reached.
            break

        ref_image = im_deconv/im_deconv.max()

    return im_deconv
