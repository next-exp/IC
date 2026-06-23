"""
Contains functions used in calibration.
"""

import numpy  as np
import tables as tb

from scipy.signal import find_peaks_cwt
from scipy.signal import find_peaks

from .. core                 import  system_of_units as units
from .. core.core_functions  import         in_range
from .. core.stat_functions  import    poisson_sigma
from .. core                 import    fit_functions as fitf
from .. database             import          load_db as DB
from .. types.symbols        import       SensorType
from .. evm.ic_containers    import     SensorParams
from .. evm.ic_containers    import   PedestalParams



def bin_waveforms(waveforms, bins):
    """
    A function to bin waveform data. Bins the current event
    data and adds it to the file level bin array.
    """
    def bin_waveform(wf):
        return np.histogram(wf, bins)[0]
    return np.apply_along_axis(bin_waveform, 1, waveforms)


def spaced_integrals(wfs, limits):
    """
    Function to get integrals in certain regions of buffers.
    Returns an array with the integrals between each point
    in the limits array.

    Parameters
    ----------
    wfs: np.ndarray with shape (n, m)
        Buffer waveforms
    limits: np.ndarray with shape (d,)
        Sequence of integration limits

    Returns
    -------
    integrals: np.ndarray with shape (n, d)
        Array with the sum of the waveform between consecutive
        values in the limits array
    """
    if min(limits) < 0 or max(limits) >= np.shape(wfs)[1]:
        raise ValueError(f"Invalid integral limits: {limits}."
                         f" Must be between 0 and {np.shape(wfs)[1]}")
    return np.add.reduceat(wfs, limits, axis=1)


def integral_limits(sample_width, n_integrals, integral_start, integral_width, period):
    """
    Define the integrals to be used for calibration.

    Parameters
    ----------
    sample_width   : float
        Sample width for sensors under study.
    n_integrals    : int
        Number of integrals per buffer
    integral_start : float
        Start in mus of first integral
    integral_width : float
        Width in mus of integrals
    period         : float
        Period in mus between integrals

    Returns
    -------
    corr: np.ndarray
            Correlated limits for integrals
    anti: np.ndarray
        Anticorrelated limits for integrals
    """
    f_int = int(np.floor(integral_start * units.mus / sample_width)) # Position in samples of start of first integral
    w_int = int(np.ceil (integral_width * units.mus / sample_width)) # Width in samples
    p_int = int(np.ceil (        period * units.mus / sample_width)) # Period of repetition in samples
    e_int = f_int + w_int                                            # End first integral

    corr = np.column_stack((f_int + np.arange(0, n_integrals) * p_int,
                            e_int + np.arange(0, n_integrals) * p_int)).flatten()
    anti = corr - w_int - int(2 * units.mus // sample_width)

    return corr, anti


def filter_limits(limits, buffer_length):
    """
    Check that no part of the defined limits falls outside
    the buffer and removes limits if necessary.

    Parameters
    ----------
    limits        : np.ndarray
        Array of integral limits
    buffer_length : int
        Number of waveform samples

    Returns
    -------
    f_limits : np.ndarray
        Filtered limits
    """
    within_buffer = in_range(limits, 0, buffer_length + 1)
    half          = len(limits) // 2

    #check if odd falses at start or end
    n_false_first_half  = np.count_nonzero(~within_buffer[:half])
    n_false_second_half = np.count_nonzero(~within_buffer[half:])

    if n_false_first_half  % 2: within_buffer[  n_false_first_half     ] = False
    if n_false_second_half % 2: within_buffer[- n_false_second_half - 1] = False

    return limits[within_buffer]


def valid_integral_limits(sample_width, n_integrals, integral_start, integral_width, period, buffer_length):
    corr, anti = integral_limits(sample_width, n_integrals, integral_start, integral_width, period)
    return (filter_limits(corr, buffer_length),
            filter_limits(anti, buffer_length))
    



def integrate_peaks_ercilia(bls,
                             n_sigma=5.0,
                             min_distance=10,
                             min_prominence_sigma=2.0,
                             pre_samples=5,
                             post_samples=10,
                             top_fraction=0.95,
                             max_top_width=3,
                             merge_distance=30,
                             integrate_noise=True,
                             noise_window_gap=0,
                             mad_to_sigma=0.6745):
    """
    Peak finding function for fiber barrel calibration.

    - Basic peak finder with threshold/prominence (scipy.find_peaks)
    - Merges peaks that are close together and integrates them as one
    - Removes merged groups whose true maximum sample has a saturated
      (flat) top
    - Optionally integrates fixed-width "noise" windows (width =
      pre_samples + post_samples) placed in peak-free regions, appended
      to the same charges array as a zero-charge reference population

    Parameters
    ----------
    mad_to_sigma : float
        Conversion factor from median absolute deviation to an
        equivalent Gaussian sigma (0.6745 for a normal distribution).
    """
    noise_window_width = pre_samples + post_samples
    noise_step = noise_window_width + noise_window_gap
    if integrate_noise and noise_window_width > 0 and noise_step <= 0:
        raise ValueError(
            "noise_window_gap is too negative: "
            "pre_samples + post_samples + noise_window_gap must be > 0"
        )

    bls_arr = np.asarray(bls)
    if not np.issubdtype(bls_arr.dtype, np.floating):
        bls_arr = bls_arr.astype(np.float64)

    noises = np.median(np.abs(bls_arr), axis=1) / mad_to_sigma
    thresholds  = n_sigma * noises
    prominences = min_prominence_sigma * noises

    charges = []

    for wf, threshold, prominence in zip(bls_arr, thresholds, prominences):
        n = len(wf)

        # cumulative sum -> O(1) window-sum lookups for both peak and
        # noise integration below (built once per waveform)
        csum = np.empty(n + 1, dtype=np.float64)
        csum[0] = 0.0
        csum[1:] = np.cumsum(wf)

        peaks, _ = find_peaks(
            wf,
            height=threshold,
            distance=min_distance,
            prominence=prominence,
        )

        merged_groups = []
        if len(peaks) > 0:
            current_group = [peaks[0]]
            for p in peaks[1:]:
                if p - current_group[-1] <= merge_distance:
                    current_group.append(p)
                else:
                    merged_groups.append(current_group)
                    current_group = [p]
            merged_groups.append(current_group)

        fiber_charges = []
        occupied_windows = []  # every peak region, incl. ones later rejected as saturated

        for group in merged_groups:
            group_arr = np.asarray(group)
            start = max(0, int(group_arr.min()) - pre_samples)
            end   = min(n, int(group_arr.max()) + post_samples)
            occupied_windows.append((start, end))

            # true highest sample in the group, not the mean position
            # (mean can land in a valley between two merged peaks and
            # understate the saturation reference value)
            peak_idx = group_arr[np.argmax(wf[group_arr])]
            peak_val = wf[peak_idx]
            thr = top_fraction * peak_val

            left = right = peak_idx
            while True:
                grew = False
                if left > 0 and wf[left - 1] > thr:
                    left -= 1
                    grew = True
                if right < n - 1 and wf[right + 1] > thr:
                    right += 1
                    grew = True
                if (right - left + 1) > max_top_width:
                    break
                if not grew:
                    break

            if (right - left + 1) > max_top_width:
                continue

            fiber_charges.append(csum[end] - csum[start])

        # --- Noise window integration, appended into the same charges array ---
        if integrate_noise and noise_window_width > 0:
            occupied_windows.sort()
            free_regions = []
            cursor = 0
            for start, end in occupied_windows:
                if start > cursor:
                    free_regions.append((cursor, start))
                cursor = max(cursor, end)
            if cursor < n:
                free_regions.append((cursor, n))

            for region_start, region_end in free_regions:
                length = region_end - region_start
                if length < noise_window_width:
                    continue
                n_windows = (length - noise_window_width) // noise_step + 1
                starts = region_start + noise_step * np.arange(n_windows)
                fiber_charges.extend(csum[starts + noise_window_width] - csum[starts])

        charges.append(np.asarray(fiber_charges))

    return charges




def copy_sensor_table(h5in_name : str,
                      h5out     : tb.file.File):

    with tb.open_file(h5in_name) as dIn:
        try:
            sensor_info = dIn.root.Sensors
            h5out.copy_node(sensor_info,
                            newparent = h5out.root,
                            recursive = True)
        except tb.exceptions.NoSuchNodeError:
            sensor_info = None



def dark_scaler(dark_spectrum):
    """
    A function to scale dark spectrum with mu value.
    """
    def scaled_spectrum(x, mu):
        return np.exp(-mu) * dark_spectrum
    return scaled_spectrum


def seeds_db(sensor_type, detector, run_no, n_chann):
    """
    Take gain and sigma values of previous runs in the database
    to use them as seeds.
    """
    if sensor_type is SensorType.SIPM:
        gain_seed       = DB.DataSiPM(detector, run_no).adc_to_pes.iloc[n_chann]
        gain_sigma_seed = DB.DataSiPM(detector, run_no).     Sigma.iloc[n_chann]
    elif sensor_type is SensorType.PMT:
        gain_seed       = DB.DataPMT(detector, run_no).adc_to_pes.iloc[n_chann]
        gain_sigma_seed = DB.DataPMT(detector, run_no).     Sigma.iloc[n_chann]
    else:
        raise ValueError("SensorType.SIPM or SensorType.PMT must be given for sensor_type")
    return gain_seed, gain_sigma_seed


def poisson_mu_seed(sensor_type, scaler, bins, spectrum, ped_vals):
    """
    Calculate poisson mu using the scaler function.
    """
    if sensor_type is SensorType.SIPM:
        sel    = (bins>=-5) & (bins<=5)
        gdist  = fitf.gauss(bins[sel], *ped_vals)
        dscale = spectrum[sel].sum() / gdist.sum()
        errs   = poisson_sigma(spectrum[sel], default=1)
        return fitf.fit(scaler,
                        bins    [sel],
                        spectrum[sel],
                        dscale, sigma=errs).values[0]
    elif sensor_type is SensorType.PMT:
        sel   = bins<0
        dscale = spectrum[sel].sum() / fitf.gauss(bins[sel], *ped_vals).sum()
        return fitf.fit(scaler,
                        bins    [sel],
                        spectrum[sel],
                        dscale).values[0]
    else:
        raise ValueError("SensorType.SIPM or SensorType.PMT must be given for sensor_type")


def sensor_values(sensor_type, scaler, bins, spectrum, ped_vals):
    """
    Define different values and ranges of the spectra depending on the sensor type.
    """
    if sensor_type is SensorType.SIPM:
        spectra         = spectrum
        peak_range      = np.arange(4, 20)
        min_bin_peak    = 10
        max_bin_peak    = 22
        half_peak_width = 5
        p1pe_seed       = 3
        lim_ped         = 10000
    elif sensor_type is SensorType.PMT:
        sel             = bins<0
        scale           = spectrum[sel].sum() / fitf.gauss(bins[sel], *ped_vals).sum()
        spectra         = spectrum - fitf.gauss(bins, *ped_vals) * scale
        peak_range      = np.arange(10, 20)
        min_bin_peak    = 15
        max_bin_peak    = 50
        half_peak_width = 10
        p1pe_seed       = 7
        lim_ped         = 10000
    else:
        raise ValueError("SensorType.SIPM or SensorType.PMT must be given for sensor_type")
    return SensorParams(spectra, peak_range, min_bin_peak, max_bin_peak, half_peak_width, p1pe_seed, lim_ped)


def pedestal_values(ped_vals, lim_ped, ped_errs):
    """Define pedestal values for 'gau' functions.
    """
    ped_seed     = ped_vals[1]
    ped_min      = ped_seed - lim_ped * ped_errs[1]
    ped_max      = ped_seed + lim_ped * ped_errs[1]
    ped_sig_seed = ped_vals[2]
    ped_sig_min  = max(0.001, ped_sig_seed - lim_ped * ped_errs[2])
    ped_sig_max  = ped_sig_seed + lim_ped * ped_errs[2]

    return PedestalParams(ped_seed, ped_min, ped_max, ped_sig_seed, ped_sig_min, ped_sig_max)


def compute_seeds_from_spectrum(sens_values, bins, ped_vals):

    spectra = sens_values.spectra
    p_range = sens_values.peak_range
    min_b   = sens_values.min_bin_peak
    max_b   = sens_values.max_bin_peak
    hpw     = sens_values.half_peak_width
    p_seed  = sens_values.p1pe_seed

    peaks_dark_led  = find_peaks_cwt(spectra, p_range, min_snr=1, noise_perc=5)
    p1pe_samples    = peaks_dark_led[(bins[peaks_dark_led]>min_b) & (bins[peaks_dark_led]<max_b)]
    if len(p1pe_samples) == 0:
        try:
            p1pe_centroid = np.argwhere(bins==(min_b+max_b)/2)[0][0]
        except IndexError:
            p1pe_centroid = len(bins)-1
    else:
        p1pe_centroid = p1pe_samples[spectra[p1pe_samples].argmax()]

    fit_seed  = (spectra[p1pe_centroid], bins[p1pe_centroid], p_seed)
    fit_sigma = np.sqrt(spectra[p1pe_centroid - hpw : p1pe_centroid + hpw])
    fgaus = fitf.fit(fitf.gauss,
                     bins   [p1pe_centroid - hpw : p1pe_centroid + hpw],
                     spectra[p1pe_centroid - hpw : p1pe_centroid + hpw],
                     seed   = fit_seed,
                     sigma  = fit_sigma,
                     bounds = [(0, -100, 0), (np.inf, 100, 10000)],
                     maxfev = 2000)
    gain_seed = fgaus.values[1] - ped_vals[1]

    if fgaus.values[2] <= ped_vals[2]: gain_sigma_seed = 0.5
    else                             : gain_sigma_seed = np.sqrt(fgaus.values[2]**2 - ped_vals[2]**2)

    return gain_seed, gain_sigma_seed


def seeds_and_bounds(sensor_type, run_no, n_chann, scaler, bins, spectrum, ped_vals,
                     detector, ped_errs, func='dfunc', use_db_gain_seeds=True):
    """ Define the seeds and bounds to be used for calibration fits.

        Parameters
        ----------
        sensor_type   : SensorType (SensorType.SIPM or SensorType.PMT).
        run_no        : int
        Run number.
        n_chann       : int
        Channel number (sensor ID).
        scaler        : callable
        Scale function.
        bins          : np.array
        Number of divisions in the x axis.
        spectrum      : np.array
        Spectra, charge values of the signal.
        ped_vals      : np.array
        Values for the pedestal fit.
        detector      : string
        Input for the used detector.
        ped_errs      : np.array
        Errors of the values for the pedestal fit.
        func          : callable, optional
        Function used for fitting. Defaults to dfunc.
        use_db_gain_seeds : bool, optional
        If True, seeds are taken from previous runs in database.
        If False, peaks are found with find_peaks_cwt function.

        Returns
        -------
        sd0 : sequence
        Seeds for normalization, mu, gain and sigma.
        bd0 : sequence
        Minimum and maximum limits for the previous variables.
    """

    norm_seed = spectrum.sum()
    sens_values = sensor_values(sensor_type, scaler, bins, spectrum, ped_vals)
    if use_db_gain_seeds:
        gain_seed, gain_sigma_seed = seeds_db(sensor_type, detector, run_no, n_chann)

    else:
        gain_seed, gain_sigma_seed = compute_seeds_from_spectrum(sens_values, bins, ped_vals)

    mu_seed = poisson_mu_seed(sensor_type, scaler, bins, spectrum, ped_vals)
    if mu_seed < 0: mu_seed = 0.001

    ped_seed      = ()
    ped_bound_low = ()
    ped_bound_upp = ()

    if 'gau' in func:
        ped_values    = pedestal_values(ped_vals, sens_values.lim_ped , ped_errs)
        ped_seed      = (ped_values.gain        , ped_values.sigma    )
        ped_bound_low = (ped_values.gain_min    , ped_values.gain_max )
        ped_bound_upp = (ped_values.sigma_min   , ped_values.sigma_max)

    seed   = (norm_seed, mu_seed) + ped_seed      + (gain_seed, gain_sigma_seed)
    bound1 = (0       ,       0)  + ped_bound_low + (0        ,           0.001)
    bound2 = (np.inf  ,   10000)  + ped_bound_upp + (10000    ,           10000)
    return seed, (bound1, bound2)
