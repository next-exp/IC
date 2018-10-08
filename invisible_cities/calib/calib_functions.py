"""
Contains functions used in calibration.
"""

import numpy  as np
import tables as tb

from .. core.system_of_units_c import units
from .. core.core_functions    import in_range


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


def copy_sensor_table(h5in, h5out):
    # Copy sensor table if exists (needed for Non DB calibrations)

    with tb.open_file(h5in) as dIn:
        if 'Sensors' not in dIn.root:
            return
        group    = h5out.create_group(h5out.root, "Sensors")

        if 'DataPMT' in dIn.root.Sensors:
            datapmt  = dIn.root.Sensors.DataPMT
            datapmt.copy(newparent=group)

        if 'DataSiPM' in dIn.root.Sensors:
            datasipm = dIn.root.Sensors.DataSiPM
            datasipm.copy(newparent=group)


def dark_scaler(dark_spectrum):
    """
    A function to scale dark spectrum with mu value.
    """
    def scaled_spectrum(x, mu):
        return np.exp(-mu) * dark_spectrum
    return scaled_spectrum


def seeds_and_bounds(sensor_type, run_no, n_chann, scaler, bins, spec, ped_vals,
                     ped_errs, func='dfunc', use_db_gain_seeds=True):

    """
    Define the seeds and bounds to be used for calibration fits.

    Parameters
    ----------
    sensor_type   : string
    Input of type of sensor: sipm or pmt.
    run_no        : int
    Run number.
    n_chann       : int
    Channel number (sensor ID).
    scaler        : callable
    Scale function.
    bins          : np.array
    Number of divisions in the x axis.
    spec          : np.array
    Spectra, charge values of the signal.
    ped_vals      : np.array
    Values for the pedestal fit.
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

    norm_seed = spec.sum()

    GSeed  = 0
    GSSeed = 0

    if n_chann > 1000: #SiPMs
        GainSeeds  = DB.DataSiPM(run_no).adc_to_pes.values
        SigSeeds   = DB.DataSiPM(run_no).Sigma.values
        spectra    = spec
        peak_range = np.arange(4, 20)
        min_bin    = 10
        max_bin    = 22
        hpw        = 5
        mins       = 5
        dscale     = spec[bins<5].sum() / fitf.gauss(bins[bins<5], *ped_vals).sum()
        lim_ped    = 10000

    else: #PMTs
        GainSeeds  = DB.DataPMT(run_no).adc_to_pes.values
        SigSeeds   = DB.DataPMT(run_no).Sigma.values
        dscale     = spec[bins<0].sum() / fitf.gauss(bins[bins<0], *ped_vals).sum()
        spectra    = spec - fitf.gauss(bins, *ped_vals) * dscale
        peak_range = np.arange(10, 20)
        min_bin    = 15
        max_bin    = 50
        hpw        = 10
        mins       = 0
        lim_ped    = 10000

    if use_db_gain_seeds:
        GSeed     = GainSeeds[n_chann]
        GSSeed    = SigSeeds[n_chann]

    else:
        pDL  = find_peaks_cwt(spectra, peak_range, min_snr=1, noise_perc=5)
        p1pe = pDL[(bins[pDL]>10) & (bins[pDL]<22)][0]
        if not p1pe:
            p1pe = np.argwhere(bins==(min_bin+max_bin)/2)[0][0]
        else:
            p1pe = p1pe[spec[p1pe].argmax()]

        fgaus = fitf.fit(fitf.gauss, bins[p1pe-hpw:p1pe+hpw], spectra[p1pe-hpw:p1pe+hpw],
                         seed=(spectra[p1pe], bins[p1pe], 7), sigma=np.sqrt(spectra[p1pe-hpw:p1pe+hpw]))

        GSeed = fgaus.values[1] - ped_vals[1]
        if fgaus.values[2] <= ped_vals[2]:
            GSSeed = 0.5
        else:
            GSSeed = np.sqrt(fgaus.values[2]**2 - ped_vals[2]**2)

    errs          = np.sqrt(spectra[bins<mins])
    errs[errs==0] = 1
    fscale        = fitf.fit(scaler, bins[bins<mins], spec[bins<mins], (dscale), sigma=errs)
    mu_seed        = -np.log(fscale.values[0])
    if mu_seed < 0: mu_seed = 0.001

    if 'gau' in func:
        ped_seed     = ped_vals[1]
        ped_min      = ped_seed - lim_ped * ped_errs[1]
        ped_max      = ped_seed + lim_ped * ped_errs[1]
        ped_sig_seed = ped_vals[2]
        ped_sig_min  = max(0.001, ped_sig_seed - lim_ped * ped_errs[2])
        ped_sig_max  = ped_sig_seed + lim_ped * ped_errs[2]

        sd0          = (norm_seed, mu_seed, ped_seed, ped_sig_seed, GSeed, GSSeed)
        bd0          = [(0, 0, ped_min, ped_sig_min, 0, 0.001), (1e10, 10000, ped_max, ped_sig_max, 10000, 10000)]
        return sd0, bd0

    sd0 = (norm_seed, mu_seed, GSeed, GSSeed)
    bd0 = [(0, 0, 0, 0.001), (1e10, 10000, 10000, 10000)]
    return sd0, bd0
