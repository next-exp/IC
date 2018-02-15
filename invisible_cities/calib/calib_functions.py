"""
Contains useful functions common to all the calibration
modules
"""


import numpy as np

from .. reco                   import calib_sensors_functions as csf
from .. core                   import system_of_units as units


## A function to bin waveform data
def bin_waveforms(waveforms, bins):
        """
        Bins the current event data and adds it
        to the file level bin array
        """
        bin_waveform = lambda x: np.histogram(x, bins)[0]
        return np.apply_along_axis(bin_waveform, 1, waveforms)


## Function to get integrals in certain regions of buffers
def spaced_integrals(SWF, limits):
        """ Returns an array with the integrals between
        each point in the limits array """
        assert min(limits) >= 0
        assert max(limits) < len(SWF[0])
        return np.add.reduceat(SWF, limits, axis=1)


## Define integral limits
def int_limits(samp_wid, nint, start_int, wid_int, period):
        """
        function to define the integrals to be used
        for calibration.
        input:
        samp_wid  : Sample width for sensors under study.
        nintegral : number of integrals per buffer
        start_int : start in mus of first integral
        wid_int   : width in mus of integrals
        period    : peariod in mus between integrals
        output:
        correlated and anticorrelated limits for integrals
        """
        # Position in samples of start of first integral
        f_int = int(np.floor(start_int * units.mus / samp_wid))
        # Width in samples
        w_int = int(np.ceil(wid_int * units.mus / samp_wid))
        # End first integral
        e_int = f_int + w_int
        # Period of repetition in samples
        p_int = int(np.ceil(period * units.mus / samp_wid))
        ## We define an array of the limits
        corr = np.vstack((np.arange(f_int, f_int+nint*p_int, p_int),
                                  np.arange(e_int, e_int+nint*p_int, p_int)))
        corr = corr.reshape((-1,), order='F')
        ## Anti-correlated
        anti = corr - w_int - int(np.floor(2 * units.mus/samp_wid))

        return corr, anti
        

## Check if limits will be valid for buffer.
def filter_limits(limits, buff_len):
        """
        Checks that no part of the defined limits
        falls outside the buffer and removes limits
        if necessary.
        input:
        limits   : array of integral limits
        buff_len : the length of the buffer for these data
        output:
        filterd limits
        """
        range_cond = (limits>=0) & (limits<=buff_len)
        #check if odd falses at start or end
        n_fal = np.count_nonzero(range_cond[:int(len(range_cond)/2)]==False)
        if n_fal & 0x1:
            range_cond[n_fal] = False
        n_fal = np.count_nonzero(range_cond[int(len(range_cond)/2):]==False)
        if n_fal & 0x1:
            range_cond[-(n_fal+1)] = False
        return limits[range_cond]

#def calibrate_with_mean(self, wfs):
#        f = csf.subtract_baseline_and_calibrate
#        return f(wfs, self.sipm_adc_to_pes)


#def calibrate_with_mau(self, wfs):
#        f = csf.subtract_baseline_mau_and_calibrate
#        return f(wfs, self.sipm_adc_to_pes, self.n_MAU_sipm)
