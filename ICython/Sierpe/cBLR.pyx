
import numpy as np
cimport numpy as np
from scipy import signal as SGN

cpdef test():
    print('hello')

cpdef BLR(np.ndarray[np.int16_t, ndim=1] signal_daq, float coef,
           int nm, float thr1, float thr2, float thr3):
    """
    Deconvolution offline of the DAQ signal
    """
    cdef int len_signal_daq
    len_signal_daq = signal_daq.shape[0]

    cdef np.ndarray[np.float64_t, ndim=1] signal_i = signal_daq.astype(float)
    cdef np.ndarray[np.float64_t, ndim=1] B_MAU = (1./nm)*np.ones(nm, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] MAU = np.zeros(len_signal_daq, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] acum = np.zeros(len_signal_daq, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] signal_r = np.zeros(len_signal_daq, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] pulse_ = np.zeros(len_signal_daq, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] wait_ = np.zeros(len_signal_daq, dtype=np.float64)

    cdef float BASELINE,upper,lower

    MAU[0:nm] = SGN.lfilter(B_MAU,1, signal_daq[0:nm])

    acum[nm] =  MAU[nm]
    BASELINE = MAU[nm-1]

#----------

# While MAU inits BLR is switched off, thus signal_r = signal_daq

    signal_r[0:nm] = signal_daq[0:nm]

    # MAU has computed the offset using nm samples
    # now loop until the end of DAQ window

    cdef int k,j
    cdef float trigger_line, pulse_on, wait_over, offset
    cdef float part_sum

    pulse_on=0
    wait_over=0
    offset = 0

    for k in range(nm,len_signal_daq):
        trigger_line = MAU[k-1] + thr1
        pulse_[k] = pulse_on
        wait_[k] = wait_over

        # condition: raw signal raises above trigger line and
        # we are not in the tail
        # (wait_over == 0)

        if signal_daq[k] > trigger_line and wait_over == 0:

            # if the pulse just started pulse_on = 0.
            # In this case compute the offset as value
            #of the MAU before pulse starts (at k-1)

            if pulse_on == 0: # pulse just started
                #offset computed as the value of MAU before pulse starts
                offset = MAU[k-1]
                pulse_on = 1

            #Pulse is on: Freeze the MAU
            MAU[k] = MAU[k-1]
            signal_i[k] = MAU[k-1]  #signal_i follows the MAU

            #update recovered signal, correcting by offset

            acum[k] = acum[k-1] + signal_daq[k] - offset;
            signal_r[k] = signal_daq[k] + coef*acum[k]

            #signal_r[k] = signal_daq[k] + signal_daq[k]*(coef/2) + coef*acum[k-1]
            #acum[k] = acum[k-1] + signal_daq[k] - offset

        else:  #no signal or raw signal has dropped below threshold
        # but raw signal can be negative for a while and still contribute to the
        # reconstructed signal.

            if pulse_on == 1: #reconstructed signal still on
                # switch the pulse off only when recovered signal
                # drops below threshold
                # lide the MAU, still frozen.
                # keep recovering signal

                MAU[k] = MAU[k-1]
                signal_i[k] = MAU[k-1]

                acum[k] = acum[k-1] + signal_daq[k] - offset;
                signal_r[k] = signal_daq[k] + coef*acum[k]

                #signal_r[k] = signal_daq[k] + signal_daq[k]*(coef/2) + coef*acum[k-1]
                #acum[k] = acum[k-1] + signal_daq[k] - offset

                #if the recovered signal drops before trigger line
                #rec pulse is over!
                if signal_r[k] < trigger_line + thr2:
                    wait_over = 1  #start tail compensation
                    pulse_on = 0   #recovered pulse is over

            else:  #recovered signal has droped below trigger line
            #need to compensate the tail to avoid drifting due to erros in
            #baseline calculatoin

                if wait_over == 1: #compensating pulse
                    # recovered signal and raw signal
                    #must be equal within a threshold
                    # otherwise keep compensating pluse

                    if signal_daq[k-1] < signal_r[k-1] - thr3:
                        # raw signal still below recovered signal
                        # keep compensating pulse
                        # is the recovered signal near offset?
                        upper = offset + (thr3 + thr2)
                        lower = offset - (thr3 + thr2)

                        if lower < signal_r[k-1] < upper:
                            # we are near offset, activate MAU.

                            signal_i[k] = signal_r[k-1]
                            part_sum = 0.
                            for j in range(k-nm, k):
                                part_sum += signal_i[j]/nm
                            MAU[k] = part_sum

                            #MAU[k] = np.sum(signal_i[k-nm:k])*1./nm

                        else:
                            # rec signal not near offset MAU frozen
                            MAU[k] = MAU[k-1]
                            signal_i[k] = MAU[k-1]

                        # keep adding recovered signal
                        acum[k] = acum[k-1] + signal_daq[k] - offset;
                        signal_r[k] = signal_daq[k] + coef*acum[k]
                        #signal_r[k] = signal_daq[k] + signal_daq[k]*(coef/2) + coef*acum[k-1]
                        #acum[k] = acum[k-1] + signal_daq[k] - offset

                    else:  # raw signal above recovered signal: we are done

                        wait_over = 0
                        acum[k] = MAU[k-1]
                        signal_r[k] = signal_daq[k]
                        signal_i[k] = signal_r[k]
                        part_sum = 0.
                        for j in range(k-nm, k):
                            part_sum += signal_i[j]/nm
                        MAU[k] = part_sum
                        #MAU[k] = np.sum(signal_i[k-nm:k])*1./nm

                else: #signal still not found

                    #update MAU and signals
                    part_sum = 0.
                    for j in range(k-nm, k):
                        part_sum += signal_i[j]/nm
                    MAU[k] = part_sum
                    #MAU[k] = np.sum(signal_i[k-nm:k]*1.)/nm
                    acum[k] = MAU[k-1]
                    signal_r[k] = signal_daq[k]
                    signal_i[k] = signal_r[k]
    #energy = np.dot(pulse_f,(signal_r-BASELINE))

    signal_r = signal_r - BASELINE

    #return  signal_r.astype(int)
    return  signal_r.astype(int), MAU, pulse_, wait_

cpdef deconvolve_signal_acum_simple(np.ndarray[np.int16_t, ndim=1] signal_i,
                             int n_baseline=500,
                             float coef_clean=2.905447E-06,
                             float coef_blr=1.632411E-03,
                             float noise_rms = 0.9,
                             float thr_trigger=5, float thr_acum=800,
                             float coeff_acum = 0.9995):

    """
    The accumulator approach by Master VHB

    """

    cdef float coef = coef_blr
    cdef int nm = n_baseline

    cdef int len_signal_daq = len(signal_i)
    cdef np.ndarray[np.float64_t, ndim=1] signal_r = np.zeros(len_signal_daq,
                                                              dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] acum = np.zeros(len_signal_daq,
                                                          dtype=np.double)

    cdef np.ndarray[np.float64_t, ndim=1] signal_daq = signal_i.astype(float)
    cdef int j
    cdef float baseline = 0.
    for j in range(0,nm):
        baseline += signal_daq[j]
    baseline /= nm
    cdef float trigger_line
    trigger_line = thr_trigger*noise_rms

    signal_daq =  baseline - signal_daq

    b_cf, a_cf = SGN.butter(1, coef_clean, 'high', analog=False);
    signal_daq = SGN.lfilter(b_cf,a_cf,signal_daq)

    # BLR
    signal_r[0:nm] = signal_daq[0:nm]
    cdef int k
    for k in range(nm,len_signal_daq):
        # condition: raw signal raises above trigger line
        if (signal_daq[k] > trigger_line) or (acum[k-1] > thr_acum):

            signal_r[k] = signal_daq[k] + signal_daq[k]*(coef/2.0) + coef*acum[k-1]
            acum[k] = acum[k-1] + signal_daq[k]

        else:
            signal_r[k] = signal_daq[k]
            # deplete the accumulator before or after the signal to avoid runoffs
            if (acum[k-1]>0):
                acum[k]=acum[k-1]*coeff_acum

    return signal_r.astype(int), acum


cpdef deconvolve_signal_acum_v1(np.ndarray[np.int16_t, ndim=1] signal_i,
                            int n_baseline=500,
                            float coef_clean=2.905447E-06,
                            float coef_blr=1.632411E-03,
                            float thr_trigger=5,
                            float thr_acum=1000,
                            int acum_discharge_length = 5000,
                            float acum_tau=2500,
                            float acum_compress=0.0025):

    """
    The accumulator approach by Master VHB
    decorated and cythonized  by JJGC
    """

    cdef float coef = coef_blr
    cdef int nm = n_baseline
    cdef int len_signal_daq = len(signal_i)

    cdef np.ndarray[np.float64_t, ndim=1] signal_r = np.zeros(len_signal_daq,
                                                              dtype=np.double)
    cdef np.ndarray[np.float64_t, ndim=1] acum = np.zeros(len_signal_daq,
                                                          dtype=np.double)
    # signal_daq in floats
    cdef np.ndarray[np.float64_t, ndim=1] signal_daq = signal_i.astype(float)

    # compute baseline and noise

    cdef int j
    cdef float baseline = 0.
    for j in range(0,nm):
        baseline += signal_daq[j]
    baseline /= nm

    # reverse sign of signal and subtract baseline
    signal_daq =  baseline - signal_daq

    # compute noise
    cdef float noise =  0.
    for j in range(0,nm):
        noise += signal_daq[j]*signal_daq[j]
    noise /= nm
    cdef float noise_rms = np.sqrt(noise)

    # trigger line

    cdef float trigger_line
    trigger_line = thr_trigger*noise_rms

    # cleaning signal
    cdef np.ndarray[np.float64_t, ndim=1] b_cf
    cdef np.ndarray[np.float64_t, ndim=1] a_cf

    b_cf, a_cf = SGN.butter(1, coef_clean, 'high', analog=False);
    signal_daq = SGN.lfilter(b_cf,a_cf,signal_daq)

    # compute discharge curve
    cdef np.ndarray[np.float64_t, ndim=1] t_discharge
    cdef np.ndarray[np.float64_t, ndim=1] exp
    cdef np.ndarray[np.float64_t, ndim=1] cf
    cdef np.ndarray[np.float64_t, ndim=1] discharge_curve

    cdef float d_length = float(acum_discharge_length)
    t_discharge = np.arange(0, d_length, 1, dtype=np.double)
    exp =  np.exp(-(t_discharge - d_length/2.)/acum_tau)
    cf = 1./(1. + exp)
    discharge_curve = acum_compress*(1. - cf) + (1. - acum_compress)

    # signal_r equals to signal_d (baseline suppressed and change signed)
    # for the first nm samples
    signal_r[0:nm] = signal_daq[0:nm]

    # print ("baseline = {}, noise (LSB_rms) = {} ".format(
    #        baseline, noise_rms,))

    cdef int k
    j=0
    for k in range(nm,len_signal_daq):

        # update recovered signal
        signal_r[k] = signal_daq[k] + signal_daq[k]*(coef/2.0) +\
                      coef_blr * acum[k-1]

        # condition: raw signal raises above trigger line
        # once signal raises above trigger line condition is on until
        # accumulator drops below thr_acum
        if (signal_daq[k] > trigger_line) or (acum[k-1] > thr_acum):

            # update accumulator (signal_daq is already baseline subtracted)
            acum[k] = acum[k-1] + signal_daq[k]
        else:
            j = 0
            # discharge acumulator
            if acum[k-1]>1:
                acum[k] = acum[k-1] * discharge_curve[j]
                if j < acum_discharge_length - 1:
                    j = j + 1
                else:
                    j = acum_discharge_length - 1
            else:
                acum[k]=0
                j=0

    return signal_r.astype(int), acum.astype(int)


cpdef deconvolve_signal_acum_v2(np.ndarray[np.int16_t, ndim=1] signal_i,
                            int n_baseline=500,
                            float coef_clean=2.905447E-06,
                            float coef_blr=1.632411E-03,
                            float thr_trigger=5,
                            int acum_discharge_length = 5000,
                            float acum_tau=2500,
                            float acum_compress=0.01):

    """
    The accumulator approach by Master VHB
    decorated and cythonized  by JJGC
    22.11 Compute baseline using all the waveforms
          computes accumulator threshold from thr_trigger
    """

    cdef float coef = coef_blr
    cdef int nm = n_baseline
    cdef float thr_acum = thr_trigger/coef
    cdef int len_signal_daq = len(signal_i)

    cdef np.ndarray[np.float64_t, ndim=1] signal_r = np.zeros(len_signal_daq,
                                                              dtype=np.double)
    cdef np.ndarray[np.float64_t, ndim=1] acum = np.zeros(len_signal_daq,
                                                          dtype=np.double)
    # signal_daq in floats
    cdef np.ndarray[np.float64_t, ndim=1] signal_daq = signal_i.astype(float)

    # compute baseline and noise

    cdef int j
    cdef float baseline = 0.
    cdef float baseline_end = 0.

    for j in range(0,len_signal_daq):
        baseline += signal_daq[j]
    baseline /= len_signal_daq

    for j in range(len_signal_daq-nm, len_signal_daq):
        baseline_end += signal_daq[j]
    baseline_end /= nm

    # reverse sign of signal and subtract baseline
    signal_daq =  baseline - signal_daq

    # compute noise
    cdef float noise =  0.
    for j in range(0,nm):
        noise += signal_daq[j]*signal_daq[j]
    noise /= nm
    cdef float noise_rms = np.sqrt(noise)

    # trigger line

    cdef float trigger_line
    trigger_line = thr_trigger*noise_rms

    # cleaning signal
    cdef np.ndarray[np.float64_t, ndim=1] b_cf
    cdef np.ndarray[np.float64_t, ndim=1] a_cf

    b_cf, a_cf = SGN.butter(1, coef_clean, 'high', analog=False);
    signal_daq = SGN.lfilter(b_cf,a_cf,signal_daq)

    # compute discharge curve
    cdef np.ndarray[np.float64_t, ndim=1] t_discharge
    cdef np.ndarray[np.float64_t, ndim=1] exp
    cdef np.ndarray[np.float64_t, ndim=1] cf
    cdef np.ndarray[np.float64_t, ndim=1] discharge_curve

    cdef float d_length = float(acum_discharge_length)
    t_discharge = np.arange(0, d_length, 1, dtype=np.double)
    exp =  np.exp(-(t_discharge - d_length/2.)/acum_tau)
    cf = 1./(1. + exp)
    discharge_curve = acum_compress*(1. - cf) + (1. - acum_compress)

    # signal_r equals to signal_d (baseline suppressed and change signed)
    # for the first nm samples
    signal_r[0:nm] = signal_daq[0:nm]

    # print ("baseline = {}, noise (LSB_rms) = {} ".format(
    #        baseline, noise_rms,))

    cdef int k
    j=0
    for k in range(nm,len_signal_daq):

        # update recovered signal
        signal_r[k] = signal_daq[k] + signal_daq[k]*(coef/2.0) +\
                      coef_blr * acum[k-1]

        # condition: raw signal raises above trigger line
        # once signal raises above trigger line condition is on until
        # accumulator drops below thr_acum
        if (signal_daq[k] > trigger_line) or (acum[k-1] > thr_acum):

            # update accumulator (signal_daq is already baseline subtracted)
            acum[k] = acum[k-1] + signal_daq[k]
        else:
            j = 0
            # discharge acumulator
            if acum[k-1]>1:
                acum[k] = acum[k-1] * discharge_curve[j]
                if j < acum_discharge_length - 1:
                    j = j + 1
                else:
                    j = acum_discharge_length - 1
            else:
                acum[k]=0
                j=0
    # return signal and friends
    return signal_r.astype(int), acum.astype(int), baseline, baseline_end, noise_rms

cpdef deconvolve_signal_acum(np.ndarray[np.int16_t, ndim=1] signal_i,
                            int n_baseline=28000,
                            float coef_clean=2.905447E-06,
                            float coef_blr=1.632411E-03,
                            float thr_trigger=5,
                            int acum_discharge_length = 5000):

    """
    The accumulator approach by Master VHB
    decorated and cythonized  by JJGC
    22.11 Compute baseline using all the waveforms
          computes accumulator threshold from thr_trigger
          simplify discharge wrt previous versions:
    In this verison the recovered signal and the accumulator are
    always being charged. At the same time, the accumulator is being
    discharged when there is no signal. This avoids runoffs
    The baseline is computed using a window of 700 mus (by default)
    which should be good for Na and Kr
    """

    cdef float coef = coef_blr
    cdef int nm = n_baseline
    cdef float thr_acum = thr_trigger/coef
    cdef int len_signal_daq = len(signal_i)

    cdef np.ndarray[np.float64_t, ndim=1] signal_r = np.zeros(len_signal_daq,
                                                              dtype=np.double)
    cdef np.ndarray[np.float64_t, ndim=1] acum = np.zeros(len_signal_daq,
                                                          dtype=np.double)
    # signal_daq in floats
    cdef np.ndarray[np.float64_t, ndim=1] signal_daq = signal_i.astype(float)

    # compute baseline and noise

    cdef int j
    cdef float baseline = 0.
    cdef float baseline_end = 0.

    for j in range(0,nm):
        baseline += signal_daq[j]
    baseline /= nm

    cdef nf = len_signal_daq - nm

    for j in range(nm, len_signal_daq):
        baseline_end += signal_daq[j]
    baseline_end /= nf

    # reverse sign of signal and subtract baseline
    signal_daq =  baseline - signal_daq

    # compute noise
    cdef float noise =  0.
    cdef nn = 400 # fixed at 100 mud
    for j in range(0,nn):
        noise += signal_daq[j]*signal_daq[j]
    noise /= nn
    cdef float noise_rms = np.sqrt(noise)

    # trigger line

    cdef float trigger_line
    trigger_line = thr_trigger*noise_rms

    # cleaning signal
    cdef np.ndarray[np.float64_t, ndim=1] b_cf
    cdef np.ndarray[np.float64_t, ndim=1] a_cf

    b_cf, a_cf = SGN.butter(1, coef_clean, 'high', analog=False);
    signal_daq = SGN.lfilter(b_cf,a_cf,signal_daq)

    cdef int k
    j=0
    signal_r[0] = signal_daq[0]
    for k in range(1,len_signal_daq):

        # always update signal and accumulator
        signal_r[k] = signal_daq[k] + signal_daq[k]*(coef/2.0) +\
                      coef * acum[k-1]

        acum[k] = acum[k-1] + signal_daq[k]

        if (signal_daq[k] < trigger_line) and (acum[k-1] < thr_acum):
            # discharge accumulator

            if acum[k-1]>1:
                acum[k] = acum[k-1] * (1. - coef)
                if j < acum_discharge_length - 1:
                    j = j + 1
                else:
                    j = acum_discharge_length - 1
            else:
                acum[k]=0
                j=0
    # return signal and friends
    return signal_r.astype(int), acum.astype(int), baseline, baseline_end, noise_rms
