import numpy as np

##################################
######### WAVEFORMS ##############
##################################
def bincounter(xs : np.array,
               dx : float =1.,
               x0 : float =0.):
    ixs = ((xs - x0) // dx).astype(int)
    return np.unique(ixs, return_counts=True)


def create_waveform(times : np.array,
                    pes   : np.array,
                    bins  : np.array,
                    wf_bin_time : float,
                    nsamples    : int) -> np.array:
    wf = np.zeros(len(bins))

    if np.sum(pes)==0:
        return wf
    if nsamples==0:
        sel = pes>0
        t = np.repeat(times[sel], pes[sel])
        t = np.clip  (t, 0, bins[-1])
        indexes, counts = bincounter(t, wf_bin_time)
        wf[indexes] = counts
        return wf

    sel = pes>0
    t = np.repeat(times[sel], pes[sel])
    t = np.clip  (t, 0, bins[-nsamples])
    indexes, counts = bincounter(t, wf_bin_time)

    spread_pes = np.repeat(counts[:, np.newaxis]/nsamples, nsamples, axis=1)
    for index, counts in zip(indexes, spread_pes):
        wf[index:index+nsamples] = wf[index:index+nsamples] + counts        
    return wf


def create_sensor_waveforms(times   : np.array,
                            pes_at_sensors : np.array,
                            wf_buffer_time : float,
                            wf_bin_time    : float,
                            nsamples : int,
                            poisson  : bool =False) -> np.array:
    bins = np.arange(0, wf_buffer_time, wf_bin_time)
    wfs = np.array([create_waveform(times, pes, bins, wf_bin_time, nsamples) for pes in pes_at_sensors])

    if poisson:
        wfs = np.random.poisson(wfs)

    return wfs
