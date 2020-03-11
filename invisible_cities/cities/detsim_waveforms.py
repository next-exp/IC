import numpy as np

##################################
######### WAVEFORMS ##############
##################################
def create_waveform(times : np.array,
                    pes   : np.array,
                    bins  : np.array,
                    nsamples    : int) -> np.array:
    wf = np.zeros(len(bins))

    if np.sum(pes)==0:
        return wf
    if nsamples==0:
        t = np.repeat(times, pes)
        sel = (bins[0]<=t) & (t<=bins[-1])
        t = t[sel]
        indexes = np.digitize(t, bins)-1
        indexes, counts = np.unique(indexes, return_counts=True)
        wf[indexes] = counts
        return wf

    t = np.repeat(times, pes)
    sel = (bins[0]<=t) & (t<=bins[-1])
    t = np.clip  (t[sel], bins[0], bins[-nsamples])

    indexes = np.digitize(t, bins)-1
    indexes, counts = np.unique(indexes, return_counts=True)

    spread_counts = np.repeat(counts[:, np.newaxis]/nsamples, nsamples, axis=1)
    for index, counts in zip(indexes, spread_pes):
        wf[index:index+nsamples] += counts
    return wf


def create_sensor_waveforms(times   : np.array,
                            pes_at_sensors : np.array,
                            wf_buffer_time : float,
                            bin_width    : float,
                            nsamples : int,
                            poisson  : bool =False) -> np.array:
    bins = np.arange(0, wf_buffer_time, bin_width)
    wfs = np.array([create_waveform(times, pes, bins, nsamples) for pes in pes_at_sensors])

    if poisson:
        wfs = np.random.poisson(wfs)

    return wfs
