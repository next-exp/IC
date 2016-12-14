"""
Definition of high level objects.

Bridges is meant to be a connecting structure between two different worlds:
the one of the computers (pure numbers) and the one of the humans (abstract
data).

GML November 2016
"""
import numpy as np


class Signal:
    """
    Structure to hold peak types.

    Attributes
    ----------
    UNKNOWN : string
        Unknown signal type.
    S1 : string
        S1 signal type.
    S2 : string
        S2 signal type.
    """
    UNKNOWN = "??"
    S1 = "S1"
    S2 = "S2"


class Peak:
    """
    A Peak is a collection of consecutive 1mus-slices containing both
    anode and cathode information.

    Parameters
    ----------
    times : 1-dim np.ndarray
        Time for each slice
    pmt_ene : 1-dim np.ndarray
        Signal recorded by the summed PMT for each slice
    sipm_enes : 2-dim np.ndarray
        Signal recorded by the SiPMs (axis 1) for each slice (axis 0)
    tothrs : 1-dim np.ndarray
        Number os 25ns-samples with signal for each slice
    peaktype : string, optional
        Peak type: Signal.S1, Signal.S2 or Signal.UNKNOWN
        default is Signal.UNKNOWN
    """

    def __init__(self, times, pmt_ene, sipm_enes,
                 tothrs, peaktype=Signal.UNKNOWN):
        self.times = np.copy(times)
        self.cathode = np.copy(pmt_ene)
        self.anode = np.copy(sipm_enes)
        self.tothrs = np.copy(tothrs)
        self.signal = peaktype

        self.tmin = self.times[0]
        self.tmax = self.times[-1] + 1.
        self.width = self.tmax - self.tmin

        self.peakmax = (self.times[np.argmax(self.cathode)],
                        np.max(self.cathode))
        self.cathode_integral = self.cathode.sum()
        self.anode_integral = np.nansum(self.anode)

    def __len__(self):
        return self.times.size

    def __iter__(self):
        for data in zip(self.times, self.tothrs, self.cathode, self.anode):
            yield data

    def __str__(self):
        def zs(qs):
            return filter(lambda x: x[1] > 0, enumerate(qs))
        header = ("Peak type: {} Cathode sum: {} Anode sum: {}\n"
                  "".format(self.signal,
                            self.cathode_integral, self.anode_integral))
        header = header + "time ToT cathode anode\n"
        body = "\n".join(["{} {} {} {}".format(t, tot, e, zs(q))
                         for t, tot, e, q in self])
        return header + body

    def __repr__(self):
        return str(self)


class PMap:
    """
    A PMap is a collection of peaks found in the same event.

    Parameters
    ----------
    t0 : float, optional
        Starting time of the event. Default is -1.
    peaks : sequence
        List of peaks in the event.
    """
    def __init__(self, t0=-1., peaks=[]):
        self.t0 = t0
        self.peaks = list(peaks)

    def get(self, type_):
        return filter(lambda peak: peak.signal == type_, self.peaks)

    def __str__(self):
        header = "PMAP with {} peaks. Event t0 = {} mus".format(
                 len(self.peaks), self.t0)
        body = "\n".join(["Peak #{}\n{}".format(i, peak)
                         for i, peak in enumerate(self.peaks)])
        return header + "\n\n" + body

    def __iter__(self):
        for peak in self.peaks:
            yield peak

    def __repr__(self):
        return str(self)
