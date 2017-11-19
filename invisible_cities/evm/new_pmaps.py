class PMap:

    def peaks(self, kind = object):
        return [ peak for peak in self._peaks if isinstance(peak, kind) ]

    @property
    def n_peaks(self): pass

    def get_peak(self): pass


class _Peak:

    @property
    def sipms(self): pass

    @property
    def pmts(self): pass

    @property
    def times(self): pass


class S1(_Peak): pass
class S2(_Peak): pass


class SensorResponses:

    @property
    def all_waveforms(self): pass

    def waveform(self, sensor_id): pass

    def time_slice(self, slice_number): pass

    @property
    def ids(self): pass

    @property
    def sum_over_times(self): pass

    @property
    def sum_over_sensors(self): pass
