class PMap:

    @property
    def s1s(self): pass

    @property
    def s2s(self): pass

    # Optionally:
    @property
    def number_of_s1s(self): pass

    @property
    def number_of_s2s(self): pass


class _Peak:

    @property
    def sipms(self): pass

    @property
    def pmts(self): pass

    @property
    def times(self): pass

    @property
    def time_at_max_energy(self): pass

    @property
    def total_energy(self): pass

    @property
    def height(self): pass

    @property
    def width(self): pass

    @property
    def rms(self): pass

    def energy_above_threshold(self, thr): pass

    def  width_above_threshold(self, thr): pass

    def    rms_above_threshold(self, thr): pass


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
