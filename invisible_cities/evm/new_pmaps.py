from textwrap import dedent

import numpy as np

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


class _SensorResponses:
    def __init__(self, ids, wfs):
        self._check_valid_input(ids, wfs)

        self._ids              = np.array(ids, copy=False, ndmin=1)
        self._all_waveforms    = np.array(wfs, copy=False, ndmin=2)
        self._wfs_dict         = dict(zip(self._ids, self._all_waveforms))
        self._sum_over_sensors = np.sum(self._all_waveforms, axis=0)
        self._sum_over_times   = np.sum(self._all_waveforms, axis=1)

    @property
    def all_waveforms(self):
        return self._all_waveforms

    def waveform(self, sensor_id):
        return self._wfs_dict[sensor_id]

    def time_slice(self, slice_number):
        return self._all_waveforms[:, slice_number]

    @property
    def ids(self):
        return self._ids

    @property
    def sum_over_times(self):
        return self._sum_over_times

    @property
    def sum_over_sensors(self):
        return self._sum_over_sensors

    def where_above_threshold(self, thr):
        return np.where(self.sum_over_sensors > thr)[0]

    @classmethod
    def build_empty_instance(cls):
        return cls([], [])

    def __repr__(self):
        n_sensors = len(self.ids)
        header = dedent(f"""
            ------------------------
            {self.__class__.__name__} instance
            ------------------------
            Number of sensors: {n_sensors}""")

        sensors = [f"""
            | ID: {ID}
            | WF: {wf}
            """ for ID, wf in self._wfs_dict.items()]
        return header + "".join(map(dedent, sensors))

    def _check_valid_input(self, ids, wfs):
        if len(ids) != len(wfs):
            msg  =  "Shapes do not match\n"
            msg += f"ids has length {len(ids)}\n"
            msg += f"wfs has length {len(wfs)}\n"
            raise ValueError(msg)


class PMTResponses (_SensorResponses): pass
class SiPMResponses(_SensorResponses): pass
