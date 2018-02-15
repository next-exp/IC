from textwrap import dedent

import numpy as np

from .. core.system_of_units_c import units
from .. core.core_functions    import weighted_mean_and_std


class PMap:
    def __init__(self, s1s, s2s):
        self.s1s = tuple(s1s)
        self.s2s = tuple(s2s)

    def __repr__(self):
        s  = f"""
        ---------------------
        PMap instance
        ---------------------
        Number of S1s: {len(self.s1s)}
        Number of S2s: {len(self.s2s)}
        """
        return dedent(s)


class _Peak:
    def __init__(self, times, pmts, sipms):
        self._check_valid_input(times, pmts, sipms)

        self.times = np.asarray(times)
        self.pmts  = pmts
        self.sipms = sipms

        i_max                   = np.argmax(self.pmts.sum_over_sensors)
        self.time_at_max_energy = self.times[i_max]
        self.height             = np.max(self.pmts.sum_over_sensors)
        self.total_energy       = self.energy_above_threshold(0)
        self.total_charge       = self.charge_above_threshold(0)
        self.width              = self. width_above_threshold(0)
        self.rms                = self.   rms_above_threshold(0)

    def energy_above_threshold(self, thr):
        i_above_thr  = self.pmts.where_above_threshold(thr)
        wf_above_thr = self.pmts.sum_over_sensors[i_above_thr]
        return np.sum(wf_above_thr)

    def charge_above_threshold(self, thr):
        i_above_thr  = self.sipms.where_above_threshold(thr)
        wf_above_thr = self.sipms.sum_over_sensors[i_above_thr]
        return np.sum(wf_above_thr)

    def  width_above_threshold(self, thr):
        i_above_thr = self.pmts.where_above_threshold(thr)
        if np.size(i_above_thr) < 1:
            return 0

        times_above_thr = self.times[i_above_thr]
        return times_above_thr[-1] - times_above_thr[0]

    def    rms_above_threshold(self, thr):
        i_above_thr     = self.pmts.where_above_threshold(thr)
        times_above_thr = self.times[i_above_thr]
        wf_above_thr    = self.pmts.sum_over_sensors[i_above_thr]
        if np.size(i_above_thr) < 2 or np.sum(wf_above_thr) == 0:
            return 0

        return weighted_mean_and_std(times_above_thr, wf_above_thr)[1]

    def __repr__(self):
        s  = f"""
        ---------------------
        {self.__class__.__name__} instance
        ---------------------
        Number of samples: {len(self.times)}
        Times: {self.times / units.mus} µs
        Time @ max energy: {self.time_at_max_energy / units.mus}
        Width: {self.width / units.mus} µs
        Height: {self.height} pes
        Energy: {self.total_energy} pes
        Charge: {self.total_charge} pes
        RMS: {self.rms / units.mus} µs
        """
        return dedent(s)

    def _check_valid_input(self, times, pmts, sipms):
        length_times = len(times)
        length_pmts  = pmts .all_waveforms.shape[1]
        length_sipms = sipms.all_waveforms.shape[1]

        if length_times == 0:
            raise ValueError("Attempt to initialize an empty"
                            f"{self.__class__.__name} instance.")
        if ((length_times != length_pmts ) or
            (length_times != length_sipms != 0)):
            msg  =  "Shapes don't match!\n"
            msg += f"times has length {length_times}\n"
            msg += f"pmts  has length {length_pmts} \n"
            msg += f"sipms has length {length_sipms}\n"
            raise ValueError(msg)

class S1(_Peak): pass
class S2(_Peak): pass


class _SensorResponses:
    def __init__(self, ids, wfs):
        self._check_valid_input(ids, wfs)

        self.ids              = np.array(ids, copy=False, ndmin=1)
        self.all_waveforms    = np.array(wfs, copy=False, ndmin=2)
        self.sum_over_sensors = np.sum(self.all_waveforms, axis=0)
        self.sum_over_times   = np.sum(self.all_waveforms, axis=1)

        self._wfs_dict        = dict(zip(self.ids, self.all_waveforms))

    def waveform(self, sensor_id):
        return self._wfs_dict[sensor_id]

    def time_slice(self, slice_number):
        return self.all_waveforms[:, slice_number]

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
