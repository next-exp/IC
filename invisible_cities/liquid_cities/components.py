from functools   import wraps
from functools   import partial
from collections import Sequence
from argparse    import Namespace
from glob        import glob
from os.path     import expandvars
from itertools   import count
from enum        import Enum

import tables as tb
import numpy  as np

from .. dataflow               import dataflow      as fl
from .. evm .ic_containers     import SensorData
from .. evm .event_model       import KrEvent
from .. evm .event_model       import Hit
from .. evm .event_model       import Cluster
from .. evm .event_model       import HitCollection
from .. core.system_of_units_c import units
from .. core.exceptions        import XYRecoFail
from .. core.exceptions        import NoInputFiles
from .. core.exceptions        import NoOutputFile
from .. reco                   import         calib_functions as  cf
from .. reco                   import calib_sensors_functions as csf
from .. reco                   import          peak_functions as pkf
from .. reco                   import         pmaps_functions as pmf
from .. reco.xy_algorithms     import corona
from .. filters.s1s2_filter    import S12Selector
from .. filters.s1s2_filter    import pmap_filter
from .. database               import load_db
from .. sierpe                 import blr
from .. io.pmaps_io            import load_pmaps
from .. types.ic_types         import xy
from .. types.ic_types         import NN
from .. types.ic_types         import NNN


def city(city_function):
    @wraps(city_function)
    def proxy(**kwds):
        conf = Namespace(**kwds)

        # TODO: remove these in the config parser itself, before
        # they ever gets here
        del conf.config_file
        # TODO: these will disappear once hierarchical config files
        # are removed
        del conf.print_config_only
        del conf.hide_config
        del conf.no_overrides
        del conf.no_files
        del conf.full_files

        # TODO: we have decided to remove verbosity.
        # Needs to be removed form config parser
        del conf.verbosity

        # TODO Check raw_data_type in parameters for RawCity

        if 'files_in' not in kwds: raise NoInputFiles
        if 'file_out' not in kwds: raise NoOutputFile

        conf.files_in  = sorted(glob(expandvars(conf.files_in)))
        conf.file_out  =             expandvars(conf.file_out)

        conf.event_range  = _event_range(conf)
        # TODO There were deamons! self.daemons = tuple(map(summon_daemon, kwds.get('daemons', [])))

        return city_function(**vars(conf))
    return proxy


def _event_range(conf):
    if not hasattr(conf, 'event_range'):        return None, 1
    er = conf.event_range
    if not isinstance(er, Sequence): er = (er,)
    if len(er) == 1:                            return None, er[0]
    if len(er) == 2:                            return tuple(er)
    if len(er) == 0:
        raise ValueError('event_range needs at least one value')
    if len(er) >  2:
        raise ValueError('event_range accepts at most 2 values but was given {}'
                        .format(er))


def print_every(N):
    counter = count()
    return fl.branch(fl.map  (lambda _: next(counter), args="event_number", out="index"),
                     fl.slice(None, None, N),
                     fl.sink (lambda data: print(f"events processed: {data['index']}, event number: {data['event_number']}")))


def print_every_alternative_implementation(N):
    @fl.coroutine
    def print_every_loop(target):
        with fl.closing(target):
            for i in count():
                data = yield
                if not i % N:
                    print(f"events processed: {i}, event number: {data['event_number']}")
                target.send(data)
    return print_every_loop


# TODO: consider caching database
def deconv_pmt(run_number, n_baseline, selection=None):
    DataPMT    = load_db.DataPMT(run_number = run_number)
    pmt_active = np.nonzero(DataPMT.Active.values)[0].tolist() if selection is None else selection
    coeff_c    = DataPMT.coeff_c  .values.astype(np.double)
    coeff_blr  = DataPMT.coeff_blr.values.astype(np.double)

    def deconv_pmt(RWF):
        return blr.deconv_pmt(RWF,
                              coeff_c,
                              coeff_blr,
                              pmt_active = pmt_active,
                              n_baseline = n_baseline)
    return deconv_pmt


def get_run_number(h5in):
    if "runInfo" in h5in.root.Run: return h5in.root.Run.runInfo[0]['run_number']
    else:                          return h5in.root.Run.RunInfo[0]['run_number']


class WfType(Enum):
    rwf  = 0
    mcrd = 1


def get_pmt_wfs(h5in, wf_type):
    if   wf_type is WfType.rwf : return h5in.root.RD.pmtrwf
    elif wf_type is WfType.mcrd: return h5in.root.   pmtrd
    else                       : raise  TypeError(f"Invalid WfType: {type(wf_type)}")

def get_sipm_wfs(h5in, wf_type):
    if   wf_type is WfType.rwf : return h5in.root.RD.sipmrwf
    elif wf_type is WfType.mcrd: return h5in.root.   sipmrd
    else                       : raise  TypeError(f"Invalid WfType: {type(wf_type)}")


def wf_from_files(paths, wf_type):
    for path in paths:
        with tb.open_file(path, "r") as h5in:
            event_infos = h5in.root.Run.events
            run_number  = get_run_number(h5in)
            pmt_wfs     = get_pmt_wfs (h5in, wf_type)
            sipm_wfs    = get_sipm_wfs(h5in, wf_type)
            mc_tracks  = h5in.root.MC.MCTracks if run_number <= 0 else None
            for pmt, sipm, (event_number, timestamp) in zip(pmt_wfs, sipm_wfs, event_infos[:]):
                yield dict(pmt=pmt, sipm=sipm, mc=mc_tracks,
                           run_number=run_number, event_number=event_number, timestamp=timestamp)
            # NB, the monte_carlo writer is different from the others:
            # it needs to be given the WHOLE TABLE (rather than a
            # single event) at a time.


def pmap_from_files(paths):
    for path in paths:
        pmaps = load_pmaps(path)
        with tb.open_file(path, "r") as h5in:
            run_number  = get_run_number(h5in)
            event_infos = h5in.root.Run.events
            mc_tracks   = h5in.root.MC .MCTracks if run_number <= 0 else None
            for event_number, timestamp in event_infos[:]:
                yield dict(pmap=pmaps[event_number], mc=mc_tracks,
                           run_number=run_number, event_number=event_number, timestamp=timestamp)
            # NB, the monte_carlo writer is different from the others:
            # it needs to be given the WHOLE TABLE (rather than a
            # single event) at a time.


def sensor_data(path, wf_type):
    with tb.open_file(path, "r") as h5in:
        if   wf_type is WfType.rwf :   (pmt_wfs, sipm_wfs) = (h5in.root.RD .pmtrwf,   h5in.root.RD .sipmrwf)
        elif wf_type is WfType.mcrd:   (pmt_wfs, sipm_wfs) = (h5in.root.    pmtrd ,   h5in.root.    sipmrd )
        else                       :   raise TypeError(f"Invalid WfType: {type(wf_type)}")
        _, NPMT ,  PMTWL =  pmt_wfs.shape
        _, NSIPM, SIPMWL = sipm_wfs.shape
        return SensorData(NPMT=NPMT, PMTWL=PMTWL, NSIPM=NSIPM, SIPMWL=SIPMWL)

####### Transformers ########

def calibrate_pmts(run_number, n_MAU, thr_MAU):
    DataPMT    = load_db.DataPMT(run_number = run_number)
    adc_to_pes = np.abs(DataPMT.adc_to_pes.values)
    adc_to_pes = adc_to_pes[adc_to_pes > 0]

    def calibrate_pmts(cwf):# -> CCwfs:
        return csf.calibrate_pmts(cwf,
                                  adc_to_pes = adc_to_pes,
                                  n_MAU      = n_MAU,
                                  thr_MAU    = thr_MAU)
    return calibrate_pmts


def calibrate_sipms(run_number, thr_sipm):
    DataSiPM   = load_db.DataSiPM(run_number)
    adc_to_pes = np.abs(DataSiPM.adc_to_pes.values)

    def calibrate_sipms(rwf):
        return csf.calibrate_sipms(rwf,
                                   adc_to_pes = adc_to_pes,
                                   thr        = thr_sipm,
                                   bls_mode   = csf.BlsMode.mode)

    return calibrate_sipms


def calibrate_with_mean(run_number):
    DataSiPM   = load_db.DataSiPM(run_number)
    adc_to_pes = np.abs(DataSiPM.adc_to_pes.values)
    def calibrate_with_mean(wfs):
        return csf.subtract_baseline_and_calibrate(wfs, adc_to_pes)
    return calibrate_with_mean

def calibrate_with_mau(run_number, n_mau_sipm):
    DataSiPM   = load_db.DataSiPM(run_number)
    adc_to_pes = np.abs(DataSiPM.adc_to_pes.values)
    def calibrate_with_mau(wfs):
        return csf.subtract_baseline_mau_and_calibrate(wfs, adc_to_pes, n_mau_sipm)
    return calibrate_with_mau


def zero_suppress_wfs(thr_csum_s1, thr_csum_s2):
    def ccwfs_to_zs(ccwf_sum, ccwf_sum_mau):
        return (pkf.indices_and_wf_above_threshold(ccwf_sum_mau, thr_csum_s1).indices,
                pkf.indices_and_wf_above_threshold(ccwf_sum    , thr_csum_s2).indices)
    return ccwfs_to_zs

####### Filters ########

def check_nonempty_indices(indices):
    return indices.size


def peak_classifier(**params):
    selector = S12Selector(**params)
    return partial(pmap_filter, selector)


def compute_xy_position(qthr, qlm, lm_radius, new_lm_radius, msipm):
    def compute_xy_position(xys, qs):
        return corona(xys, qs,
                      Qthr           =          qthr,
                      Qlm            =           qlm,
                      lm_radius      =     lm_radius,
                      new_lm_radius  = new_lm_radius,
                      msipm          =         msipm)
    return compute_xy_position


def compute_z_and_dt(t_s2, t_s1, drift_v):
    dt  = t_s2 - np.array(t_s1)
    z   = dt * drift_v
    dt *= units.ns / units.mus
    return z, dt


def build_pointlike_event(run_number, drift_v, reco):
    datasipm = load_db.DataSiPM(run_number)
    sipm_xs  = datasipm.X.values
    sipm_ys  = datasipm.Y.values
    sipm_xys = np.stack((sipm_xs, sipm_ys), axis=1)

    def build_pointlike_event(pmap, selector_output, event_number, timestamp):
        evt = KrEvent(event_number, timestamp * 1e-3)

        evt.nS1 = 0
        for passed, peak in zip(selector_output.s1_peaks, pmap.s1s):
            if not passed: continue

            evt.nS1 += 1
            evt.S1w.append(peak.width)
            evt.S1h.append(peak.height)
            evt.S1e.append(peak.total_energy)
            evt.S1t.append(peak.time_at_max_energy)

        evt.nS2 = 0

        for passed, peak in zip(selector_output.s2_peaks, pmap.s2s):
            if not passed: continue

            evt.nS2 += 1
            evt.S2w.append(peak.width / units.mus)
            evt.S2h.append(peak.height)
            evt.S2e.append(peak.total_energy)
            evt.S2t.append(peak.time_at_max_energy)

            xys = sipm_xys[peak.sipms.ids           ]
            qs  =          peak.sipms.sum_over_times
            try:
                clusters = reco(xys, qs)
            except XYRecoFail:
                c = NNN()
                Z, DT, Zrms = NN, NN, NN
            else:
                c = clusters[0]
                Z, DT = compute_z_and_dt(evt.S2t[-1], evt.S1t, drift_v)
                Zrms  = peak.rms / units.mus

            evt.Nsipm.append(c.nsipm)
            evt.S2q  .append(c.Q)
            evt.X    .append(c.X)
            evt.Y    .append(c.Y)
            evt.Xrms .append(c.Xrms)
            evt.Yrms .append(c.Yrms)
            evt.R    .append(c.R)
            evt.Phi  .append(c.Phi)
            evt.DT   .append(DT)
            evt.Z    .append(Z)
            evt.Zrms .append(Zrms)

        return evt

    return build_pointlike_event


def split_energy(total_e, clusters):
    if len(clusters) == 1:
        return [total_e]
    qs = np.array([c.Q for c in clusters])
    return total_e * qs / np.sum(qs)


def hit_builder(run_number, drift_v, reco, rebin_slices):
    datasipm = load_db.DataSiPM(run_number)
    sipm_xs  = datasipm.X.values
    sipm_ys  = datasipm.Y.values
    sipm_xys = np.stack((sipm_xs, sipm_ys), axis=1)

    baricenter = partial(corona,
                         Qthr           =  0 * units.pes,
                         Qlm            =  0 * units.pes,
                         lm_radius      = -1 * units.mm,
                         new_lm_radius  = -1 * units.mm,
                         msipm          =  1)

    def empty_cluster():
        return Cluster(NN, xy(0,0), xy(0,0), 0)

    def build_hits(pmap, selector_output, event_number, timestamp):
        hitc = HitCollection(event_number, timestamp * 1e-3)

        # in order to compute z one needs to define one S1
        # for time reference. By default the filter will only
        # take events with exactly one s1. Otherwise, the
        # convention is to take the first peak in the S1 object
        # as reference.
        first_s1 = np.where(selector_output.s1_peaks)[0][0]
        s1_t     = pmap.s1s[first_s1].time_at_max_energy

        # here hits are computed for each peak and each slice.
        # In case of an exception, a hit is still created with a NN cluster.
        # (NN cluster is a cluster where the energy is an IC not number NN)
        # this allows to keep track of the energy associated to non reonstructed hits.
        for peak_no, (passed, peak) in enumerate(zip(selector_output.s2_peaks,
                                                     pmap.s2s)):
            if not passed: continue

            peak = pmf.rebin_peak(peak, rebin_slices)

            xys     = sipm_xys[peak.sipms.ids           ]
            qs      =          peak.sipms.sum_over_times
            try              : cluster = baricenter(xys, qs)[0]
            except XYRecoFail: xy_peak = xy(NN, NN)
            else             : xy_peak = xy(cluster.X, cluster.Y)

            for slice_no, t_slice in enumerate(peak.times):
                z_slice = (t_slice - s1_t) * units.ns * drift_v
                e_slice = peak.pmts.sum_over_sensors[slice_no]
                try:
                    xys      = sipm_xys[peak.sipms.ids                 ]
                    qs       =          peak.sipms.time_slice(slice_no)
                    clusters = reco(xys, qs)
                    es       = split_energy(e_slice, clusters)
                    for c, e in zip(clusters, es):
                        hit       = Hit(peak_no, c, z_slice, e, xy_peak)
                        hitc.hits.append(hit)
                except XYRecoFail:
                    hit = Hit(peak_no, empty_cluster(), z_slice, e_slice, xy_peak)
                    hitc.hits.append(hit)

        return hitc
    return build_hits


def waveform_binner(bins):
    def bin_waveforms(wfs):
        return cf.bin_waveforms(wfs, bins)
    return bin_waveforms
