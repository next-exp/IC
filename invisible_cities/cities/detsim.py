"""
-----------------------------------------------------------------------
                              Detsim
-----------------------------------------------------------------------
From Detector Simulation. This city reads energy deposits (hits) and simulates
S1 and S2 signals. This is accomplished using light tables to compute the signal
generated in the sensors. The input of this city is nexus output containing hits.
This city outputs:
    - pmtrd : PMT  waveforms
    - sipmrd: SIPM waveforms
    - MC  info
    - Run info
    - Filters
"""

import os
import numpy  as np
import tables as tb

from . components import city
from . components import print_every
from . components import copy_mc_info
from . components import collect
from . components import calculate_and_save_buffers
from . components import MC_hits_from_files
from . components import check_max_time

from .. core.configure import check_annotations
from .. reco     import tbl_functions as tbl
from .. database import load_db       as db
from .. dataflow import dataflow      as fl

from .. io.event_filter_io  import event_filter_writer

from .. detsim.simulate_electrons import generate_ionization_electrons
from .. detsim.simulate_electrons import drift_electrons
from .. detsim.simulate_electrons import diffuse_electrons
from .. detsim.light_tables_c     import LT_SiPM
from .. detsim.light_tables_c     import LT_PMT
from .. detsim.s2_waveforms_c     import create_wfs
from .. detsim.detsim_waveforms   import s1_waveforms_creator


@check_annotations
def hits_selector(active_only: bool=True):
    """
    Filtering function that selects hits. (see :active_only: description)

    Parameters:
        :active_only: bool
            if True, returns hits in ACTIVE
            if False, returns hits in ACTIVE and BUFFER
    Returns:
        :select_hits: Callable
            function that select the hits depending on :active_only: parameter
    """
    def select_hits(x, y, z, energy, time, label):
        sel = (label == "ACTIVE")
        if not active_only:
            sel =  sel | (label == "BUFFER")
        return x[sel], y[sel], z[sel], energy[sel], time[sel], label[sel]
    return select_hits


@check_annotations
def ielectron_simulator(*, wi: float, fano_factor: float, lifetime: float,
                        transverse_diffusion: float, longitudinal_diffusion: float, drift_velocity:float,
                        el_gain: float, conde_policarpo_factor: float):
    """
    Function that simulates electron creation, drift, diffusion and photon generation at the EL

    Parameters: floats
        parameter names are self-descriptive.
    Returns:
        :simulate_ielectrons:
            function that returns the positions emission times and number of photons at the EL
    """
    def simulate_ielectrons(x, y, z, time, energy):
        nelectrons = generate_ionization_electrons(energy, wi, fano_factor)
        nelectrons = drift_electrons(z, nelectrons, lifetime, drift_velocity)
        dx, dy, dz = diffuse_electrons(x, y, z, nelectrons, transverse_diffusion, longitudinal_diffusion)
        dtimes = dz/drift_velocity + np.repeat(time, nelectrons)
        nphotons = np.random.normal(el_gain, np.sqrt(el_gain * conde_policarpo_factor), size=nelectrons.sum())
        nphotons = np.round(nphotons).astype(np.int32)
        return dx, dy, dz, dtimes, nphotons
    return simulate_ielectrons


def buffer_times_and_length_getter(pmt_width, sipm_width, el_gap, el_dv, max_length):
    """
    Auxiliar function that computes the signal absolute starting-time and an estimated buffer_length
    """
    max_sensor_bin = max(pmt_width, sipm_width)
    def get_buffer_times_and_length(time, times_ph):
        start_time = np.floor(min(time) / max_sensor_bin) * max_sensor_bin
        el_traverse_time = el_gap / el_dv
        end_time   = np.ceil((max(times_ph) + el_traverse_time)/max_sensor_bin) * max_sensor_bin
        buffer_length = min(max_length, end_time-start_time)
        return start_time, buffer_length
    return get_buffer_times_and_length


def s2_waveform_creator(sns_bin_width, LT, el_drift_velocity):
    """
    Same function as create_wfs in module detsim.s2_waveforms_c with poissonization.
    See description of the refered function for more details.
    """
    def create_s2_waveform(xs, ys, ts, phs, tmin, buffer_length):
        waveforms = create_wfs(xs, ys, ts, phs, LT, el_drift_velocity, sns_bin_width, buffer_length, tmin)
        return np.random.poisson(waveforms)
    return create_s2_waveform


def bin_edges_getter(pmt_width, sipm_width):
    """
    Auxiliar function that returns the waveform bin edges
    """
    def get_bin_edges(pmt_wfs, sipm_wfs):
        pmt_bins  = np.arange(0, pmt_wfs .shape[1]) * pmt_width
        sipm_bins = np.arange(0, sipm_wfs.shape[1]) * sipm_width
        return pmt_bins, sipm_bins
    return get_bin_edges


@city
def detsim(*, files_in, file_out, event_range, print_mod, compression,
           detector_db, run_number, s1_lighttable, s2_lighttable, sipm_psf,
           buffer_params, physics_params, rate):

    buffer_params_  = buffer_params .copy()
    physics_params_ = physics_params.copy()

    buffer_params_["max_time"] = check_max_time(buffer_params_["max_time"], buffer_params_["length"])

    ws    = physics_params_.pop("ws")
    el_dv = physics_params_.pop("el_drift_velocity")

    # derived parameters
    datapmt  = db.DataPMT (detector_db, run_number)
    datasipm = db.DataSiPM(detector_db, run_number)
    lt_pmt   = LT_PMT (fname=os.path.expandvars(s2_lighttable))
    lt_sipm  = LT_SiPM(fname=os.path.expandvars(sipm_psf), sipm_database=datasipm)
    el_gap   = lt_sipm.el_gap_width

    select_s1_candidate_hits = fl.map(hits_selector(False),
                                item = ('x', 'y', 'z', 'energy', 'time', 'label'))

    select_active_hits = fl.map(hits_selector(True),
                                args = ('x', 'y', 'z', 'energy', 'time', 'label'),
                                out = ('x_a', 'y_a', 'z_a', 'energy_a', 'time_a', 'labels_a'))

    filter_events_no_active_hits = fl.map(lambda x:np.any(x),
                                          args= 'energy_a',
                                          out = 'passed_active')
    events_passed_active_hits = fl.count_filter(bool, args='passed_active')

    simulate_electrons = fl.map(ielectron_simulator(**physics_params_),
                                args = ('x_a', 'y_a', 'z_a', 'time_a', 'energy_a'),
                                out  = ('x_ph', 'y_ph', 'z_ph', 'times_ph', 'nphotons'))

    get_buffer_info = buffer_times_and_length_getter(buffer_params_["pmt_width"],
                                                     buffer_params_["sipm_width"],
                                                     el_gap, el_dv,
                                                     buffer_params_["max_time"])
    get_buffer_times_and_length = fl.map(get_buffer_info,
                                         args = ('time', 'times_ph'),
                                         out = ('tmin', 'buffer_length'))

    create_pmt_s1_waveforms = fl.map(s1_waveforms_creator(s1_lighttable, ws, buffer_params_["pmt_width"]),
                                     args = ('x_a', 'y_a', 'z_a', 'time_a', 'energy_a', 'tmin', 'buffer_length'),
                                     out = 's1_pmt_waveforms')

    create_pmt_s2_waveforms = fl.map(s2_waveform_creator(buffer_params_["pmt_width"], lt_pmt, el_dv),
                                     args = ('x_ph', 'y_ph', 'times_ph', 'nphotons', 'tmin', 'buffer_length'),
                                     out = 's2_pmt_waveforms')

    sum_pmt_waveforms = fl.map(lambda x, y : x+y,
                               args = ('s1_pmt_waveforms', 's2_pmt_waveforms'),
                               out = 'pmt_bin_wfs')

    create_pmt_waveforms = fl.pipe(create_pmt_s1_waveforms, create_pmt_s2_waveforms, sum_pmt_waveforms)

    create_sipm_waveforms = fl.map(s2_waveform_creator(buffer_params_["sipm_width"], lt_sipm, el_dv),
                                   args = ('x_ph', 'y_ph', 'times_ph', 'nphotons', 'tmin', 'buffer_length'),
                                   out = 'sipm_bin_wfs')

    get_bin_edges  = fl.map(bin_edges_getter(buffer_params_["pmt_width"], buffer_params_["sipm_width"]),
                            args = ('pmt_bin_wfs', 'sipm_bin_wfs'),
                            out = ('pmt_bins', 'sipm_bins'))

    event_count_in = fl.spy_count()
    evtnum_collect = collect()

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:
        buffer_calculation = calculate_and_save_buffers( buffer_params_["length"]
                                                       , buffer_params_["max_time"]
                                                       , buffer_params_["pre_trigger"]
                                                       , buffer_params_["pmt_width"]
                                                       , buffer_params_["sipm_width"]
                                                       , buffer_params_["trigger_thr"]
                                                       , h5out
                                                       , run_number
                                                       , len(datapmt)
                                                       , len(datasipm)
                                                       , int(buffer_params_["length"] / buffer_params_["pmt_width"])
                                                       , int(buffer_params_["length"] / buffer_params_["sipm_width"]))

        write_nohits_filter   = fl.sink(event_filter_writer(h5out, "active_hits"), args=("event_number", "passed_active"))
        result = fl.push(source= MC_hits_from_files(files_in, rate),
                         pipe  = fl.pipe( fl.slice(*event_range, close_all=True)
                                        , event_count_in.spy
                                        , print_every(print_mod)
                                        , select_s1_candidate_hits
                                        , select_active_hits
                                        , filter_events_no_active_hits
                                        , fl.branch(write_nohits_filter)
                                        , events_passed_active_hits.filter
                                        , simulate_electrons
                                        , get_buffer_times_and_length
                                        , create_pmt_waveforms
                                        , create_sipm_waveforms
                                        , get_bin_edges
                                        , buffer_calculation
                                        , "event_number"
                                        , evtnum_collect.sink),
                         result = dict(events_in     = event_count_in.future,
                                       evtnum_list   = evtnum_collect.future))

        copy_mc_info(files_in, h5out, result.evtnum_list,
                     detector_db, run_number)

        return result
