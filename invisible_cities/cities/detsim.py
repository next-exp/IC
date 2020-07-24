import numpy  as np
import tables as tb
import pandas as pd

from functools import partial

from invisible_cities.core import system_of_units as units
from invisible_cities.reco import tbl_functions   as tbl

import invisible_cities.database.load_db          as db

from invisible_cities.cities.components import city
from invisible_cities.cities.components import print_every

from invisible_cities.dataflow  import dataflow   as fl

from invisible_cities.io.rwf_io           import rwf_writer
from invisible_cities.io.run_and_event_io import run_and_event_writer

# DETSIM IMPORTS
from invisible_cities.cities.detsim_source             import load_MC
from invisible_cities.cities.detsim_simulate_electrons import generate_ionization_electrons as generate_ionization_electrons_
from invisible_cities.cities.detsim_simulate_electrons import drift_electrons               as drift_electrons_
from invisible_cities.cities.detsim_simulate_electrons import diffuse_electrons             as diffuse_electrons_
from invisible_cities.cities.detsim_simulate_electrons import voxelize                      as voxelize_

from invisible_cities.cities.detsim_simulate_photons   import generate_s1_photons           as generate_S1_photons_
from invisible_cities.cities.detsim_simulate_photons   import generate_s2_photons           as generate_S2_photons_

from invisible_cities.cities.detsim_simulate_signal    import pes_at_pmts
from invisible_cities.cities.detsim_simulate_signal    import pes_at_sipms
from invisible_cities.cities.detsim_simulate_signal    import generate_S1_times_from_pes    as generate_S1_times_from_pes_

from invisible_cities.cities.detsim_waveforms          import create_sensor_waveforms       as create_sensor_waveforms_
from invisible_cities.cities.detsim_waveforms          import add_empty_sipmwfs             as add_empty_sipmwfs_

from invisible_cities.cities.detsim_get_psf            import get_psf
from invisible_cities.cities.detsim_get_psf            import get_ligthtables


def get_derived_parameters(detector_db, run_number,
                           s1_ligthtable, s2_ligthtable, sipm_psf,
                           el_gain, conde_policarpo_factor, EL_dz, drift_velocity_EL,
                           wf_buffer_length, wf_pmt_bin_width, wf_sipm_bin_width):
    ########################
    ######## Globals #######
    ########################
    datapmt  = db.DataPMT (detector_db, run_number)
    datasipm = db.DataSiPM(detector_db, run_number)

    S1_LT = get_ligthtables(s1_ligthtable, "S1")
    S2_LT = get_ligthtables(s2_ligthtable, "S2")
    S2sipm_psf, psf_info = get_psf(sipm_psf)

    EL_GAP  = float(str(psf_info[b"EL_GAP"] , errors="ignore")) * units.mm
    if EL_GAP!=EL_dz:
        raise Exception("EL_dz and EL_GAP from sipmpsf must be equal")
    el_pitch = float(str(psf_info[b"pitch_z"], errors="ignore")) * units.mm
    n_el_partitions = int(EL_GAP/el_pitch)

    el_gain_sigma = np.sqrt(el_gain * conde_policarpo_factor)

    EL_dtime      =  EL_dz / drift_velocity_EL
    s2_pmt_nsamples  = np.max((int(EL_dtime // wf_pmt_bin_width ), 1))
    s2_sipm_nsamples = np.max((int(el_pitch // wf_sipm_bin_width), 1))

    return datapmt, datasipm,\
           S1_LT, S2_LT, S2sipm_psf,\
           el_pitch, n_el_partitions, el_gain_sigma,\
           s2_pmt_nsamples, s2_sipm_nsamples


@city
def detsim(files_in, file_out, event_range, detector_db, run_number, s1_ligthtable, s2_ligthtable, sipm_psf,
           ws, wi, fano_factor, drift_velocity, lifetime, transverse_diffusion, longitudinal_diffusion,
           el_gain, conde_policarpo_factor, EL_dz, drift_velocity_EL, voxel_size, sipm_frame,
           pretrigger, wf_buffer_length, wf_pmt_bin_width, wf_sipm_bin_width,
           print_mod, compression):

    ########################
    ######## Globals #######
    ########################
    datapmt, datasipm,\
    S1_LT, S2_LT, S2sipm_psf,\
    el_pitch, n_el_partitions, el_gain_sigma,\
    s2_pmt_nsamples, s2_sipm_nsamples = get_derived_parameters(detector_db, run_number,
                                                               s1_ligthtable, s2_ligthtable, sipm_psf,
                                                               el_gain, conde_policarpo_factor, EL_dz, drift_velocity_EL,
                                                               wf_buffer_length, wf_pmt_bin_width, wf_sipm_bin_width)
    # xsipms, ysipms = datasipm["X"].values, datasipm["Y"].values

    ##########################################
    ############ SIMULATE ELECTRONS ##########
    ##########################################
    generate_ionization_electrons = partial(generate_ionization_electrons_, wi, fano_factor)
    generate_ionization_electrons = fl.map(generate_ionization_electrons, args = ("energy"), out  = ("electrons"))

    drift_electrons = partial(drift_electrons_, lifetime, drift_velocity)
    drift_electrons = fl.map(drift_electrons, args = ("z", "electrons"), out  = ("electrons"))

    count_electrons = fl.map(lambda x: np.sum(x), args=("electrons"), out=("nes"))

    diffuse_electrons = partial(diffuse_electrons_, transverse_diffusion, longitudinal_diffusion)
    diffuse_electrons = fl.map(diffuse_electrons, args = ("x",  "y",  "z", "electrons"), out  = ("dx", "dy", "dz"))

    add_emmision_times = fl.map(lambda dz, times, electrons: dz + np.repeat(times, electrons)*drift_velocity, args = ("dz", "times", "electrons"), out = ("dz"))

    voxelize_electrons = partial(voxelize_, voxel_size)
    voxelize_electrons = fl.map(voxelize_electrons, args = ("dx", "dy", "dz"), out = ("dx", "dy", "dz", "nes"))

    simulate_electrons = fl.pipe(generate_ionization_electrons, drift_electrons, count_electrons, diffuse_electrons,
                                 add_emmision_times, voxelize_electrons)

    ############################################
    ############ SIMULATE PHOTONS ##############
    ############################################
    # S1#
    generate_S1_photons = partial(generate_S1_photons_, ws)
    generate_S1_photons = fl.map(generate_S1_photons, args = ("energy"), out  = ("S1photons"))

    # S2 #
    generate_S2_photons = partial(generate_S2_photons_, el_gain, el_gain_sigma)
    generate_S2_photons = fl.map(generate_S2_photons, args = ("nes"), out  = ("S2photons"))

    simulate_photons = fl.pipe(generate_S1_photons, generate_S2_photons)

    #############################################
    ############ SIMULATE SENSOR SIGNAL #########
    #############################################
    ## PMTs ##
    compute_S1pes_at_pmts = partial(pes_at_pmts, S1_LT)
    compute_S1pes_at_pmts = fl.map(compute_S1pes_at_pmts, args = ("S1photons", "x", "y", "z"), out  = ("S1pes_at_pmts"))

    compute_S2pes_at_pmts = partial(pes_at_pmts, S2_LT)
    compute_S2pes_at_pmts = fl.map(compute_S2pes_at_pmts, args = ("S2photons", "dx", "dy"), out  = ("S2pes_at_pmts"))

    generate_S1_times_from_pes = fl.map(generate_S1_times_from_pes_, args=("S1pes_at_pmts"), out=("S1times"))
    compute_S2times = lambda zs, times, electrons: zs/drift_velocity
    compute_S2times = fl.map(compute_S2times, args=("dz", "times", "electrons"), out=("S2times"))

    simulate_pmt_signal = fl.pipe(compute_S1pes_at_pmts, compute_S2pes_at_pmts, generate_S1_times_from_pes, compute_S2times)

    ## SIPMs ##
    compute_S2pes_at_sipms = partial(pes_at_sipms, S2sipm_psf, datasipm, sipm_frame)
    compute_S2pes_at_sipms = fl.map(compute_S2pes_at_sipms, args=("S2photons", "dx", "dy"), out=("S2pes_at_sipms", "sipmids"))

    #compute_S2times_EL = lambda S2times: np.concatenate([S2times + i*el_pitch/drift_velocity_EL for i in range(1, n_el_partitions+1)])
    compute_S2times_EL = lambda S2times: np.stack([S2times[:, np.newaxis] + i*el_pitch/(2*drift_velocity_EL) for i in range(1, n_el_partitions+1)], axis=1).flatten()
    compute_S2times_EL = fl.map(compute_S2times_EL, args=("S2times"), out=("S2times_EL"))

    simulate_sipm_signal = fl.pipe(compute_S2pes_at_sipms, compute_S2times_EL)

    simulate_signal = fl.pipe(simulate_pmt_signal, simulate_sipm_signal)

    ################################
    ######### BUFFER TIMES #########
    ################################
    ## PMTs ##
    set_S1buffertimes = lambda S1times: [pretrigger + times for times in S1times]
    set_S1buffertimes = fl.map(set_S1buffertimes, args=("S1times"), out=("S1buffertimes_pmt"))

    set_S2buffertimes = lambda S2times: pretrigger + S2times
    set_S2buffertimes_pmt = fl.map(set_S2buffertimes, args=("S2times"), out=("S2buffertimes_pmt"))

    ## SIPMs ##
    set_S2buffertimes_sipm = fl.map(set_S2buffertimes, args=("S2times_EL"), out=("S2buffertimes_sipm"))

    set_buffer_times = fl.pipe(set_S1buffertimes, set_S2buffertimes_pmt, set_S2buffertimes_sipm)

    ####################################
    ######### CREATE WAVEFORMS #########
    ####################################
    ## PMTs ##
    create_S1pmtwfs = create_sensor_waveforms_("S1", wf_buffer_length, wf_pmt_bin_width)
    create_S1pmtwfs = fl.map(create_S1pmtwfs, args=("S1buffertimes_pmt"), out=("S1pmtwfs"))

    create_S2pmtwfs = create_sensor_waveforms_("S2", wf_buffer_length, wf_pmt_bin_width)
    create_S2pmtwfs = partial(create_S2pmtwfs, s2_pmt_nsamples)
    create_S2pmtwfs = fl.map(create_S2pmtwfs, args=("S2buffertimes_pmt", "S2pes_at_pmts"), out=("S2pmtwfs"))

    add_pmtwfs = fl.map(lambda x, y: x + y, args=("S1pmtwfs", "S2pmtwfs"), out=("pmtwfs"))

    ## SIPMs ##
    create_S2sipmwfs = create_sensor_waveforms_("S2", wf_buffer_length, wf_sipm_bin_width)
    create_S2sipmwfs = partial(create_S2sipmwfs, s2_sipm_nsamples)
    create_S2sipmwfs = fl.map(create_S2sipmwfs, args=("S2buffertimes_sipm", "S2pes_at_sipms"), out=("sipmwfs"))

    add_empty_sipmwfs = partial(add_empty_sipmwfs_, (len(datasipm), int(wf_buffer_length // wf_sipm_bin_width)))
    add_empty_sipmwfs = fl.map(add_empty_sipmwfs, args=("sipmwfs", "sipmids"), out=("sipmwfs"))

    create_waveforms = fl.pipe(create_S1pmtwfs, create_S2pmtwfs, add_pmtwfs, create_S2sipmwfs, add_empty_sipmwfs)

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:

        ######################################
        ############# WRITE WFS ##############
        ######################################
        write_pmtwfs  = rwf_writer(h5out, group_name = None, table_name = "pmtrd" , n_sensors = len(datapmt) , waveform_length = int(wf_buffer_length // wf_pmt_bin_width))
        write_sipmwfs = rwf_writer(h5out, group_name = None, table_name = "sipmrd", n_sensors = len(datasipm), waveform_length = int(wf_buffer_length // wf_sipm_bin_width))
        write_pmtwfs  = fl.sink(write_pmtwfs , args=("pmtwfs"))
        write_sipmwfs = fl.sink(write_sipmwfs, args=("sipmwfs"))

        write_run_event = partial(run_and_event_writer(h5out), run_number, timestamp=0)
        write_run_event = fl.sink(write_run_event, args=("event_number"))

        return fl.push(source=load_MC(files_in),
                       pipe  = fl.pipe(fl.slice(*event_range, close_all=True),
                                       print_every(print_mod),
                                       simulate_electrons,
                                       simulate_photons,
                                       # fl.spy(lambda d: [print(k) for k in d]),
                                       simulate_signal,
                                       set_buffer_times,
                                       create_waveforms,
                                       fl.fork(write_pmtwfs,
                                               write_sipmwfs,
                                               write_run_event)),
                        result = ())
