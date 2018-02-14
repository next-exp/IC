import os
import numpy  as np
from . mchits_io import load_mchits
from . mchits_io import load_mcparticles
from . mchits_io import load_mchits_nexus
from . mchits_io import load_mcparticles_nexus
from . mchits_io import load_mcsensor_response_nexus

def test_load_all_mchits(mc_all_hits_data):
    efile, number_of_hits, evt_number = mc_all_hits_data
    mchits_dict = load_mchits(efile)

    assert len(mchits_dict[evt_number]) == number_of_hits

def test_load_mchits(mc_particle_and_hits_data):
    efile, _, _, _, _, _, _, _, X, Y, Z, E, t = mc_particle_and_hits_data

    mchits_dict = load_mchits(efile)
    hX = [hit.X for hit in mchits_dict[0]]
    hY = [hit.Y for hit in mchits_dict[0]]
    hZ = [hit.Z for hit in mchits_dict[0]]
    hE = [hit.E for hit in mchits_dict[0]]
    ht = [hit.time for hit in mchits_dict[0]]

    assert np.isclose(X,hX).all()
    assert np.isclose(Y,hY).all()
    assert np.isclose(Z,hZ).all()
    assert np.isclose(E,hE).all()
    assert np.isclose(t,ht).all()

def test_load_mcparticles(mc_particle_and_hits_data):
    efile, name, pdg, vi, vf, p, Ep, nhits, X, Y, Z, E, t = mc_particle_and_hits_data

    mcparticles_dict = load_mcparticles(efile)
    particle = mcparticles_dict[0][0]

    assert particle.name == name
    assert particle.pdg  == pdg
    assert np.isclose(particle.vi,vi).all()
    assert np.isclose(particle.vf,vf).all()
    assert np.isclose(particle.p,p).all()
    assert particle.E    == Ep
    assert len(particle.hits) == nhits

    hX = [hit.X for hit in particle.hits]
    hY = [hit.Y for hit in particle.hits]
    hZ = [hit.Z for hit in particle.hits]
    hE = [hit.E for hit in particle.hits]
    ht = [hit.time for hit in particle.hits]

    assert np.isclose(X,hX).all()
    assert np.isclose(Y,hY).all()
    assert np.isclose(Z,hZ).all()
    assert np.isclose(E,hE).all()
    assert np.isclose(t,ht).all()

def test_load_mchits_nexus(mc_particle_and_hits_nexus_data):
    efile, _, _, _, _, _, _, _, X, Y, Z, E, t = mc_particle_and_hits_nexus_data

    mchits_dict = load_mchits_nexus(efile)
    hX = [hit.X for hit in mchits_dict[0]]
    hY = [hit.Y for hit in mchits_dict[0]]
    hZ = [hit.Z for hit in mchits_dict[0]]
    hE = [hit.E for hit in mchits_dict[0]]
    ht = [hit.time for hit in mchits_dict[0]]

    assert np.isclose(X,hX).all()
    assert np.isclose(Y,hY).all()
    assert np.isclose(Z,hZ).all()
    assert np.isclose(E,hE).all()
    assert np.isclose(t,ht).all()

def test_load_mcparticles_nexus(mc_particle_and_hits_nexus_data):
    efile, name, pdg, vi, vf, p, Ep, nhits, X, Y, Z, E, t = mc_particle_and_hits_nexus_data

    mcparticles_dict = load_mcparticles_nexus(efile)
    particle = mcparticles_dict[0][1]

    assert particle.name == name
    assert particle.pdg  == pdg
    assert np.isclose(particle.vi,vi).all()
    assert np.isclose(particle.vf,vf).all()
    assert np.isclose(particle.p,p).all()
    assert np.isclose(particle.E,Ep)
    assert len(particle.hits) == nhits

    hX = [hit.X for hit in particle.hits]
    hY = [hit.Y for hit in particle.hits]
    hZ = [hit.Z for hit in particle.hits]
    hE = [hit.E for hit in particle.hits]
    ht = [hit.time for hit in particle.hits]

    assert np.isclose(X,hX).all()
    assert np.isclose(Y,hY).all()
    assert np.isclose(Z,hZ).all()
    assert np.isclose(E,hE).all()
    assert np.isclose(t,ht).all()

def test_load_sensors_nexus(mc_sensors_nexus_data):
    efile, pmt0_first, pmt0_last, pmt0_tot_samples, sipm12013 = mc_sensors_nexus_data

    mcsensors_dict = load_mcsensor_response_nexus(efile)

    waveforms = mcsensors_dict[0]
    sns_number = 0
    samples = list(waveforms[sns_number])

    assert samples[0]   == pmt0_first
    assert samples[-1]  == pmt0_last
    assert len(samples) == pmt0_tot_samples

    sns_number = 12013
    samples = list(waveforms[sns_number])

    assert np.allclose(samples, sipm12013)
