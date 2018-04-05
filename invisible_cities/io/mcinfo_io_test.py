import os
import numpy  as np
import tables as tb

from .  mcinfo_io import load_mchits
from .  mcinfo_io import load_mcparticles
from .  mcinfo_io import load_mcsensor_response
from .  mcinfo_io import mc_info_writer
from .  mcinfo_io import read_mcinfo_evt

from .. reco.tbl_functions import get_mc_info

from pytest import mark
parametrize = mark.parametrize


@mark.serial
@parametrize('skipped_evt, out_filename',
            ((0, 'test_mcinfo_skip_evt0.h5'),
             (1, 'test_mcinfo_skip_evt1.h5')))
def test_mc_info_writer_non_consecutive_events(output_tmpdir, ICDATADIR, krypton_MCRD_file, skipped_evt, out_filename):
    filein  = krypton_MCRD_file
    fileout = os.path.join(output_tmpdir, out_filename)

    with tb.open_file(filein) as h5in:
        with tb.open_file(fileout, 'w') as h5out:

            mc_writer  = mc_info_writer(h5out)
            events_in  = np.unique(h5in.root.MC.extents[:]['evt_number'])

            mc_info = get_mc_info(h5in)

            #Skip the desired event (there are only 3 in the file)
            events_to_copy = [evt for evt in events_in if evt != skipped_evt]

            for evt in events_to_copy:
                mc_writer(mc_info, evt)

            events_out = np.unique(h5out.root.MC.extents[:]['evt_number'])

            np.testing.assert_array_equal(events_to_copy, events_out)


@mark.serial
@parametrize('file_to_check, evt_to_be_read',
            (('test_mcinfo_skip_evt0.h5', 1),
             ('test_mcinfo_skip_evt1.h5', 0),
             ('test_mcinfo_skip_evt1.h5', 2)))
def test_mc_info_writer_output_non_consecutive_events(output_tmpdir, ICDATADIR, krypton_MCRD_file, file_to_check, evt_to_be_read):
    filein    = krypton_MCRD_file
    filecheck = os.path.join(output_tmpdir, file_to_check)

    with tb.open_file(filein) as h5in:
        with tb.open_file(filecheck) as h5filtered:
            mc_info          = get_mc_info(h5in)
            filtered_mc_info = get_mc_info(h5filtered)
            # test the content of events to be sure that they are written
            # correctly
            hit_rows, particle_rows                   = read_mcinfo_evt(mc_info,
                                                                        evt_to_be_read)
            filtered_hit_rows, filtered_particle_rows = read_mcinfo_evt(filtered_mc_info,
                                                                        evt_to_be_read)

            for hitr, filtered_hitr in zip(hit_rows, filtered_hit_rows):
                assert np.allclose(hitr['hit_position'], filtered_hitr['hit_position'])
                assert np.allclose(hitr['hit_time']    , filtered_hitr['hit_time'])
                assert np.allclose(hitr['hit_energy']  , filtered_hitr['hit_energy'])
                assert             hitr['label']      == filtered_hitr['label']

            for partr, filtered_partr in zip(particle_rows, filtered_particle_rows):
                assert np.allclose(partr['initial_vertex'] , filtered_partr['initial_vertex'])
                assert np.allclose(partr['final_vertex']   , filtered_partr['final_vertex'])
                assert np.allclose(partr['momentum']       , filtered_partr['momentum'])
                assert np.allclose(partr['kin_energy']     , filtered_partr['kin_energy'])
                assert             partr['particle_name'] == filtered_partr['particle_name']


def test_mc_info_writer_reset(output_tmpdir, ICDATADIR, krypton_MCRD_file):
    filein  = os.path.join(ICDATADIR, krypton_MCRD_file)
    fileout = os.path.join(output_tmpdir, "test_mc_info_writer_reset.h5")

    with tb.open_file(filein) as h5in:
        with tb.open_file(fileout, 'w') as h5out:

            mc_writer  = mc_info_writer(h5out)
            events_in  = np.unique(h5in.root.MC.extents[:]['evt_number'])

            assert mc_writer.last_row              == 0
            assert mc_writer.last_written_hit      == 0
            assert mc_writer.last_written_particle == 0
            assert mc_writer.first_extent_row      == True

            mc_writer(get_mc_info(h5in), events_in[0])
            assert mc_writer.last_row              == 1
            assert mc_writer.last_written_hit      == 7
            assert mc_writer.last_written_particle == 1
            assert mc_writer.first_extent_row      == False

            mc_writer.reset()
            assert mc_writer.last_row              == 0
            assert mc_writer.last_written_hit      == 0
            assert mc_writer.last_written_particle == 0
            assert mc_writer.first_extent_row      == True


def test_mc_info_writer_automatic_reset(output_tmpdir, ICDATADIR, krypton_MCRD_file, electron_MCRD_file):
    fileout = os.path.join(output_tmpdir, "test_mc_info_writer_automatic_reset.h5")

    with tb.open_file(fileout, "w") as h5out:
        mc_writer  = mc_info_writer(h5out)

        with tb.open_file(krypton_MCRD_file) as h5in:
            events_in  = np.unique(h5in.root.MC.extents[:]['evt_number'])
            mc_writer(get_mc_info(h5in), events_in[0])

        # This would not be possible without automatic reset
        with tb.open_file(electron_MCRD_file) as h5in:
            events_in  = np.unique(h5in.root.MC.extents[:]['evt_number'])
            mc_writer(get_mc_info(h5in), events_in[0])

        assert h5out.root.MC.extents  [:].size ==  2
        assert h5out.root.MC.hits     [:].size == 12
        assert h5out.root.MC.particles[:].size ==  3


def test_load_mchits_correct_number_of_hits(mc_all_hits_data):
    efile, number_of_hits, evt_number = mc_all_hits_data
    mchits_dict = load_mchits(efile)

    assert len(mchits_dict[evt_number]) == number_of_hits


def test_load_mchits(mc_particle_and_hits_nexus_data):
    efile, _, _, _, _, _, _, X, Y, Z, E, t = mc_particle_and_hits_nexus_data

    mchits_dict = load_mchits(efile)
    hX = [hit.X    for hit in mchits_dict[0]]
    hY = [hit.Y    for hit in mchits_dict[0]]
    hZ = [hit.Z    for hit in mchits_dict[0]]
    hE = [hit.E    for hit in mchits_dict[0]]
    ht = [hit.time for hit in mchits_dict[0]]

    assert np.allclose(X, hX)
    assert np.allclose(Y, hY)
    assert np.allclose(Z, hZ)
    assert np.allclose(E, hE)
    assert np.allclose(t, ht)


def test_load_mcparticles(mc_particle_and_hits_nexus_data):
    efile, name, vi, vf, p, Ep, nhits, X, Y, Z, E, t = mc_particle_and_hits_nexus_data

    mcparticles_dict = load_mcparticles(efile)
    particle = mcparticles_dict[0][1]
    assert particle.name == name
    assert np.allclose(particle.initial_vertex,vi)
    assert np.allclose(particle.final_vertex,vf)
    assert np.allclose(particle.p,p)
    assert np.isclose(particle.E,Ep)
    assert len(particle.hits) == nhits

    hX = [hit.X    for hit in particle.hits]
    hY = [hit.Y    for hit in particle.hits]
    hZ = [hit.Z    for hit in particle.hits]
    hE = [hit.E    for hit in particle.hits]
    ht = [hit.time for hit in particle.hits]

    assert np.allclose(X, hX)
    assert np.allclose(Y, hY)
    assert np.allclose(Z, hZ)
    assert np.allclose(E, hE)
    assert np.allclose(t, ht)


def test_load_sensors_data(mc_sensors_nexus_data):
    efile, pmt0_first, pmt0_last, pmt0_tot_samples, sipm_id, sipm = mc_sensors_nexus_data

    mcsensors_dict = load_mcsensor_response(efile)

    waveforms = mcsensors_dict[0]

    sns_number = 0
    wvf        = waveforms[sns_number]
    bins       = [t / wvf.bin_width for t in wvf.times]
    samples    = list(zip(bins, wvf.charges))

    assert samples[0]   == pmt0_first
    assert samples[-1]  == pmt0_last
    assert len(samples) == pmt0_tot_samples

    wvf     = waveforms[sipm_id]
    bins    = [t / wvf.bin_width for t in wvf.times]
    samples = list(zip(bins, wvf.charges))

    assert np.allclose(samples, sipm)
