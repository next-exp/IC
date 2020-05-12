import os
import numpy  as np
import pandas as pd
import tables as tb

from glob          import glob
from os.path       import expandvars
from numpy.testing import assert_allclose

from .. database  import load_db
from .  mcinfo_io import load_mchits_df
from .  mcinfo_io import cast_mchits_to_dict
from .  mcinfo_io import load_mcparticles_df
from .  mcinfo_io import get_event_numbers_in_file
from .  mcinfo_io import get_sensor_binning
from .  mcinfo_io import get_sensor_types
from .  mcinfo_io import get_mc_tbl_list
from .  mcinfo_io import load_mcsensor_response_df
from .  mcinfo_io import MCTableType
from .  mcinfo_io import copy_mc_info
from .  mcinfo_io import read_mcinfo_evt
from .  mcinfo_io import read_mc_tables
from .  mcinfo_io import mc_writer
from .  mcinfo_io import _read_mchit_info

from .. core               import system_of_units as units
from .. core.exceptions    import NoParticleInfoInFile
from .. core.testing_utils import assert_dataframes_equal
from .. core.testing_utils import assert_MChit_equality

from pytest import fixture
from pytest import mark
from pytest import raises


def test_copy_mc_info_which_events_is_none(ICDATADIR, config_tmpdir):
    file_in  = os.path.join(ICDATADIR, "Kr83_nexus_v5_03_00_ACTIVE_7bar_10evts.sim.h5")
    file_out = os.path.join(config_tmpdir, "dummy_out.h5")

    with tb.open_file(file_out, 'w') as h5out:
        writer = mc_writer(h5out)
        with tb.open_file(file_in, 'r') as h5in:
            copy_mc_info(file_in, writer,
                         db_file    = 'new',
                         run_number = -6400)
            events_in_h5in  = h5in .root.MC.extents.cols.evt_number[:]
            events_in_h5out = np.unique(h5out.root.MC.hits.cols.event_id[:])
            assert all(events_in_h5in == events_in_h5out)


def test_copy_mc_info_multiple_files(ICDATADIR, config_tmpdir):
    file_in1 = os.path.join(ICDATADIR    , "Kr83m_nexus_HEAD20200327.sim.0.h5")
    file_in2 = os.path.join(ICDATADIR    , "Kr83m_nexus_HEAD20200327.sim.1.h5")
    file_out = os.path.join(config_tmpdir,                      "dummy_out.h5")

    f1_evt = get_event_numbers_in_file(file_in1)
    f2_evt = get_event_numbers_in_file(file_in2)
    with tb.open_file(file_out, 'w') as h5out:
        writer = mc_writer(h5out)
        for fn in [file_in1, file_in2]:
            copy_mc_info(fn, writer, db_file='new', run_number=-6400)
        assert 'event_mapping' in h5out.root.MC
        mapping = h5out.root.MC.event_mapping
        config  = h5out.root.MC.configuration
        assert np.unique(mapping.cols.file_index[:]).shape == (2,)
        assert hasattr(config.cols, 'file_index')
        assert np.all(np.unique(mapping.cols.file_index[:]) ==
                      np.unique(config .cols.file_index[:])   )
        evt_out = np.unique(mapping.cols.event_id[:])
        assert np.all(evt_out == np.concatenate([f1_evt, f2_evt]))


def test_copy_mc_info_which_events_is_subset(ICDATADIR, config_tmpdir):
    file_in  = os.path.join(ICDATADIR, "Kr83_nexus_v5_03_00_ACTIVE_7bar_10evts.sim.h5")
    file_out = os.path.join(config_tmpdir, "dummy_out.h5")
    which_events = [0, 3, 6, 9]

    with tb.open_file(file_out, 'w') as h5out:
        writer = mc_writer(h5out)
        copy_mc_info(file_in, writer, which_events      ,
                     db_file = 'new', run_number = -6400)
        events_in_h5out = np.unique(h5out.root.MC.hits.cols.event_id[:])
        assert events_in_h5out.tolist() == which_events


def test_copy_mc_info_which_events_out_of_range(ICDATADIR, config_tmpdir):
    file_in  = os.path.join(ICDATADIR, "Kr83_nexus_v5_03_00_ACTIVE_7bar_10evts.sim.h5")
    file_out = os.path.join(config_tmpdir, "dummy_out.h5")
    which_events = [10]

    with tb.open_file(file_out, 'w') as h5out:
        writer = mc_writer(h5out)
        copy_mc_info(file_in, writer, which_events      ,
                     db_file = 'new', run_number = -6400)

    ## Check that we can read the output but it's empty
    hits = load_mchits_df(file_out)
    assert hits.shape == (0, 6)


@mark.parametrize("fn_oldformat fn_newformat".split(),
                  (("nexus_scint.oldformat.sim.h5"         ,
                    "nexus_scint.newformat.sim.h5"         ),
                   ("nexus_new_kr83m_fast.oldformat.sim.h5",
                    "nexus_new_kr83m_fast.newformat.sim.h5"),
                   ("nexus_new_kr83m_full.oldformat.sim.h5",
                    "nexus_new_kr83m_full.newformat.sim.h5")))
def test_copy_mc_info_same_results(ICDATADIR   , config_tmpdir,
                                   fn_oldformat,  fn_newformat):
    file_in_old  = os.path.join(ICDATADIR    ,    fn_oldformat)
    file_in_new  = os.path.join(ICDATADIR    ,    fn_newformat)
    file_out_old = os.path.join(config_tmpdir, "tmpOut_old.h5")
    file_out_new = os.path.join(config_tmpdir, "tmpOut_new.h5")

    with tb.open_file(file_out_old, 'w') as h5out:
        writer = mc_writer(h5out)
        copy_mc_info(file_in_old, writer, db_file='new', run_number=-6400)
    with tb.open_file(file_out_new, 'w') as h5out:
        writer = mc_writer(h5out)
        copy_mc_info(file_in_new, writer)

    old_tbls = read_mc_tables(file_out_old)
    new_tbls = read_mc_tables(file_out_new)
    assert len(old_tbls) == len(new_tbls)
    for key in new_tbls.keys():
        if key is not MCTableType.configuration:
            print(key)
            assert_dataframes_equal(new_tbls[key], old_tbls[key],
                                    check_types=False)


def test_read_mc_tables(mc_particle_and_hits_nexus_data_new):
    file_in, *_ = mc_particle_and_hits_nexus_data_new

    input_tbls  = get_mc_tbl_list          (file_in)
    all_evt     = get_event_numbers_in_file(file_in)

    shape_dict  = {}
    with tb.open_file(file_in) as h5in:
        for tbl in h5in.root.MC:
            shape_dict[MCTableType[tbl.name]] = (tbl.shape[0], len(tbl.cols))

    tbl_dict    = read_mc_tables(file_in, all_evt)
    for tbl, shp in shape_dict.items():
        assert tbl in tbl_dict.keys()
        assert shp == tbl_dict[tbl].shape


def test_mc_writer(mc_particle_and_hits_nexus_data_new, config_tmpdir):
    file_in, *_ = mc_particle_and_hits_nexus_data_new
    file_out    = os.path.join(config_tmpdir, 'dummy_out.h5')

    input_tbls  = get_mc_tbl_list          (file_in)
    all_evt     = get_event_numbers_in_file(file_in)

    tbl_dict    = read_mc_tables(file_in, all_evt)
    with tb.open_file(file_out, 'w') as h5out:
        mcwrite = mc_writer(h5out)
        mcwrite(tbl_dict)

    saved_tbls = get_mc_tbl_list(file_out)
    assert np.all(input_tbls == saved_tbls)


def test_mc_writer_oldformat(mc_sensors_nexus_data, config_tmpdir):
    file_in, *_ = mc_sensors_nexus_data
    file_out    = os.path.join(config_tmpdir, 'dummy_out.h5')

    input_tbls  = get_mc_tbl_list          (file_in)
    all_evt     = get_event_numbers_in_file(file_in)

    tbl_dict    = read_mc_tables(file_in, all_evt, 'new', 6400)
    with tb.open_file(file_out, 'w') as h5out:
        mcwrite = mc_writer(h5out)
        mcwrite(tbl_dict)

    saved_tbls = get_mc_tbl_list(file_out)
    ## The extents and events tables are not saved.
    assert len(input_tbls) == len(saved_tbls) + 2
    assert MCTableType.configuration in saved_tbls
    assert MCTableType.hits          in saved_tbls
    assert MCTableType.particles     in saved_tbls
    assert MCTableType.sns_positions in saved_tbls
    assert MCTableType.sns_response  in saved_tbls


@fixture(scope = 'module')
def mc_particle_and_hits_nexus_data_new(ICDATADIR):
    X = [-395.8089294433594 , -395.08221435546875, -394.3164367675781 ,
         -393.5862121582031 , -392.73992919921875, -391.91583251953125,
         -391.0397644042969 , -390.815673828125  , -390.8048400878906 ,
         -390.97186279296875, -391.08892822265625, -391.2257385253906 ,
         -391.6145324707031 , -391.7532958984375 , -391.2930908203125 ,
         -390.40191650390625, -389.55841064453125, -388.8764343261719 ,
         -388.4765319824219 , -388.0899963378906 , -387.8658447265625 ,
         -387.83648681640625, -387.8304748535156 , -388.13482666015625,
         -388.63177490234375, -389.1610412597656 , -389.2224426269531 ,
         -389.2239074707031 , -389.2264099121094 , -389.2304992675781 ,
         -389.2414245605469 , -389.2499084472656 ]
    Y = [-351.8671569824219 , -352.09417724609375, -352.6225891113281 ,
         -353.1514892578125 , -353.60101318359375, -354.1516418457031 ,
         -354.55340576171875, -355.15118408203125, -355.9131774902344 ,
         -356.5508728027344 , -357.07452392578125, -357.6239013671875 ,
         -358.3233947753906 , -359.18603515625   , -359.7363586425781 ,
         -359.7808532714844 , -359.7142333984375 , -360.1387939453125 ,
         -360.65020751953125, -361.1523742675781 , -361.7250671386719 ,
         -362.6084289550781 , -363.6018981933594 , -364.539794921875  ,
         -365.4069519042969 , -366.2047119140625 , -366.2724304199219 ,
         -366.2742004394531 , -366.2762756347656 , -366.2796936035156 ,
         -366.2888488769531 , -366.2962646484375 ]
    Z = [ 318.184326171875  ,  317.53680419921875,  317.2337951660156 ,
          316.8293151855469 ,  316.69366455078125,  316.7612609863281 ,
          316.6136779785156 ,  315.9402160644531 ,  315.2978515625    ,
          314.5556640625    ,  313.7184753417969 ,  312.9024658203125 ,
          312.33935546875   ,  311.9386901855469 ,  311.46380615234375,
          311.0255432128906 ,  310.50152587890625,  309.9288024902344 ,
          309.175048828125  ,  308.40380859375   ,  307.6185302734375 ,
          307.21099853515625,  307.2699279785156 ,  307.2888488769531 ,
          307.2987060546875 ,  307.1123962402344 ,  307.02117919921875,
          307.0187683105469 ,  307.0152282714844 ,  307.00909423828125,
          306.99267578125   ,  306.9793701171875 ]
    t = [0.00352346641011536, 0.00707153975963592, 0.010553872212767601,
         0.01407294347882270, 0.01752243563532829, 0.02106502465903759 ,
         0.02454467304050922, 0.02785783261060714, 0.031417425721883774,
         0.03496412560343742, 0.0385206900537014 , 0.042075447738170624,
         0.04558039829134941, 0.04902618378400802, 0.05211244896054268 ,
         0.05568173155188560, 0.05925865843892097, 0.06280753016471863 ,
         0.06639063358306885, 0.06998835504055023, 0.07358630001544952 ,
         0.07710118591785431, 0.08069833368062973, 0.08426941186189651 ,
         0.08788952231407166, 0.09148057550191879, 0.09194891154766083 ,
         0.09196097403764725, 0.09197842329740524, 0.09200788289308548 ,
         0.09208673238754272, 0.09214994311332703]
    E = [0.01772147417068481, 0.02709103375673294, 0.006793275941163301,
         0.00874292198568582, 0.00539836753159761, 0.007498408202081919,
         0.00476538250222802, 0.00282234791666269, 0.003994516562670469,
         0.01004616636782884, 0.00658490229398012, 0.006791558582335710,
         0.00771093275398016, 0.00592870777472853, 0.00812870915979147 ,
         0.00827718339860439, 0.00713903456926345, 0.011037695221602917,
         0.00546147162094712, 0.00651847803965210, 0.008895126171410084,
         0.00659707654267549, 0.01343776658177375, 0.002142422599717974,
         0.00521089043468236, 0.00416701100766658, 0.000186472039786167,
         0.00014594482490792, 0.00001760456325428, 0.000024740596927586,
         0.00006724067497998, 0.00113796326331794]

    efile = os.path.join(ICDATADIR, 'NextFlex_mc_hits.h5')

    name  = 'e-'
    k_eng = 1.0
    vi    = [-396.4495 , -351.88654,  318.9442 , 0.0      ]
    vf    = [-389.12915, -368.3956 ,  304.85535, 0.1059136]
    p     = [ 0.7167202, 0.3602375 , -1.174112]

    return efile, name, vi, vf, p, k_eng, X, Y, Z, E, t


def test_load_mchits_df(mc_particle_and_hits_nexus_data):
    efile, _, _, _, _, _, _, X, Y, Z, E, t = mc_particle_and_hits_nexus_data
    assert_load_hits_good(efile, X, Y, Z, E, t)


def test_load_mchits_df_newformat(mc_particle_and_hits_nexus_data_new):
    efile, _, _, _, _, _, X, Y, Z, E, t = mc_particle_and_hits_nexus_data_new
    assert_load_hits_good(efile, X, Y, Z, E, t)


def assert_load_hits_good(efile, X, Y, Z, E, t):
    hit_df = load_mchits_df(efile)

    evt = 0
    assert np.allclose(X, hit_df.loc[evt].x     .values)
    assert np.allclose(Y, hit_df.loc[evt].y     .values)
    assert np.allclose(Z, hit_df.loc[evt].z     .values)
    assert np.allclose(E, hit_df.loc[evt].energy.values)
    assert np.allclose(t, hit_df.loc[evt].time  .values)


def test_cast_mchits_to_dict(mc_particle_and_hits_nexus_data):
    efile, *_ = mc_particle_and_hits_nexus_data
    hit_df    = load_mchits_df(efile)

    hit_dict  = cast_mchits_to_dict(hit_df)
    for evt, evt_hits in hit_df.groupby(level=0):
        hits_evt  = [[h.X, h.Y, h.Z, h.time, h.E]
                     for h in hit_dict[evt]]
        evt_xyztE = evt_hits[['x', 'y', 'z', 'time', 'energy']]
        assert_allclose(evt_xyztE.values, hits_evt)


def test_cast_mchits_to_dict_same_as_old(mc_particle_and_hits_nexus_data):
    efile, _, _, _, _, _, _, X, Y, Z, E, t = mc_particle_and_hits_nexus_data

    mchits_df   = load_mchits_df(efile)
    mchits_dict = cast_mchits_to_dict(mchits_df)

    with tb.open_file(efile) as h5in:
        old_mchit_dict = _read_mchit_info(h5in)

    assert np.all(mchits_dict.keys() == old_mchit_dict.keys())
    for old_hts, new_hts in zip(old_mchit_dict.values(), mchits_dict.values()):
        for old_hit, new_hit in zip(old_hts, new_hts):
            assert_MChit_equality(old_hit, new_hit)


def test_load_mcparticles_df(mc_particle_and_hits_nexus_data):
    efile, name, vi, vf, p, k_eng, *_ = mc_particle_and_hits_nexus_data
    assert_load_particles_good(efile, name, vi, vf, p, k_eng)


def test_load_mcparticles_df_newformat(mc_particle_and_hits_nexus_data_new):
    efile, name, vi, vf, p, k_eng, *_ = mc_particle_and_hits_nexus_data_new
    assert_load_particles_good(efile, name, vi, vf, p, k_eng)


def assert_load_particles_good(efile, name, vi, vf, p, k_eng):
    mcparticle_df = load_mcparticles_df(efile)

    evt  = 0
    p_id = 1
    particle = mcparticle_df.loc[evt].loc[p_id]
    assert particle.particle_name == name
    assert np.isclose(particle.kin_energy, k_eng)

    init_vtx = particle[['initial_x', 'initial_y',
                         'initial_z', 'initial_t']]
    assert_allclose(init_vtx.tolist(), vi)

    fin_vtx = particle[['final_x', 'final_y',
                        'final_z', 'final_t']]
    assert_allclose(fin_vtx.tolist(), vf)

    init_mom = particle[['initial_momentum_x',
                         'initial_momentum_y',
                         'initial_momentum_z']]
    assert_allclose(init_mom.tolist(), p)


@fixture(scope = 'module')
def new_format_mcsensor_data(ICDATADIR):
    sensor_binning = (0.003 * units.mus,
                      0.010 * units.mus,
                      0.001 * units.mus)
    evt_no         = 1280
    sns_list       = [20005  , 20016  , 57  , 51  ]
    charge_list    = [    1  ,     1  ,  1  ,  1  ]
    time_list      = [   48.0,    27.0, 90.0, 50.0]
    efile = os.path.join(ICDATADIR, 'NextFlex_mc_sensors.h5')
    return efile, sensor_binning, evt_no, sns_list, charge_list, time_list


@fixture(scope = 'module')
def old_format_mcsensor_data(mc_sensors_nexus_data):
    sensor_binning = 100 * units.nanosecond, 1 * units.microsecond
    return mc_sensors_nexus_data[0], sensor_binning


## mark.parametrize currently doesn't work with fixtures
## use different tests which call a common function
def test_get_sensor_binning_old_format(old_format_mcsensor_data):
    file_name, sensor_binning = old_format_mcsensor_data
    check_get_sensor_binning_asserts(file_name, sensor_binning)


def test_get_sensor_binning_new_format(new_format_mcsensor_data):
    file_name, sensor_binning, *_ = new_format_mcsensor_data
    check_get_sensor_binning_asserts(file_name, sensor_binning)


def check_get_sensor_binning_asserts(file_name, sensor_binning):
    binning = get_sensor_binning(file_name)

    assert isinstance(binning, pd.DataFrame)
    assert len(binning) == len(sensor_binning)
    assert np.all(np.isin(binning.bin_width.values, sensor_binning))


def test_load_mcsensor_response_df_old(mc_sensors_nexus_data):
    efile, pmt0_first, pmt0_last, pmt0_tot_samples, sipm_id, sipm = mc_sensors_nexus_data

    ## Should be generalised for other detectors but data
    ## available at the moment is for new.
    ## Should consider adding to test data!
    wfs      = load_mcsensor_response_df(efile, False, 'new', -6400)

    sns_bins = get_sensor_binning(efile)

    pmt_bin  = sns_bins.bin_width[sns_bins.index.str.contains( 'Pmt')].iloc[0]
    sipm_bin = sns_bins.bin_width[sns_bins.index.str.contains('SiPM')].iloc[0]

    ## Check first pmt
    pmt0_id   = 0
    list_evt  = wfs.index.levels[0]
    wf        = wfs.loc[list_evt[0]].loc[pmt0_id]
    n_samp    = len(wf)
    indx0     = (wf.index[0], wf.iloc[0].charge)
    last_tbin = wf.iloc[n_samp - 1].time / pmt_bin
    indx_last = (last_tbin, wf.iloc[n_samp - 1].charge)

    assert n_samp    == pmt0_tot_samples
    assert indx0     == pmt0_first
    assert indx_last == pmt0_last

    ## Check chosen SiPM
    sipm_wf = wfs.loc[list_evt[0]].loc[sipm_id]
    bin_q   = [(sipm_wf.iloc[i].time / sipm_bin, sipm_wf.iloc[i].charge)
               for i in range(len(sipm_wf))]

    assert np.all(bin_q == sipm)


def test_load_mcsensor_response_df_new(new_format_mcsensor_data):
    efile, _, evt_no, sns_list, charge_list, time_list = new_format_mcsensor_data

    wfs = load_mcsensor_response_df(efile)

    assert len(wfs.loc[evt_no]) == len(sns_list)
    assert np.all(wfs.loc[evt_no].index == sns_list)
    assert np.all(wfs.loc[evt_no].charge == charge_list)
    assert np.all(wfs.loc[evt_no].time == time_list)


def test_get_sensor_types(new_format_mcsensor_data):
    efile, sensor_binning, *_ = new_format_mcsensor_data

    sns_types = get_sensor_types(efile)

    assert len(sns_types.sensor_name.unique()) == len(sensor_binning)


def test_get_mc_tbl_list_raises_NoNode(ICDATADIR):
    file_in = os.path.join(ICDATADIR, 'run_2983.h5')
    with raises(tb.exceptions.NoSuchNodeError):
        get_mc_tbl_list(file_in)
