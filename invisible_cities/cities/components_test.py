import os

import numpy as np
import tables as tb
import pandas as pd

from argparse  import Namespace
from functools import partial

from pytest import mark
from pytest import raises
from pytest import warns

from .. core.configure     import configure
from .. core.exceptions    import InvalidInputFileStructure
from .. core.exceptions    import          SensorIDMismatch
from .. core.testing_utils import    assert_tables_equality
from .. core               import system_of_units as units
from .. types.symbols      import WfType
from .. types.symbols      import EventRange as ER
from .. types.symbols      import XYReco

from .  components import event_range
from .  components import collect
from .  components import copy_mc_info
from .  components import wf_from_files
from .  components import pmap_from_files
from .  components import compute_xy_position
from .  components import city
from .  components import hits_and_kdst_from_files
from .  components import mcsensors_from_file
from .  components import create_timestamp
from .  components import check_max_time

from .. dataflow   import dataflow as fl

from typing import Union

def _create_dummy_conf_with_event_range(value):
    return Namespace(event_range = value)


@mark.parametrize("given expected".split(),
                  ((       9          , (   9,     )),
                   ( (     9,        ), (   9,     )),
                   ( (     5,       9), (   5,    9)),
                   ( (     5, ER.last), (   5, None)),
                   (  ER.all          , (None,     )),
                   ( (ER.all,        ), (None,     ))))
def test_event_range_valid_options(given, expected):
    conf = _create_dummy_conf_with_event_range(given)
    assert event_range(conf) == expected


@mark.parametrize("given",
                  ( ER.last    ,
                   (ER.last,)  ,
                   (ER.last, 4),
                   (ER.all , 4),
                   ( 1,  2,  3)))

def test_event_range_invalid_options_raises_ValueError(given):
    conf = _create_dummy_conf_with_event_range(given)
    with raises(ValueError):
        event_range(conf)


_rwf_from_files = partial(wf_from_files, wf_type=WfType.rwf)
@mark.parametrize("source filename".split(),
                  ((_rwf_from_files, "defective_rwf_rd_pmtrwf.h5"      ),
                   (_rwf_from_files, "defective_rwf_rd_sipmrwf.h5"     ),
                   (_rwf_from_files, "defective_rwf_run_events.h5"     ),
                   (_rwf_from_files, "defective_rwf_trigger_events.h5" ),
                   (_rwf_from_files, "defective_rwf_trigger_trigger.h5"),
                   (pmap_from_files, "defective_pmp_pmap_all.h5"       ),
                   (pmap_from_files, "defective_pmp_run_events.h5"     )))
def test_sources_invalid_input_raises_InvalidInputFileStructure(ICDATADIR, source, filename):
    full_filename = os.path.join(ICDATADIR, "defective_files", filename)
    s = source((full_filename,))
    with raises(InvalidInputFileStructure):
        next(s)


def write_config_file(filename, **kwargs):
    with open(filename, "w") as f:
        for k, v in kwargs.items():
            # Using repr will format strings and containers with
            # strings correctly (i.e. with quotes). The file will
            # look something like
            # files_in   = ["file_1", "file_2"]
            # file_out   = "an_output_file"
            # run_number = 12345
            f.write(f"{k} = {repr(v)}\n")


@mark.parametrize( "case_ files_in expected".split()
                 , ( ( 0 # a single file without wildcard
                     , "electrons_511keV_z250_RWF.h5"
                     , "electrons_511keV_z250_RWF.h5")
                   , ( 1 # a single wildcard leading to three files
                     , "electrons_*keV_z250_RWF.h5"
                     , ["electrons_511keV_z250_RWF.h5" , "electrons_1250keV_z250_RWF.h5", "electrons_2500keV_z250_RWF.h5"])
                   , ( 2 # a list of two files without wildcards
                     , [ "electrons_511keV_z250_RWF.h5", "electrons_1250keV_z250_RWF.h5"]
                     , [ "electrons_511keV_z250_RWF.h5", "electrons_1250keV_z250_RWF.h5"] )
                   , ( 3 # a list of wildcards leading to n files
                     , [ "electrons_5*keV_z250_RWF.h5", "electrons_1*keV_z250_RWF.h5"]
                     , [ "electrons_511keV_z250_RWF.h5", "electrons_1250keV_z250_RWF.h5"] )
                   , ( 4 # a single file plus a wildcard that overlaps with the previous file
                     , ["electrons_511keV_z250_RWF.h5", "electrons_*keV_z250_RWF.h5"]
                     , ["electrons_511keV_z250_RWF.h5" , "electrons_1250keV_z250_RWF.h5", "electrons_2500keV_z250_RWF.h5"] )
                   )
                 )
@mark.filterwarnings("ignore:files_in contains repeated values")
def test_city_files_in(case_, files_in, expected, config_tmpdir, ICDATADIR):
    """
    Check that all possible files_in inputs are handled properly:
    - a single file
    - a single wildcard
    - a list of files
    - a list of wildcards
    - a list of a file plus a wildcard with overlap
    """
    # need to prepend ICDATADIR here to avoid the mess in the parametrization
    if isinstance(files_in, str): files_in =  os.path.join(ICDATADIR, files_in)
    else                        : files_in = [os.path.join(ICDATADIR, f) for f in files_in]
    if isinstance(expected, str): expected = [os.path.join(ICDATADIR, expected)] # always a list for simplicity
    else                        : expected = [os.path.join(ICDATADIR, f) for f in expected]

    # - files_in, file_out and event range are mandatory city arguments
    # - cities must be annotated, but we keep it simple.
    # - we need to open the output file so it exists because the city
    # attempts to create an index there
    @city
    def dummy_city( files_in    : Union[str, list]
                  , file_out    : str
                  , event_range : tuple):
        with tb.open_file(file_out, "w"): pass
        return files_in

    config_file = os.path.join(config_tmpdir, f"test_city_files_in_case_{case_}.conf")
    file_out    = os.path.join(config_tmpdir, f"test_city_files_in_case_{case_}.h5"  )

    write_config_file( config_file
                     , files_in    = files_in
                     , file_out    = file_out
                     , event_range = 0
                     )

    conf = configure(f"dummy {config_file}".split())

    result = dummy_city(**conf)
    assert sorted(result) == sorted(expected)


@mark.parametrize("order", ((0,1), (1,0)))
def test_city_keeps_input_file_ordering(ICDATADIR, config_tmpdir, order):
    files_in = [ os.path.join(ICDATADIR, "electrons_511keV_z250_RWF.h5")
               , os.path.join(ICDATADIR, "electrons_1250keV_z250_RWF.h5") ]
    files_in = [files_in[i] for i in order]

    @city
    def dummy_city( files_in    : Union[str, list]
                  , file_out    : str
                  , event_range : tuple):
        with tb.open_file(file_out, "w"): pass
        return files_in

    config_file = os.path.join(config_tmpdir, f"test_city_keeps_input_file_ordering.conf")
    file_out    = os.path.join(config_tmpdir, f"test_city_keeps_input_file_ordering.h5"  )

    write_config_file( config_file
                     , files_in    = files_in
                     , file_out    = file_out
                     , event_range = 0
                     )

    conf = configure(f"dummy {config_file}".split())

    result = dummy_city(**conf)

    # no sorting: ensure that the files keep their original ordering
    assert result == files_in


def test_compute_xy_position_depends_on_actual_run_number():
    """
    The channels entering the reco algorithm are the ones in a square of 3x3
    that includes the masked channel.
    Scheme of SiPM positions (the numbers are the SiPM charges):
    x - - - >
    y | 5 5 5
      | X 7 5
      v 5 5 5

    This test is meant to fail if them compute_xy_position function
    doesn't use the run_number parameter.
    """
    minimum_seed_charge = 6*units.pes
    reco_parameters = {'Qthr': 2*units.pes,
                       'Qlm': minimum_seed_charge,
                       'lm_radius': 0*units.mm,
                       'new_lm_radius': 15 * units.mm,
                       'msipm': 9,
                       'consider_masked': True}
    run_number = 6977
    find_xy_pos = compute_xy_position( 'new', run_number
                                     , XYReco.corona, **reco_parameters)

    xs_to_test  = np.array([-65, -65, -55, -55, -55, -45, -45, -45])
    ys_to_test  = np.array([  5,  25,   5,  15,  25,   5,  15,  25])
    xys_to_test = np.stack((xs_to_test, ys_to_test), axis=1)

    charge         = minimum_seed_charge - 1
    seed_charge    = minimum_seed_charge + 1
    charge_to_test = np.array([charge, charge, charge, seed_charge, charge, charge, charge, charge])

    find_xy_pos(xys_to_test, charge_to_test)


@mark.skip(reason="there should not be a default detector db")
def test_city_adds_default_detector_db(config_tmpdir):
    default_detector_db = 'new'
    args = {'files_in'    : 'dummy_in',
            'file_out'    : os.path.join(config_tmpdir, 'dummy_out')}
    @city
    def dummy_city(files_in, file_out, event_range, detector_db):
        with tb.open_file(file_out, 'w'):
            pass
        return detector_db

    db = dummy_city(**args)
    assert db == default_detector_db


@mark.skip(reason="there should not be a default detector db")
def test_city_does_not_overwrite_detector_db(config_tmpdir):
    args = {'detector_db' : 'some_detector',
            'files_in'    : 'dummy_in',
            'file_out'    : os.path.join(config_tmpdir, 'dummy_out')}
    @city
    def dummy_city(files_in, file_out, event_range, detector_db):
        with tb.open_file(file_out, 'w'):
            pass
        return detector_db

    db = dummy_city(**args)
    assert db == args['detector_db']


@mark.skip(reason="there should not be a default detector db")
def test_city_only_pass_default_detector_db_when_expected(config_tmpdir):
    args = {'files_in'    : 'dummy_in',
            'file_out'    : os.path.join(config_tmpdir, 'dummy_out')}
    @city
    def dummy_city(files_in, file_out, event_range):
        with tb.open_file(file_out, 'w'):
            pass

    dummy_city(**args)

def test_hits_and_kdst_from_files(ICDATADIR):
    event_number = 1
    timestamp    = 0.
    num_hits     = 13
    keys = ['hits', 'kdst', 'run_number', 'event_number', 'timestamp']
    file_in     = os.path.join(ICDATADIR    ,  'Kr83_nexus_v5_03_00_ACTIVE_7bar_3evts.HDST.h5')
    generator = hits_and_kdst_from_files([file_in], "RECO", "Events")
    output = next(generator)
    assert set(keys) == set(output.keys())
    assert output['event_number']   == event_number
    assert output['timestamp']      == timestamp
    assert len(output['hits'].hits) == num_hits
    assert type(output['kdst'])     == pd.DataFrame


def test_collect():
    the_source    = list(range(0,10))
    the_collector = collect()
    the_result    = fl.push(source = the_source,
                            pipe   = fl.pipe(the_collector.sink),
                            result = the_collector.future)
    assert the_source == the_result


def test_copy_mc_info_noMC(ICDATADIR, config_tmpdir):
    file_in  = os.path.join(ICDATADIR, 'run_2983.h5')
    file_out = os.path.join(config_tmpdir, 'dummy_out.h5')
    with tb.open_file(file_out, "w") as h5out:
        with warns(UserWarning):
            copy_mc_info([file_in], h5out, [], 'new', -6400)


@mark.xfail
def test_copy_mc_info_repeated_event_numbers(ICDATADIR, config_tmpdir):
    file_in  = os.path.join(ICDATADIR, "Kr83_nexus_v5_03_00_ACTIVE_7bar_10evts.sim.h5")
    file_out = os.path.join(config_tmpdir, "dummy_out.h5")

    with tb.open_file(file_out, 'w') as h5out:
        copy_mc_info([file_in, file_in], h5out, [0,1,0,9])
        events_in_h5out = h5out.root.MC.extents.cols.evt_number[:]
        assert events_in_h5out.tolist() == [0,1,0,9]


def test_copy_mc_info_split_nexus_events(ICDATADIR, config_tmpdir):
    file_in  = os.path.join(ICDATADIR                                       ,
                            "nexus_new_kr83m_full.newformat.splitbuffers.h5")
    file_out = os.path.join(config_tmpdir, "dummy_out.h5")

    with tb.open_file(file_out, 'w') as h5out:
        copy_mc_info([file_in], h5out, [0, 10, 11], 'new', -6400)

    tables = ("MC/hits"        , "MC/particles", "MC/sns_positions",
              "MC/sns_response", "Run/eventMap")
    with tb.open_file(file_in) as h5in, tb.open_file(file_out) as h5out:
        for table in tables:
            assert hasattr(h5out.root, table)
            got      = getattr(h5out.root, table)
            expected = getattr(h5in .root, table)
            assert_tables_equality(got, expected)



def test_mcsensors_from_file_fast_returns_empty(ICDATADIR):
    rate = 0.5
    file_in = os.path.join(ICDATADIR, "nexus_new_kr83m_fast.newformat.sim.h5")
    sns_gen = mcsensors_from_file([file_in], 'new', -7951, rate)
    with warns(UserWarning, match='No binning info available.'):
        first_evt = next(sns_gen)
    assert first_evt[ 'pmt_resp'].empty
    assert first_evt['sipm_resp'].empty


def test_mcsensors_from_file_correct_yield(ICDATADIR):
    evt_no         =    0
    rate           =    0.5
    npmts_hit      =   12
    total_pmthits  = 4303
    nsipms_hit     =  313
    total_sipmhits =  389
    keys           = ['event_number', 'timestamp', 'pmt_resp' , 'sipm_resp']

    file_in   = os.path.join(ICDATADIR, "nexus_new_kr83m_full.newformat.sim.h5")
    sns_gen   = mcsensors_from_file([file_in], 'new', -7951, rate)
    first_evt = next(sns_gen)

    assert set(keys) == set(first_evt.keys())

    assert      first_evt['event_number']                 == evt_no
    assert      first_evt[   'timestamp']                 >= evt_no / rate
    assert type(first_evt[    'pmt_resp'])                == pd.DataFrame
    assert type(first_evt[   'sipm_resp'])                == pd.DataFrame
    assert  len(first_evt[    'pmt_resp'].index.unique()) == npmts_hit
    assert      first_evt[    'pmt_resp'].shape[0]        == total_pmthits
    assert  len(first_evt[   'sipm_resp'].index.unique()) == nsipms_hit
    assert      first_evt[   'sipm_resp'].shape[0]        == total_sipmhits


def test_create_timestamp_greater_with_greater_eventnumber():
    """
    Value of timestamp must be always positive and
    greater with greater event numbers.
    """

    rate_1   =   0.5
    rate_2   =   0.6
    evt_no_1 =  10.
    evt_no_2 = 100.

    timestamp_1 = create_timestamp(rate_1)
    timestamp_2 = create_timestamp(rate_2)

    assert     timestamp_1(evt_no_1)  <  timestamp_2(evt_no_2)


@mark.filterwarnings("ignore:Zero rate"    )
@mark.filterwarnings("ignore:Negative rate")
def test_create_timestamp_physical_rate():
    """
    Check the rate is always physical.
    """

    rate_1   =   0.   * units.hertz
    rate_2   = - 0.42 * units.hertz
    evt_no_1 =  11.
    evt_no_2 = 111.

    timestamp_1 = create_timestamp(rate_1)
    timestamp_2 = create_timestamp(rate_2)

    assert timestamp_1(evt_no_1) >= 0
    assert timestamp_2(evt_no_2) >= 0



@mark.filterwarnings("ignore:`max_time` shorter than `buffer_length`")
def test_check_max_time_eg_buffer_length():
    """
    Check if `max_time` is always equal or greater
        than `buffer_length` and filter warnings.
    """

    max_time_1      =  10 * units.ms
    buffer_length_1 = 800 * units.mus

    max_time_2      = 600 * units.mus
    buffer_length_2 = 700 * units.mus

    max_time_1 = check_max_time(max_time_1, buffer_length_1)
    max_time_2 = check_max_time(max_time_2, buffer_length_2)

    assert max_time_1 >  buffer_length_1
    assert max_time_2 == buffer_length_2

def test_check_max_time_units():
    """
    Check if `check_max_time` rejects values of `max_time`
        that are not multiples of 1 mus.
    """
    max_time      = 1000.5 * units.mus
    buffer_length = 800 * units.mus

    with raises(ValueError):
        check_max_time(max_time, buffer_length)

def test_read_wrong_pmt_ids(ICDATADIR):
    """
    The input file of this test contains sensor IDs that are not present in the database.
    This should raise an error and this test check that it is actually raised.
    """
    file_in    = os.path.join(ICDATADIR, "nexus_next100_full_wrong_PMT_IDs.h5")
    run_number = 0
    rate       = 0.5

    sns_gen = mcsensors_from_file([file_in], 'next100', run_number, rate)
    with raises(SensorIDMismatch):
        next(sns_gen)
