import os

import warnings

import numpy  as np
import tables as tb

from pytest import raises
from pytest import mark
from pytest import param

from .. core                   import           system_of_units as units
from .. core    .configure     import                 configure
from .. core    .testing_utils import    assert_tables_equality
from .. database               import                   load_db
from .. io      .mcinfo_io     import load_mcsensor_response_df

from .  buffy import buffy


def test_buffy_kr(config_tmpdir, full_sim_file):
    file_in  = full_sim_file
    file_out = os.path.join(config_tmpdir, 'Kr_fullsim.buffers.h5')

    buffer_length =  800 * units.mus
    pmt_samp_wid  =  100 * units.ns
    sipm_samp_wid =    1 * units.mus
    n_pmt         =   12
    n_sipm        = 1792

    sipm_ids      = load_db.DataSiPM('new', -6400).SensorID.values

    conf = configure('buffy invisible_cities/config/buffy.conf'.split())
    conf.update(dict(files_in      = file_in      ,
                     file_out      = file_out     ,
                     buffer_length = buffer_length))

    buffy(**conf)
    with tb.open_file(file_out, mode='r') as h5out:
        assert hasattr(h5out.root    ,           'MC')
        assert hasattr(h5out.root    ,          'Run')
        assert hasattr(h5out.root    ,        'pmtrd')
        assert hasattr(h5out.root    ,       'sipmrd')
        assert hasattr(h5out.root.MC ,         'hits')
        assert hasattr(h5out.root.MC ,    'particles')
        assert hasattr(h5out.root.MC , 'sns_response')
        assert hasattr(h5out.root.Run,      'runInfo')
        assert hasattr(h5out.root.Run,       'events')
        assert hasattr(h5out.root.Run,     'eventMap')

        sns_resp     = load_mcsensor_response_df(file_in        ,
                                                 db_file = 'new',
                                                 run_no  = -6400)
        sns_sums      = sns_resp.loc[0].groupby('sensor_id').charge.sum()
        pmt_sum_indx  = sns_sums.index[sns_sums.index < 12]
        sipm_sum_indx = sns_sums.index[sns_sums.index > 12]
        pmt_buffers   = h5out.root.pmtrd [:]
        sipm_buffers  = h5out.root.sipmrd[:]

        assert pmt_buffers .shape[0] == len(sns_resp.index.levels[0])
        assert pmt_buffers .shape[1] == n_pmt
        assert pmt_buffers .shape[2] == int(buffer_length //  pmt_samp_wid)

        assert sipm_buffers.shape[0] == len(sns_resp.index.levels[0])
        assert sipm_buffers.shape[1] == n_sipm
        assert sipm_buffers.shape[2] == int(buffer_length // sipm_samp_wid)

        pmt_integ   = pmt_buffers[0].sum(axis=1)
        pmt_nonzero = np.argwhere(pmt_integ != 0).flatten()
        assert np.all(pmt_nonzero            ==                  pmt_sum_indx)
        assert np.all(pmt_integ[pmt_nonzero] == sns_sums[sns_sums.index < 12])

        sipm_integ   = sipm_buffers[0].sum(axis=1)
        sipm_nonzero = np.argwhere(sipm_integ != 0).flatten()
        assert np.all(sipm_ids  [sipm_nonzero] ==                 sipm_sum_indx)
        assert np.all(sipm_integ[sipm_nonzero] == sns_sums[sns_sums.index > 12])


def test_buffy_no_file_without_sns_response(config_tmpdir, ICDATADIR):
    """
    Check that if a file has no sensor response, the code raises an exception
    and no output file is created.
    """
    file_in  = os.path.join(ICDATADIR, 'nexus_new_kr83m_fast.oldformat.sim.h5')
    file_out = os.path.join(config_tmpdir, 'test_buffy_no_file_without_sns_response.h5')

    nevt = 2
    conf = configure('buffy invisible_cities/config/buffy.conf'.split())
    conf.update(dict(files_in=file_in, file_out=file_out, event_range=nevt))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        buffy_result = buffy(**conf)
    assert not os.path.exists(file_out)
    assert buffy_result.events_in   == 0
    assert buffy_result.events_resp == 0
    assert buffy_result.evtnum_list == []


def test_buffy_filters_empty(config_tmpdir, ICDATADIR):
    """
    Check that events without sensor response are filtered out.
    We use a fast simulation file to mimic events with no sensor response.
    """
    files_in = [os.path.join(ICDATADIR, 'nexus_new_kr83m_full.oldformat.sim.h5'),
                os.path.join(ICDATADIR, 'nexus_new_kr83m_fast.oldformat.sim.h5')]
    file_out = os.path.join(config_tmpdir, 'test_buffy_filters_empty.h5')

    nevt     = 4
    n_passed = 2
    conf = configure('buffy invisible_cities/config/buffy.conf'.split())
    conf.update(dict(files_in=files_in, file_out=file_out, event_range=nevt))

    # Exception expected since no MC sensor response is present
    # in one of the files. Suppress since irrelevant in test.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        buffy_result = buffy(**conf)

    assert buffy_result.events_in            == nevt
    assert buffy_result.events_resp.n_passed == n_passed
    with tb.open_file(file_out) as h5out:

        assert h5out.root.Run.events.shape == (2,)

        assert hasattr(h5out.root        ,         'Filters')
        assert hasattr(h5out.root.Filters, 'detected_events')

        evt_filter = h5out.root.Filters.detected_events.read()
        assert len(evt_filter) == nevt
        assert np.count_nonzero(evt_filter['passed']) == 2


@mark.parametrize("fn_first fn_second".split(),
                  (param("nexus_new_kr83m_fast.oldformat.sim.h5",
                         "nexus_new_kr83m_full.oldformat.sim.h5", marks=mark.xfail),
                  ("nexus_new_kr83m_full.oldformat.sim.h5",
                   "nexus_new_kr83m_fast.oldformat.sim.h5")))
def test_buffy_empty_file(config_tmpdir, ICDATADIR, fn_first, fn_second):
    """
    Check that the code works even if the first file to be read has
    no sensor response.
    """
    file_in_first  = os.path.join(ICDATADIR,  fn_first)
    file_in_second = os.path.join(ICDATADIR, fn_second)
    file_out       = os.path.join(config_tmpdir, 'test_buffy_empty_file.h5')

    nevt = 4
    conf = configure('buffy invisible_cities/config/buffy.conf'.split())
    conf.update(dict(files_in=[file_in_first, file_in_second], file_out=file_out, event_range=nevt))

    # Exception expected since no MC sensor response is present
    # in one of the files. Suppress since irrelevant in test.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        buffy(**conf)


def test_buffy_exact_result(config_tmpdir, ICDATADIR):
    np.random.seed(27)

    file_in     = os.path.join(ICDATADIR                              ,
                               'nexus_new_kr83m_full.newformat.sim.h5')
    file_out    = os.path.join(config_tmpdir, 'exact_result.buffers.h5')
    true_output = os.path.join(ICDATADIR                                  ,
                               'nexus_new_kr83m_full.newformat.buffers.h5')

    conf = configure('buffy invisible_cities/config/buffy.conf'.split())
    conf.update(dict(files_in=file_in, file_out=file_out))

    buffy(**conf)

    tables = ("MC/hits"        , "MC/particles", "MC/sns_positions",
              "MC/sns_response", "Run/events"  , "Run/eventMap"    ,
              "pmtrd"          , "sipmrd"      )
    with tb.open_file(true_output)  as true_output_file:
        with tb.open_file(file_out) as      output_file:
            for table in tables:
                assert hasattr(output_file.root, table)
                got      = getattr(     output_file.root, table)
                expected = getattr(true_output_file.root, table)
                assert_tables_equality(got, expected)


def test_buffy_splits_event(config_tmpdir, ICDATADIR):
    file_in  = os.path.join(ICDATADIR                              ,
                            'nexus_new_kr83m_full.newformat.sim.h5')
    file_out = os.path.join(config_tmpdir,   'split_evt.buffers.h5')

    evt = 1
    conf = configure('buffy invisible_cities/config/buffy.conf'.split())
    conf.update(dict(files_in      = file_in        ,
                     file_out      = file_out       ,
                     event_range   = (evt, evt + 1) ,
                     buffer_length = 100 * units.mus,
                     pre_trigger   =  10 * units.mus,
                     max_time      = 900 * units.mus))

    buffy(**conf)

    out_evts   = [9, 10]

    sns_resp   = load_mcsensor_response_df(file_in).loc[evt]
    sns_sums   = sns_resp.groupby('sensor_id').charge.sum()
    pmt_integ  = sns_sums[sns_sums.index < 12].values
    sipm_integ = sns_sums[sns_sums.index > 12].values
    with tb.open_file(file_out) as h5out:
        assert len(h5out.root.Run.events[:]) == 2
        assert np.all(h5out.root.Run.events  .cols.evt_number[:] == out_evts)

        assert np.all(h5out.root.Run.eventMap.cols.evt_number[:] == out_evts)
        assert np.all(h5out.root.Run.eventMap.cols.nexus_evt [:] ==     evt )

        assert len(h5out.root.pmtrd [:]) == 2
        assert len(h5out.root.sipmrd[:]) == 2

        pmtout_sum  = h5out.root.pmtrd [:].sum(axis=0).sum(axis=1)
        assert np.all(pmtout_sum == pmt_integ)

        sipmout_sum = h5out.root.sipmrd[:].sum(axis=0).sum(axis=1)
        non_zero    = np.argwhere(sipmout_sum != 0).flatten()
        assert np.all(sipmout_sum[non_zero] == sipm_integ)
