import numpy as np

from pytest                    import fixture
from pytest                    import mark
from hypothesis                import given
from numpy     .testing        import assert_allclose
from hypothesis.strategies     import integers
from hypothesis.strategies     import lists
from hypothesis.strategies     import booleans
from hypothesis.strategies     import composite

from .. reco.trigger_functions import get_trigger_candidates
from .. reco.trigger_functions import retrieve_trigger_information
from .. reco.trigger_functions import check_trigger_coincidence
from .. reco.trigger_functions import TriggerInfo
from .. core.core_functions    import in_range


def channel_conf(**kwargs):
    channel_dict = {'q_min'          : 5000, 'q_max'   :  50000,
                    'time_min'       : 2000, 'time_max':  10000,
                    'baseline_dev'   :    5, 'amp_max' :   1000,
                    'pulse_valid_ext':   15}
    channel_dict.update(kwargs)
    return channel_dict


def trigger_conf(**kwargs):
    trigger_dict = {'coincidence_window':64, 'discard_width':40, 'multipeak':None}
    trigger_dict.update(kwargs)
    return trigger_dict


@composite
def double_square_wf(draw):
    # Fake signal with two-nearby square pulses and a smaller additional one at the beginning.
    n_baseline  = 64000
    base_window = 1024
    thr         = 5
    wf          = np.zeros(n_baseline)
    start1      = draw(integers(2*base_window   , n_baseline // 2))
    length      = draw(integers(n_baseline // 50, n_baseline // 4))
    stop1       = start1 + length
    start2      = stop1 + 10
    stop2       = start2 + length

    heights = draw(lists(integers(thr+1, 50), min_size=3, max_size=3))
    wf[start1:stop1] = np.full(length, heights[0])
    wf[start2:stop2] = np.full(length, heights[1])
    wf[1500  : 1520] = np.full(20    , heights[2]) # Extra pulse to check discard_width

    return wf, thr, length, start1, start2


@composite
def trigger_time_and_validity(draw):
    # Set of trigger times an validity flags, to test trigger coincidence
    size  = draw (integers(min_value=1, max_value=100))
    flag  = lists(integers(min_value=0, max_value=1  ), min_size=4, max_size=4)

    i     = draw(lists(integers(min_value=0), min_size=size, max_size=size))
    flags = draw(lists(flag                 , min_size=size, max_size=size))
    return (i, flags)


@given(double_square_wf=double_square_wf(), evt_id=integers(1, 10), pmt_id=integers(1, 10))
def test_get_trigger_candidates_single(double_square_wf, evt_id, pmt_id):
    # Given a square pulse check that is triggering where the pulse starts and values match the expected ones.
    channel_config = channel_conf()
    trigger_config = trigger_conf()

    wf     = double_square_wf[0]
    thr    = double_square_wf[1]
    length = double_square_wf[2]
    start1 = double_square_wf[3]
    start2 = double_square_wf[4]

    peak   = wf[start1:start2+length+channel_config['pulse_valid_ext']]
    t_true = start1
    q_true = peak[:-2].sum()
    val_q  = in_range(q_true
                     , channel_config['q_min'], channel_config['q_max']
                     , left_closed=False, right_closed=False)
    w_true = len(peak)
    val_w  = in_range(w_true
                     , channel_config['time_min'], channel_config['time_max']
                     , left_closed=True , right_closed=False)
    h_true = peak.max()
    val_h  = h_true < channel_config['amp_max']
    val_p  = True
    val_t  = all([val_q, val_w, val_h, val_p])

    triggers = get_trigger_candidates(wf, evt_id, pmt_id
                                     ,channel_config, trigger_config
                                     ,np.zeros(wf.shape), wf.max())

    elements = [evt_id    , pmt_id
               ,t_true    , q_true  , w_true, h_true
               ,val_q     , val_w   , val_h , val_p , val_t
               ,wf[start1-1], wf.max()
               ,-1        , -1      , -1]

    assert len(triggers) == 1
    assert elements      == list(triggers[0])


@given(double_square_wf=double_square_wf(), evt_id=integers(1, 10), pmt_id=integers(1, 10), pulse_valid=integers(0, 9))
def test_get_trigger_candidates_double(double_square_wf, evt_id, pmt_id, pulse_valid):
    # Check that the square pulses are identified as separated when pulse valid ext is lower than their separation.
    channel_config = channel_conf(pulse_valid_ext=pulse_valid)
    trigger_config = trigger_conf()

    wf     = double_square_wf[0]
    thr    = double_square_wf[1]
    length = double_square_wf[2]
    start1 = double_square_wf[3]
    start2 = double_square_wf[4]

    triggers = get_trigger_candidates(wf, evt_id, pmt_id
                                     ,channel_config, trigger_config
                                     ,np.zeros(wf.shape), wf.max())
    assert len(triggers) == 2
    for j, start in enumerate([start1, start2]):
        peak   = wf[start:start+length+pulse_valid]
        t_true = start
        q_true = peak[:-2].sum()
        val_q  = in_range(q_true
                         , channel_config['q_min'], channel_config['q_max']
                         , left_closed=False, right_closed=False)
        w_true = len(peak)
        val_w  = in_range(w_true
                         , channel_config['time_min'], channel_config['time_max']
                         , left_closed=True , right_closed=False)
        h_true = peak.max()
        val_h  = h_true < channel_config['amp_max']
        val_p  = True
        val_t  = all([val_q, val_w, val_h, val_p])

        elements = [evt_id    , pmt_id
                   ,t_true    , q_true  , w_true, h_true
                   ,val_q     , val_w   , val_h , val_p , val_t
                   ,wf[start1-1], wf.max()
                   ,-1        , -1      , -1]

        assert elements == list(triggers[j])


@given(double_square_wf=double_square_wf(), evt_id=integers(1, 10), pmt_id=integers(1, 10), discard=integers(0, 30))
def test_get_trigger_candidates_discard(double_square_wf, evt_id, pmt_id, discard):
    # Check that the small pulse is kept when discard_width is low enough
    channel_config = channel_conf()
    trigger_config = trigger_conf(discard_width=discard)

    wf     = double_square_wf[0]
    thr    = double_square_wf[1]
    length = double_square_wf[2]
    start1 = double_square_wf[3]
    start2 = double_square_wf[4]

    triggers = get_trigger_candidates(wf, evt_id, pmt_id
                                     ,channel_config, trigger_config
                                     ,np.zeros(wf.shape), wf.max())
    assert len(triggers) == 2
    for j in range(2):
        if j == 0:
            peak   = wf[1500:1520+channel_config['pulse_valid_ext']]
            t_true = 1500
        else:
            peak   = wf[start1:start2+length+channel_config['pulse_valid_ext']]
            t_true = start1
        q_true = peak[:-2].sum()
        val_q  = in_range(q_true
                         , channel_config['q_min'], channel_config['q_max']
                         , left_closed=False, right_closed=False)
        w_true = len(peak)
        val_w  = in_range(w_true
                         , channel_config['time_min'], channel_config['time_max']
                         , left_closed=True , right_closed=False)
        h_true = peak.max()
        val_h  = h_true < channel_config['amp_max']
        val_p  = True
        val_t  = all([val_q, val_w, val_h, val_p])

        elements = [evt_id    , pmt_id
                   ,t_true    , q_true  , w_true, h_true
                   ,val_q     , val_w   , val_h , val_p , val_t
                   ,wf[start1-1], wf.max()
                   ,-1        , -1      , -1]

        assert elements == list(triggers[j])


@given(double_square_wf=double_square_wf(), evt_id=integers(1, 10), pmt_id=integers(1, 10))
def test_get_trigger_candidates_multipeak(double_square_wf, evt_id, pmt_id):
    # Check that the multipeak protection discards the trigger if conditions are met.
    channel_config = channel_conf(pulse_valid_ext=0)
    trigger_config = trigger_conf()

    wf     = double_square_wf[0]
    thr    = double_square_wf[1]
    length = double_square_wf[2]
    start1 = double_square_wf[3]
    start2 = double_square_wf[4]

    triggers = get_trigger_candidates(wf, evt_id, pmt_id
                                     ,channel_config, trigger_config
                                     ,np.zeros(wf.shape), wf.max())
    for t in triggers:
        assert t.valid_all  == all(t[6:10])
        assert t.valid_peak == True


    ### Second peak characteristics
    peak   = wf[start2:start2+length+channel_config['pulse_valid_ext']]
    t_true = start2
    q_true = peak[:-2].sum()
    val_q  = in_range(q_true
                     , channel_config['q_min'], channel_config['q_max']
                     , left_closed=False, right_closed=False)
    w_true = len(peak)
    val_w  = in_range(w_true
                     , channel_config['time_min'], channel_config['time_max']
                     , left_closed=True , right_closed=False)
    h_true = peak.max()
    val_h  = h_true < channel_config['amp_max']
    val_p  = True
    val_t  = all([val_q, val_w, val_h, val_p])

    # Multipeak condition metretrieve_trigger_information
    trigger_config['multipeak'] = {'q_min':q_true//2, 'time_min':w_true//2, 'time_after':length+10}
    triggers = get_trigger_candidates(wf, evt_id, pmt_id
                                     ,channel_config, trigger_config
                                     ,np.zeros(wf.shape), wf.max())

    assert triggers[0].valid_peak == False
    assert triggers[1].valid_peak == True
    for t in triggers:
        assert t.valid_all == all(t[6:10])

    # Multipeak condition no longer met
    trigger_config['multipeak'] = {'q_min':q_true*2, 'time_min':w_true*2, 'time_after':length+10}
    triggers = get_trigger_candidates(wf, evt_id, pmt_id
                                     ,channel_config, trigger_config
                                     ,np.zeros(wf.shape), wf.max())

    for t in triggers:
        assert t.valid_all == all(t[6:10])
        assert t.valid_peak == True


@given(double_square_wf=double_square_wf(), evt_id=integers(0, 3))
def test_retrieve_trigger_information(double_square_wf, evt_id):
    channel_config = channel_conf()
    trigger_config = trigger_conf()

    retriever = retrieve_trigger_information({0:channel_config, 1:channel_config}, trigger_config)

    wf     = double_square_wf[0]
    wfs    = np.array([i*wf for i in range(1, 4)])
    thr    = double_square_wf[1]
    length = double_square_wf[2]
    start1 = double_square_wf[3]
    start2 = double_square_wf[4]

    triggers  = retriever(-wfs, wfs[:2], np.zeros(wfs.shape), evt_id)

    assert len(triggers) == 2
    for j in range(2):
        peak   = wfs[j][start1:start2+length+channel_config['pulse_valid_ext']]
        t_true = start1
        q_true = peak[:-2].sum()
        val_q  = in_range(q_true
                         , channel_config['q_min'], channel_config['q_max']
                         , left_closed=False, right_closed=False)
        w_true = len(peak)
        val_w  = in_range(w_true
                         , channel_config['time_min'], channel_config['time_max']
                         , left_closed=True , right_closed=False)
        h_true = peak.max()
        val_h  = h_true < channel_config['amp_max']
        val_p  = True
        val_t  = all([val_q, val_w, val_h, val_p])

        elements = [evt_id    , j
                   ,t_true    , q_true  , w_true, h_true
                   ,val_q     , val_w   , val_h , val_p , val_t
                   ,wf[start1-1], -wfs[j].max()
                   ,-1        , -1      , -1]

        assert np.allclose(elements, list(triggers[j]))


@given(trigger_time_and_validity(), integers())
def test_check_trigger_coincidence(trigger_info, coincidence_window):
    trigger_time = trigger_info[0]
    validity     = trigger_info[1]
    trigg        = np.zeros((len(validity), 16), dtype='object')

    # Initiate sim. trigger values
    trigg[:, 1   ] = list(range(len(validity)))
    trigg[:, 2   ] = trigger_time
    trigg[:, 6:11] = True#validity
    trigg[:, -3: ] = np.full((len(validity), 3), -1)

    order_idx  = np.lexsort((trigg[:,1], trigger_time))
    old_order  = np.argsort(order_idx)

    ordered_times = trigg[:, 2   ][order_idx]
    ordered_val   = trigg[:, 6:11][order_idx]
    ordered_ids   = trigg[:, 1   ][order_idx]
    trigger_back  = trigg[order_idx].copy()

    for i, time1 in enumerate(ordered_times):
        if not np.all(ordered_val[i]): continue
        trigger_back[i, -3] = 1
        nearest = False
        for val2, id2, time2 in zip(ordered_val[i+1:], ordered_ids[i+1:], ordered_times[i+1:]):
            if not np.all(val2): continue
            if not nearest:
                trigger_back[i, -2] = time2-time1
                trigger_back[i, -1] = id2
                nearest = True
            if (time2-time1)<coincidence_window:
                trigger_back[i, -3] += 1
            else: break

    checks_coinc = check_trigger_coincidence(coincidence_window)
    new_t        = checks_coinc([TriggerInfo(*t) for t in trigg])

    for t_old, t_new in zip(trigger_back[old_order], new_t):
        assert list(t_old) == list(t_new)
