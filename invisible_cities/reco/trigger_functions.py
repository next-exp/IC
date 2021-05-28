import numpy  as np

from typing  import List
from typing  import Dict
from typing  import Callable

from .. core.core_functions import in_range

def get_trigger_candidates(signal       : np.ndarray
                          ,event_id     : int           , pmt_id         : int
                          ,channel_conf : Dict[int, int], trigger_conf   : Dict
                          ,baseline     : np.ndarray    , wvf_max_height : int
                          ) -> List:
    """
    Calculates the trigger information of a given signal.

    Parameters
    ----------
    signal         : Deconvolved waveform.
    event_id       : Event ID associated to the waveform.
    pmt_id         : PMT ID of the waveform's PMT.
    channel_conf   : PMT specific trigger configuration.
    trigger_conf   : General trigger configuration.
    baseline       : Array with the baseline value at each time bin.
    wvf_max_height : Maximum height of the waveform.

    Returns
    ----------
    triggers       : List with all the trigger related information.
    """
    # Trigger parameters
    qrange    = [channel_conf['q_min'          ], channel_conf['q_max'   ]]
    wrange    = [channel_conf['time_min'       ], channel_conf['time_max']]
    hrange    = [channel_conf['baseline_dev'   ], channel_conf['amp_max' ]]
    pulse_ext =  channel_conf['pulse_valid_ext']

    fMulti        = trigger_conf['multipeak'    ] # Protection for later peaks
    discard_width = trigger_conf['discard_width'] # Don't store events below certain width

    # Multipeak protection parameters
    if fMulti:
        multi_q = fMulti['q_min'     ] # Minimum charge of later peaks to discard a trigger
        multi_w = int(fMulti['time_min'  ]) # Minimum time width of later peaks to discard a trigger
        multi_t = int(fMulti['time_after']) # Time window for a later peak to be considered

    # Signal above threshold, allowing for pulse_ext counts below threshold
    triggers   = []
    indexes    = np.where(signal > hrange[0])[0]
    if len(indexes) == 0: return triggers # Protection in case min. threshold is not surpassed
    index_sel  = np.diff(indexes)>(pulse_ext+1)
    candidates = np.split(indexes, np.where(index_sel)[0] + 1)

    for p in candidates:
        peak_slice = slice(p[0]-1, p[-1] + pulse_ext + 1) # DAQ gets pulse_ext counts after peaks; also, for Q it takes a bin before TODO Fix for N100 (remove -1)
        peak       = signal[peak_slice]
        if not (len(peak) > max(discard_width, 1)): continue
        width      = len(peak) - 1                        # TODO Remove -1 in N100
        height     = peak[1:].max()                       # TODO Full peak for N100
        peak       = peak[:-2][peak[:-2]>0]               # Only counts above 0 are used for Q, last bin not considered TODO ONLY -1 for N100
        charge     = peak.sum()
        if len(peak)==0: continue
        start_time = p[0]

        # Trigger validity
        trigger_w  = in_range(width , *wrange, left_closed=True , right_closed=False)
        trigger_q  = in_range(charge, *qrange, left_closed=False, right_closed=False)
        trigger_h  = height < hrange[1]

        # Baseline at the start of the pulse
        baseline_t0 = baseline[start_time]
        triggers  .append([event_id   , pmt_id
                          ,start_time , charge        , width    , height
                          ,trigger_q  , trigger_w     , trigger_h, True   # By default multi flag is set as true
                          ,baseline_t0, wvf_max_height
                          ,-1         , -1            , -1])              # Filled later in the coincidence module

    # Extra-peaks protection
    if fMulti is not None:
        # Evaluate if later peaks (must be over discard_width) are candidates for the multipeak protection
        for i, t0 in zip(range(len(triggers[:-1])), triggers[:-1]):
            for t1 in triggers[i+1:]:
                multi_candidate = ((t1[4] >=                 multi_w  ) &
                                   (t1[3] >                  multi_q  ) &
                                   (t1[2] <= t0[2] + t0[4] + multi_t+2)) # Start of trigger + peak width + time window + FPGA delay (2)
                if not multi_candidate:
                    continue
                # In case the later peak ends after the multipeak protection window ends
                if (t1[2]+t1[4]) > (t0[2] + t0[4] + multi_t+2): # Event partially contained in the multipeak window
                    peak_slice = slice(t1[2]-1, t0[2]+t0[4]+multi_t+2+1)
                    peak       = signal[peak_slice]
                    width      = len(peak) - 1 # TODO Remove -1 in N100
                    charge     = peak[peak>0].sum()

                    triggers[i][-6] = (width >= multi_w) & (charge > multi_q)
                else:
                    triggers[i][-6] = False
                if not triggers[i][-6]: break

    return triggers


def retrieve_trigger_information(channel_config : Dict[int, int]
                                ,trigger_config : Dict
                                ) -> Callable:
    """
    Calculates the trigger information of deconvolved waveforms.

    Parameters
    ----------
    channel_config : Channel-specific trigger configuration.
    trigger_config : General trigger configuration.

    Returns
    ----------
    trigger_on_channels : Function that returns each channel's trigger candidates.
    """
    def trigger_on_channels(rwfs         : np.ndarray
                           ,cwfs         : np.ndarray
                           ,baselines    : np.ndarray
                           ,event_number : int):
        triggers = []
        for i, dict_items in enumerate(channel_config.items()):
            pmt_id, pmt_conf = dict_items
            baseline         = baselines[i]
            signal           = cwfs[i] - baseline
            max_height       = rwfs[pmt_id].min()
            triggers.extend(get_trigger_candidates(signal
                                                  ,event_number, pmt_id
                                                  ,pmt_conf    , trigger_config
                                                  ,baseline    , max_height    ))
        return np.array(triggers)
    return trigger_on_channels


def check_trigger_coincidence(coinc_window : int) -> Callable:
    """
    Checks if there is time coincidence between trigger candidates and modifies
    the input list of trigger candidates accordingly.

    Parameters
    ----------
    coinc_window   : Time window to considerate the coincidence condition as valid.

    Returns
    ----------
    pmt_coincidence : Function that checks for coincidence and modififes triggers.
    """

    def pmt_coincidence(triggers : np.ndarray) -> np.ndarray:
        valid_sel  = np.all(triggers[:, 6:10], axis=1).astype('bool') # Use only valid triggers
        times      = triggers[:, 2][valid_sel]         # Times of valid triggers
        ids        = triggers[:, 1][valid_sel]         # Ids   of valid triggers

        # Order by time, index so we can use this to order both times and ids
        order_idx  = np.lexsort((ids, times))
        order_time = times[order_idx]
        pmts       = np.full(ids.shape, -1)
        # Calculate all time differences between each trigger and all later ones
        time_diffs = np.array([order_time[i+1:] - t for i, t in enumerate(order_time)])
        # Since pmts are ordered by time, closest pmt will be the next one in the array.
        pmts[:-1]  = ids[order_idx][1:]
        # Shortest time is the first value of time_diff for each trigger
        shortest   = np.array([t[0] if len(t)>0 else -1 for t in time_diffs])
        # Count how many triggers are within the coincidince window of each trigger
        n_coinc    = np.array([len(t[t<coinc_window])+1 for t in time_diffs])
        # Undo the original ordering
        old_order  = np.argsort(order_idx)
        # Assign the coincidence results to the trigger array
        triggers[valid_sel, -3] = n_coinc [old_order]
        triggers[valid_sel, -2] = shortest[old_order]
        triggers[valid_sel, -1] = pmts    [old_order]

        return triggers
    return pmt_coincidence
