import numpy  as np
import pandas as pd

from   typing       import Tuple, Optional

from .. types.symbols       import type_of_signal
from .. types.symbols       import Strictness
from .. core.core_functions import check_if_values_in_interval


def selection_nS_mask_and_checking(dst        : pd.DataFrame                         ,
                                   column     : type_of_signal                       ,
                                   input_mask : Optional[np.array]  = None           ,
                                   interval   : Tuple[float, float] = [0,1]          ,
                                   strictness : Strictness = Strictness.stop_proccess
                                   )->np.array:
    """
    Selects nS1(or nS2) == 1 for a given kr dst and
    returns the mask. It also computes selection efficiency,
    checking if the value is within a given interval, and
    saves histogram parameters.
    Parameters
    ----------
    dst: pd.Dataframe
        Krypton dst dataframe.
    column: type_of_signal
        The function can be appplied over nS1 or nS2.
    input_mask: np.array (Optional)
        Selection mask of the previous cut. If this is the first selection
        /no previous maks is input, input_mask is set to be an all True array.
    interval: length-2 tuple
        If the selection efficiency is out of this interval
        the map production will abort/just warn, depending on "strictness".
    sstrictness: Strictness
        If 'warning', function returns a False if the criteria
        is not matched. If 'stop_proccess' it raises an exception.
    Returns
    ----------
        A mask corresponding to the selected events.
    """
    input_mask = input_mask if input_mask is not None else [True] * len(dst)
    mask             = np.zeros_like(input_mask)
    mask[input_mask] = getattr(dst[input_mask], column.value) == 1

    nevts_after      = dst[mask]      .event.nunique()
    nevts_before     = dst[input_mask].event.nunique()
    eff              = nevts_after / nevts_before
    check_if_values_in_interval(data         = np.array(eff),
                                minval       = interval[0]  ,
                                maxval       = interval[1]  ,
                                display_name = column.value ,
                                strictness   = strictness   ,
                                right_closed = True)

    return mask
