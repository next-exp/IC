import numpy  as np
import tables as tb
import pandas as pd

from typing    import   List, Tuple, Callable, Generator

from invisible_cities.io.mcinfo_io import read_mchits_df
from invisible_cities.reco.corrections_new import read_maps


#######################################
############### SOURCE ################
#######################################
def load_MC(files_in : List[str]) -> Generator:
    for filename in files_in:
        with tb.open_file(filename) as h5in:
            extents = pd.read_hdf(filename, 'MC/extents')
            event_ids  = extents.evt_number
            hits_df    = read_mchits_df(h5in, extents)
            for evt in event_ids:
                hits = hits_df.loc[evt, :, :]
                yield dict(event_number = evt,
                           x      = hits["x"]     .values,
                           y      = hits["y"]     .values,
                           z      = hits["z"]     .values,
                           energy = hits["energy"].values)
