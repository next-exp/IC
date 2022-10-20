import os
import warnings
import numpy  as np
import tables as tb
import pandas as pd

from pytest import mark

from .. core                 import system_of_units as units
from .. core.core_functions  import in_range
from .. core.testing_utils   import assert_dataframes_close
from .. core.testing_utils   import assert_tables_equality
from .. core.configure       import configure
from .. io                   import dst_io as dio
from .. io.mcinfo_io         import load_mchits_df
from .. io.mcinfo_io         import load_mcparticles_df
from .. types.symbols        import all_events
from .. types.symbols        import RebinMethod
from .. types.symbols        import SiPMCharge

from .  sophronia            import sophronia


def test_sophronia_runs(KrMC_pmaps_filename, config_tmpdir):
    PATH_IN   = KrMC_pmaps_filename
    PATH_OUT  = os.path.join(config_tmpdir,'sophronia.h5')
    conf      = configure('dummy invisible_cities/config/sophronia.conf'.split())
    nevt_req  = 10

    conf.update(dict(files_in    = PATH_IN,
                     file_out    = PATH_OUT,
                     event_range = nevt_req))

    cnt = sophronia(**conf)
    assert cnt.events_in  == nevt_req
