"""
-----------------------------------------------------------------------
                              Paolina
-----------------------------------------------------------------------
This city reads beersheba deconvoluted hits.
This city outputs a similar results as esmeralda:
    - Tracking/Tracks: track info of paolina algorithm applied on deconvoluted tracks
    - PHITS/hits     : paolina hits used by the paolina algorithm
"""

import os
import numpy  as np
import tables as tb

from . components import city
from . components import print_every
from . components import collect
from . components import copy_mc_info
from . components import get_event_info

from .. dataflow import dataflow      as fl
from .. reco     import tbl_functions as tbl
from .. database import load_db       as db

from .. io.run_and_event_io import run_and_event_writer
from .. io.event_filter_io  import event_filter_writer
from .. io.hits_io          import hits_writer

# city source
from .. io.dst_io import load_dst
from . components import get_run_number
def decohits_from_files(files_in):
    """ Reads deconvoluted hits from beersheba files"""
    for filename in files_in:
        try:
            deco = load_dst(filename, "DECO", "Events")
        except tb.exceptions.NoSuchNodeError:
            continue

        with tb.open_file(filename, "r") as h5in:
            try:
                run_number = get_run_number(h5in)
                event_info = get_event_info(h5in)
            except (tb.exceptions.NoSuchNodeError, IndexError):
                continue

            for evtinfo in event_info:
                event_number, timestamp = evtinfo.fetch_all_fields()
                yield dict( deco         = deco.loc[deco.event == event_number].copy()
                          , run_number   = run_number
                          , event_number = event_number
                          , timestamp    = timestamp)


from .. io.hits_io import hits_from_df
def create_deco_hitcollection(deco):
    """ Auxiliar function to create the hit collection for DECO table hits.
        DECO hits does not contain Ec and Ep, which are needed to be used with
        hits_from_df function."""
    deco.loc[:, "Ec"]   = deco["E"]
    deco.loc[:, "Ep"]   = deco["E"]
    hitc = hits_from_df(deco)[deco["event"].unique()[0]]
    return hitc

from . esmeralda   import track_blob_info_creator_extractor
from . esmeralda   import track_writer

@city
def paolina(*, files_in, file_out, event_range, print_mod, compression,
            run_number, detector_db, paolina_params):

    # 1 peak filter (select 1S2 events)
    has_1peak  = lambda deco: (deco.npeak.nunique() == 1)
    has_1peak_ = fl.map(has_1peak, args="deco", out="passed_1peak")
    events_passsed_1peak = fl.count_filter(bool, args="passed_1peak")

    # create hit collection
    create_hitcolection = fl.map(create_deco_hitcollection, args="deco", out="hitc")

    # paolina algorithm
    paolina_algorithm = fl.map(track_blob_info_creator_extractor(**paolina_params),
                               args = "hitc", out  = ('topology_info', 'paolina_hits', 'out_of_map'))

    event_count_in = fl.spy_count()
    evtnum_collect = collect()

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:

        write_event_info   = fl.sink(run_and_event_writer(h5out), args=("run_number", "event_number", "timestamp"))
        write_1peak_filter = fl.sink(event_filter_writer(h5out, "one_peak"), args=("event_number", "passed_1peak"))
        write_tracks       = fl.sink(       track_writer(h5out)            , args="topology_info")
        # write_paolina_hits = fl.sink(        hits_writer(h5out, group_name='PHITS', table_name='hits')
        #                                                                    , args="paolina_hits")

        result = fl.push(source= decohits_from_files(files_in),
                         pipe  = fl.pipe( fl.slice(*event_range, close_all=True)
                                        , event_count_in.spy
                                        , fl.branch(write_event_info)
                                        , print_every(print_mod)
                                        , has_1peak_
                                        , fl.branch(write_1peak_filter)
                                        , events_passsed_1peak.filter
                                        , create_hitcolection
                                        , paolina_algorithm
                                        , fl.branch(write_tracks)
                                        # , fl.branch(write_paolina_hits)
                                        , "event_number"
                                        , evtnum_collect.sink),
                         result = dict(events_in     = event_count_in.future,
                                       evtnum_list   = evtnum_collect.future))

        if run_number <= 0:
            copy_mc_info(files_in, h5out, result.evtnum_list, detector_db, run_number)

        return result
