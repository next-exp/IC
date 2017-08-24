
from . dst_io              import load_dst
from ..evm.event_model     import Hit
from ..evm.event_model     import Cluster
from ..evm.event_model     import HitCollection
from .. types.ic_types     import xy
from .  table_io           import make_table
from .. evm .nh5           import HitsTable
from .. types.ic_types     import NN


# reader
def load_hits(DST_file_name):
    """Return the Hits as PD DataFrames."""

    dst = load_dst(DST_file_name,'RECO','Events')
    dst_size = len(dst)
    all_events = {}

    event = dst.event.values
    time  = dst.time .values
    npeak = dst.npeak.values
    nsipm = dst.nsipm.values
    X     = dst.X    .values
    Y     = dst.Y    .values
    Xrms  = dst.Xrms .values
    Yrms  = dst.Yrms .values
    Z     = dst.Z    .values
    Q     = dst.Q    .values
    E     = dst.E    .values

    for i in range(dst_size):
        current_event = all_events.setdefault(event[i],
                                              HitCollection(event[i], time[i] * 1e-3))
        hit = Hit(npeak[i],
                 Cluster(Q[i], xy(X[i],Y[i]), xy(Xrms[i],Yrms[i]),
                         nsipm[i], Z[i], E[i]), Z[i], E[i])
        current_event.hits.append(hit)
    return all_events


def load_hits_skipping_NN(DST_file_name):
    """Return the Hits as PD DataFrames."""

    dst = load_dst(DST_file_name,'RECO','Events')
    dst_size = len(dst)
    all_events = {}

    event = dst.event.values
    time  = dst.time .values
    npeak = dst.npeak.values
    nsipm = dst.nsipm.values
    X     = dst.X    .values
    Y     = dst.Y    .values
    Xrms  = dst.Xrms .values
    Yrms  = dst.Yrms .values
    Z     = dst.Z    .values
    Q     = dst.Q    .values
    E     = dst.E    .values

    for i in range(dst_size):
        current_event = all_events.setdefault(event[i],
                                              HitCollection(event[i], time[i] * 1e-3))
        hit = Hit(npeak[i],
                 Cluster(Q[i], xy(X[i],Y[i]), xy(Xrms[i],Yrms[i]),
                         nsipm[i], Z[i], E[i]), Z[i], E[i])
        if(hit.Q != NN):
            current_event.hits.append(hit)
    good_events = {}
    for event, hitc in all_events.items():
        if len(hitc.hits) > 0:
            good_events[event] = hitc

    return good_events

# writers
def hits_writer(hdf5_file, *, compression='ZLIB4'):
    hits_table  = make_table(hdf5_file,
                             group       = 'RECO',
                             name        = 'Events',
                             fformat     = HitsTable,
                             description = 'Hits',
                             compression = compression)

    hits_table.cols.event.create_index()

    def write_hits(hits_event):
        hits_event.store(hits_table)
    return write_hits
