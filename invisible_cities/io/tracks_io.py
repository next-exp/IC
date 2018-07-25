from .  table_io          import make_table
from .. evm.event_model   import TrackCollection
from .. evm.nh5           import TracksTable

def tracks_writer(hdf5_file, *, compression='ZLIB4'):
    tracks_table = make_table(hdf5_file,
                          group       = 'Voxels',
                          name        = 'Tracks',
                          fformat     = TracksTable,
                          description = 'Paolina Tracks',
                          compression = compression)
    # Mark column to index after populating table
    tracks_table.set_attr('columns_to_index', ['event'])

    def write_tracks(tracks_event):
        tracks_event.store(tracks_table)
    return write_tracks
