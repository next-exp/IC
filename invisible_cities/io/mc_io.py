from .. reco     import tbl_functions as tbl
from .. reco.nh5 import MCTrack


class mc_track_writer:
    """Write MCTracks to file."""
    def __init__(self, h5file, compression = 'ZLIB4'):

        self.h5file      = h5file
        self.compression = compression
        self._create_mctracks_table()
        # last visited row
        self.last_row = 0

    def _create_mctracks_table(self):
        """Create MCTracks table in MC group in file h5file."""
        if '/MC' in self.h5file:
            MC = self.h5file.root.MC
        else:
            MC = self.h5file.create_group(self.h5file.root, "MC")

        self.mc_table = self.h5file.create_table(MC, "MCTracks",
                        description = MCTrack,
                        title       = "MCTracks",
                        filters     = tbl.filters(self.compression))

    def __call__(self, mctracks, evt_number):
        for r in mctracks.iterrows(start=self.last_row):
            self.last_row += 1
            evt = (evt_number,) + r[1:]
            self.mc_table.append([evt])
        self.mc_table.flush()
