import sys
import time

import numpy  as np
import tables as tb
import pandas as pd

from .  base_cities         import City
from .  base_cities         import MapCity
from .. core.fit_functions  import in_range
from .. core.configure      import configure
from .. reco.dst_functions  import load_dst
from .. reco.dst_io         import XYcorr_writer # TODO: remove
from .. reco.dst_io         import xy_writer


class Zaira(MapCity):
    def __init__(self,
                 lifetime,

                 run_number   = 0,
                 files_in     = None,
                 file_out     = None,
                 compression  = 'ZLIB4',
                 nprint       = 10000,

                 dst_group    = "DST",
                 dst_node     = "Events",

                 xbins        =  100,
                 xmin         = None,
                 xmax         = None,

                 ybins        =  100,
                 ymin         = None,
                 ymax         = None,

                 fiducial_z   = None,
                 fiducial_e   = None):

        City.__init__(self,
                      run_number  = run_number,
                      files_in    = files_in,
                      file_out    = file_out,
                      compression = compression,
                      nprint      = nprint)

        MapCity.__init__(self,
                         lifetime,

                         xbins        = xbins,
                         xmin         = xmin,
                         xmax         = xmax,

                         ybins        = ybins,
                         ymin         = ymin,
                         ymax         = ymax)

        self._fiducial_z = ((self.det_geo.ZMIN[0], self.det_geo.ZMAX[0])
                             if fiducial_z is None
                             else fiducial_z)
        self._fiducial_e = (0, np.inf) if fiducial_e is None else fiducial_e

        self._dst_group  = dst_group
        self._dst_node   = dst_node

    def run(self):
        dsts = [load_dst(input_file, self._dst_group, self._dst_node)
                for input_file in self.input_files]

        # Correct each dataset with the corresponding lifetime
        for dst, correct in zip(dsts, self._lifetime_corrections):
            dst.S2e *= correct(dst.Z.values).value

        # Join datasets
        dst = pd.concat(dsts)
        dst = dst[in_range(dst.S2e.values, *self._fiducial_e)]
        dst = dst[in_range(dst.Z  .values, *self._fiducial_z)]
        dst = dst[in_range(dst.X  .values, *self._xrange    )]
        dst = dst[in_range(dst.Y  .values, *self._yrange    )]

        # Compute corrections and stats
        xycorr = self.xy_correction(dst.X.values, dst.Y.values, dst.S2e.values)
        nevt   = self.xy_statistics(dst.X.values, dst.Y.values)[0]

        with tb.open_file(self.output_file, 'w') as h5out:
            write_xy = xy_writer(h5out)
            write_xy(*xycorr._xs, xycorr._fs, xycorr._us, nevt)

        # # Dump to file
        # with XYcorr_writer(self.output_file) as write:
        #     write(*xycorr._xs, xycorr._fs, xycorr._us, nevt)

        return len(dst)


def ZAIRA(argv = sys.argv):
    """ZAIRA DRIVER"""

    # get parameters dictionary
    CFP = configure(argv)

    #class instance
    zaira = Zaira(run_number   = CFP.RUN_NUMBER,
                  files_in     = [CFP.FILE_IN],
                  file_out     = CFP.FILE_OUT,
                  compression  = CFP.COMPRESSION,
                  nprint       = CFP.NPRINT,

                  dst_group    = CFP.DST_GROUP,
                  dst_node     = CFP.DST_NODE,

                  lifetime     = CFP.LIFETIME,
                  xbins        = CFP.XBINS,
                  xmin         = CFP.XMIN if "XMIN" in CFP else None,
                  xmax         = CFP.XMAX if "XMAX" in CFP else None,

                  ybins        = CFP.YBINS,
                  ymin         = CFP.YMIN if "YMIN" in CFP else None,
                  ymax         = CFP.YMAX if "YMAX" in CFP else None,

                  fiducial_z   = CFP.FIDUCIAL_Z if "FIDUCIAL_Z" in CFP else None,
                  fiducial_e   = CFP.FIDUCIAL_E if "FIDUCIAL_E" in CFP else None)

    t0   = time.time()
    nevt = zaira.run()
    t1   = time.time()
    dt   = t1 - t0

    if nevt > 0:
        print("run {} evts in {} s, time/event = {}".format(nevt, dt, dt/nevt))

    return nevt

if __name__ == "__main__":
    ZAIRA(sys.argv)
