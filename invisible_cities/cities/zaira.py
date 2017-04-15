from __future__ import print_function

import sys
import time

import numpy as np

from invisible_cities.cities.base_cities    import City, MapCity
from invisible_cities.core  .fit_functions  import in_range
from invisible_cities.core  .configure      import configure
from invisible_cities.reco  .dst_functions  import load_dsts
from invisible_cities.reco  .dst_io         import Corr_writer


class Zaira(City, MapCity):
    def __init__(self, 
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

                 zbins        =  100,
                 zmin         = None,
                 zmax         = None,

                 tbins        =  100,
                 tmin         = None,
                 tmax         = None,

                 z_sampling   = 1000,
                 fiducial_r   = None,
                 fiducial_z   = None,
                 fiducial_e   = None):

        City.__init__(self,
                      run_number  = run_number,
                      files_in    = files_in,
                      file_out    = file_out,
                      compression = compression,
                      nprint      = nprint)

        MapCity.__init__(self,
                         xbins        = xbins,
                         xmin         = xmin,
                         xmax         = xmax,

                         ybins        = ybins,
                         ymin         = ymin,
                         ymax         = ymax,

                         zbins        = zbins,
                         zmin         = zmin,
                         zmax         = zmax,

                         tbins        = tbins,
                         tmin         = tmin,
                         tmax         = tmax,

                         z_sampling   = z_sampling)

        self.fiducial_r = self.det_geo.RMAX[0] if fiducial_r is not None else fiducial_r
        self.fiducial_z = (self.det_geo.ZMIN[0], self.det_geo.ZMIN[1]) if fiducial_z is None else fiducial_z
        self.fiducial_e = (0, np.inf) if fiducial_e is None else fiducial_e

        self.dst_group  = dst_group
        self.dst_node   = dst_node


    def run(self):
        dst = load_dsts(self.input_files, self.dst_group, self.dst_node)
        dst = dst[dst.nS2.values == 1]

        dst = dst[in_range(dst.S2e.values, *self.fiducial_e)]
        dst = dst[in_range(dst.Z.values  , *self.fiducial_z)]
        dst = dst[in_range(dst.X.values  , *self.xrange    )]
        dst = dst[in_range(dst.Y.values  , *self.yrange    )]

        print("Correcting in Z")
        fid      = dst[in_range(dst.R.values, 0, self.fiducial_r)]
        zcorr    = self. z_correction(fid.Z.values, fid.S2e.values, self.zrange[0])

        print("Correcting in XY")
        dst.S2e *= zcorr.fn(dst.Z.values)
        xycorr   = self.xy_correction(dst.X.values, dst.Y.values, dst.S2e.values)

        print("Correcting in T")
        dst.S2e *= xycorr(dst.X.values, dst.Y.values)[0]
        tcorr    = self. t_correction(dst.time.values, dst.S2e.values)

        print("Computing stats")
        nevt     = self.xy_statistics(dst.X.values, dst.Y.values)[0]

        print("Saving to file")
        with Corr_writer(self.output_file) as writer:
            writer.write_z_corr (zcorr.xs, zcorr.fs, zcorr.us)
            writer.write_xy_corr(*xycorr.xs.T, xycorr.fs, xycorr.us, nevt)
            writer.write_t_corr (tcorr.xs, tcorr.fs, tcorr.us)

        return len(dst), zcorr.LT, zcorr.sLT


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

                  xbins        = CFP.XBINS,
                  xmin         = CFP.XMIN if "XMIN" in CFP else None,
                  xmax         = CFP.XMAX if "XMAX" in CFP else None,

                  ybins        = CFP.YBINS,
                  ymin         = CFP.YMIN if "YMIN" in CFP else None,
                  ymax         = CFP.YMAX if "YMAX" in CFP else None,

                  zbins        = CFP.ZBINS,
                  zmin         = CFP.ZMIN if "ZMIN" in CFP else None,
                  zmax         = CFP.ZMAX if "ZMAX" in CFP else None,

                  tbins        = CFP.TBINS,
                  tmin         = CFP.TMIN if "TMIN" in CFP else None,
                  tmax         = CFP.TMAX if "TMAX" in CFP else None,

                  z_sampling   = CFP.Z_SAMPLING,
                  fiducial_r   = CFP.FIDUCIAL_R if "FIDUCIAL_R" in CFP else None,
                  fiducial_z   = CFP.FIDUCIAL_Z if "FIDUCIAL_Z" in CFP else None,
                  fiducial_e   = CFP.FIDUCIAL_E if "FIDUCIAL_E" in CFP else None)

    t0    = time.time()
    nevt, LT, sLT  = zaira.run()
    t1    = time.time()
    dt    = t1 - t0

    if nevt > 0:
        print("run {} evts in {} s, time/event = {}".format(nevt, dt, dt/nevt))
        print("LIFETIME = {} +- {}".format(LT, sLT))

    return nevt, LT, sLT

if __name__ == "__main__":
    ZAIRA(sys.argv)
