from __future__ import print_function

import sys
import time

from invisible_cities.cities.base_cities   import City, MapCity
from invisible_cities.core  .configure     import configure
from invisible_cities.reco  .dst_functions import load_dsts
from invisible_cities.reco  .dst_io        import Corr_writer


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

                 z_sampling   = 1000,
                 fiducial_cut =  100,

                 tbins        =  100,
                 tmin         = None,
                 tmax         = None):

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

                         z_sampling   = z_sampling,
                         fiducial_cut = fiducial_cut,

                         tbins        = tbins,
                         tmin         = tmin,
                         tmax         = tmax)

        self.dst_group = dst_group
        self.dst_node  = dst_node


    def run(self, max_evt=-1):
        dst = load_dsts(self.input_files, self.dst_group, self.dst_node)
        dst = dst[dst.nS2 == 1]

        fid      = dst[dst.R < self.fiducial_cut]
        zcorr    = self. z_correction(fid.Z, fid.S2e)

        dst.S2e *= zcorr.fn(dst.Z)
        xycorr   = self.xy_correction(dst.X, dst.Y, dst.S2e)

        dst.S2e *= xycorr(dst.X.values, dst.Y.values)
        tcorr    = self. t_correction(dst.time, dst.S2e)

        nevt     = self.xy_statistics(dst.X, dst.Y, self.xybins, (self.xrange, self.yrange))

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
                  files_in     = CFP.FILE_IN,
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

                  z_sampling   = CFP.Z_SAMPLING,
                  fiducial_cut = CFP.FIDUCIAL_CUT,

                  tbins        = CFP.TBINS,
                  tmin         = CFP.TMIN if "TMIN" in CFP else None,
                  tmax         = CFP.TMAX if "TMAX" in CFP else None)

    nevts = CFP.NEVENTS if not CFP.RUN_ALL else -1

    t0    = time.time()
    nevt, LT, sLT  = zaira.run(nmax=nevts, print_empty=CFP.PRINT_EMPTY_EVENTS)
    t1    = time.time()
    dt    = t1 - t0

    if nevt > 0:
        print("run {} evts in {} s, time/event = {}".format(nevt, dt, dt/nevt))
        print("LIFETIME = {} +- {}".format(LT, sLT))

    return nevts, nevt, LT, sLT

if __name__ == "__main__":
    ZAIRA(sys.argv)
