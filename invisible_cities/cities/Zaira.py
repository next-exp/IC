from __future__ import print_function

from invisible_cities.cities.base_cities   import City, MapCity
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
