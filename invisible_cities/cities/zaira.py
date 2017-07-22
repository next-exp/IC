"""
code: zaira.py
description: computation of correction map
credits: see ic_authors_and_legal.rst in /doc

last revised: JJGC, 12-July-2017
"""
import sys
import time
from functools import partial

import numpy  as np
import tables as tb
import pandas as pd
from operator import attrgetter

from .  base_cities         import DstCity
from .. core.fit_functions  import in_range
from .. core.configure      import configure
from .. io.kdst_io          import xy_writer
from ..reco.corrections     import Correction
from ..reco.corrections     import LifetimeCorrection
from .. core                import fit_functions  as fitf

class Zaira(DstCity):

    def __init__(self, **kwds):
        """Zaira Init:
        1. inits base city
        2. inits counters
        3. defines fiducial
        4. gets dst info

        """
        super().__init__(**kwds)
        self.cnt.set_name('zaira')
        self.cnt.set_counter('nmax', value=self.conf.nmax)
        self.set_lifetime_correction()
        self.set_xybins_and_range()
        self.set_z_and_e_fiducial()

    def set_z_and_e_fiducial(self):
        conf = self.conf
        fiducial_z, fiducial_e = conf.fiducial_z, conf.fiducial_e

        self.fiducial_z = ((self.det_geo.ZMIN[0], self.det_geo.ZMAX[0])
                             if fiducial_z is None
                             else fiducial_z)
        self.fiducial_e = (0, np.inf) if fiducial_e is None else fiducial_e

    def set_lifetime_correction(self):
        conf = self.conf
        lifetime =   conf.lifetime
        u_lifetime = conf.u_lifetime

        lifetimes = [lifetime]     if not np.shape(  lifetime) else lifetime
        u_lifetimes = [u_lifetime] if not np.shape(u_lifetime) else u_lifetime
        self.lifetime_corrections = tuple(map(LifetimeCorrection,
                                              lifetimes,
                                              u_lifetimes))

    def set_xybins_and_range(self):
        conf = self.conf
        xmin = self.det_geo.XMIN[0] if conf.xmin is None else conf.xmin
        xmax = self.det_geo.XMAX[0] if conf.xmax is None else conf.xmax
        ymin = self.det_geo.YMIN[0] if conf.ymin is None else conf.ymin
        ymax = self.det_geo.YMAX[0] if conf.ymax is None else conf.ymax
        self.xbins  = conf.xbins
        self.ybins  = conf.ybins
        self.xrange = xmin, xmax
        self.yrange = ymin, ymax


    def run(self):
        """Zaira overwrites the default run method """

        # Correct each dataset with the corresponding lifetime
        for dst, correct in zip(self.dsts, self.lifetime_corrections):
            dst.S2e *= correct(dst.Z.values).value

        # Join datasets
        dst = pd.concat(self.dsts)
        # select fiducial region
        dst = dst[in_range(dst.S2e.values, *self.fiducial_e)]
        dst = dst[in_range(dst.Z  .values, *self.fiducial_z)]
        dst = dst[in_range(dst.X  .values, *self.xrange    )]
        dst = dst[in_range(dst.Y  .values, *self.yrange    )]

        # Compute corrections and stats
        xycorr = self.xy_correction(dst.X.values, dst.Y.values, dst.S2e.values)
        nevt   = self.xy_statistics(dst.X.values, dst.Y.values)[0]

        with tb.open_file(self.output_file, 'w') as h5out:
            write_xy = xy_writer(h5out)
            write_xy(*xycorr._xs, xycorr._fs, xycorr._us, nevt)

        self.cnt.set_counter('n_events_tot', value=len(dst))
        return self.cnt


    def xy_correction(self, X, Y, E):
        xs, ys, es, us = fitf.profileXY(X, Y, E,
                                        self.xbins, self.ybins,
                                        self.xrange, self.yrange)

        norm_index = xs.size//2, ys.size//2
        return Correction((xs, ys), es, us, norm_strategy="index", index=norm_index)

    def xy_statistics(self, X, Y):
        return np.histogram2d(X, Y, (self.xbins, self.ybins),
                                    (self.xrange, self.yrange))
