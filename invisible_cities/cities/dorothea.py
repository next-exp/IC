"""
code: dorothea.py
description: create a KDST.
credits: see ic_authors_and_legal.rst in /doc

last revised: JJGC, September-2017
"""

rom .. io.kdst_io               import kr_writer
from .. evm.ic_containers       import PmapVectors

from .. filters.s1s2_filter    import S12Selector
from .  base_cities            import KrCity


class Dorothea(KrCity):

    def __init__(self, **kwds):
        super().__init__(**kwds)

        self.drift_v = self.conf.drift_v
        self._s1s2_selector = S12Selector(**kwds)

    def get_writers(self, h5out):
        """Get the writers needed by dorothea"""
        return  kr_writer(h5out)

    def create_dst_event(self, pmapVectors):
        """Get the writers needed by dorothea"""
        return  self.create_kr_event(pmapVectors)

    def s12_selector(self):
        """Return the selector defined for Dorothea"""
        return self._s1s2_selector
