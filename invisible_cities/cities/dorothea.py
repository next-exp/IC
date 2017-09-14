"""
code: dorothea.py
description: create a KDST.
credits: see ic_authors_and_legal.rst in /doc

last revised: JJGC, September-2017
"""

from .. io.kdst_io               import kr_writer
from .  base_cities              import KrCity

class Dorothea(KrCity):
    """Read PMAPS and produces a KDST"""

    def __init__(self, **kwds):
        super().__init__(**kwds)

    def get_writers(self, h5out):
        """Get the writers needed by dorothea"""
        return  kr_writer(h5out)

    def create_dst_event(self, pmapVectors):
        """Get the writers needed by dorothea"""
        return  self.create_kr_event(pmapVectors)

    def write_parameters(self, h5out):
        pass
