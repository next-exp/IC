"""
code: penthesila.py
description: Read PMAPS and produces hits and beyond
credits: see ic_authors_and_legal.rst in /doc

last revised: JJGC, September-2017
"""

from .. io.hits_io             import hits_writer
from .. cities.base_cities     import HitCity

class Penthesilea(HitCity):
    """Read PMAPS and produces a HDST"""

    def __init__(self, **kwds):
        super().__init__(**kwds)

    def get_writers(self, h5out):
        """Get the writers needed by dorothea"""
        return  hits_writer(h5out)

    def create_dst_event(self, pmapVectors):
        """Get the writers needed by dorothea"""
        return  self.create_hits_event(pmapVectors)

    def write_parameters(self, h5out):
        pass
