"""
code: dorothea.py
description: create a KDST.
credits: see ic_authors_and_legal.rst in /doc

last revised: JJGC, September-2017
"""

from argparse import Namespace

from .. io.kdst_io  import kr_writer
from .  base_cities import KrCity

class Dorothea(KrCity):
    """Read PMAPS and produces a KDST"""

    def __init__(self, **kwds):
        super().__init__(**kwds)

    def get_writers(self, h5out):
        return Namespace(dst = kr_writer(h5out),
                         mc  = self.get_mc_info_writer(h5out))

    def create_dst_event(self, pmapVectors, filter_output):
        return  self.create_kr_event(pmapVectors, filter_output)

    def write_parameters(self, h5out):
        pass
