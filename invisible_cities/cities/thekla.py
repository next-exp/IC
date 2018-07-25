"""
code: thekla.py
description: Read HITS and produces an ntuple
credits: see ic_authors_and_legal.rst in /doc

last revised: jrenner, July 2018
"""

from argparse import Namespace

from .. io.ntuple_io         import ntuple_writer
from .. cities.base_cities   import NtupleCity

class Thekla(NtupleCity):
    """Reads HITS and produces an Ntuple"""

    def __init__(self, **kwds):
        super().__init__(**kwds)

    def get_writers(self, h5out):
        return Namespace(dst = ntuple_writer(h5out),
                         mc  = self.get_mc_info_writer(h5out))

    def write_parameters(self, h5out):
        pass
