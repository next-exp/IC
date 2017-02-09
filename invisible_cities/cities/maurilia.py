"""
code: maurilia.py
description: extracts MC truth information from HDF5 file (writes
 to a separate file)
author: Josh Renner
IC core team: Jacek Generowicz, JJGC, G. Martinez, J.A. Hernando, J.M Benlloch
package: invisible cities. See release notes and licence
"""
from __future__ import print_function

import sys
from glob import glob
from time import time

import numpy as np
import tables as tb

import invisible_cities.core.tbl_functions as tbl
from invisible_cities.core.configure import configure, print_configuration
from invisible_cities.core.nh5 import FEE, MCTrack
import invisible_cities.core.mpl_functions as mpl
import invisible_cities.core.wfm_functions as wfm
from invisible_cities.core.random_sampling\
     import NoiseSampler as SiPMsNoiseSampler
from invisible_cities.database import load_db
import invisible_cities.sierpe.fee as FE

from   invisible_cities.cities.base_cities import City

class Maurilia(City):
    """
    The city of MAURILIA extracts MC truth PyTable from HDF5 files.
    """
    def __init__(self, run_number=0, files_in=None):
        """
        Initialize with the specified run number and input filess 
        """
        City.__init__(self, run_number=run_number, files_in=files_in)

        self.h5out = None

    def set_output_file(self, output_file, compression='ZLIB4'):
        """Set the output file."""
        filename = output_file
        self.compression = compression
        self.h5out = tb.open_file(filename, "w",
                          filters=tbl.filters(compression))
        self.set_output_file_ = True

    def run(self):
        """
        Run the machine
        """
        n_events_tot = 0
        # run the machine only if in a legal state
        if not self.input_files:
            raise IOError('must set input files before running')
        if len(self.input_files)    == 0:
            raise IOError('input file list is empty')
        if not self.h5out:
            raise IOError('must set output file before running')

        print("""
                 MAURILIA will run
                 Input Files = {}
                 Output File = {}
                          """.format(self.input_files,
                                     self.h5out))

        # create the main pytable
        MC = self.h5out.create_group(self.h5out.root, "MC")
        self.mc_table = self.h5out.\
             create_table(MC, "MCTracks", MCTrack,
                          "MCTracks",
                          tbl.filters("NOCOMPR"))

        # loop over input files
        first=False
        for ffile in self.input_files:

            print("Opening file {}".format(ffile))
            filename = ffile
            try:
                with tb.open_file(filename, "r+") as h5in:
                    mctbl  = h5in.root.MC.MCTracks
                    
                    # add all the rows to the main table 
                    for r in mctbl.iterrows():
                        self.mc_table.append([r[:]])

            except:
                print('error')
                raise

        self.mc_table.flush()
        self.h5out.close()


def MAURILIA(argv=sys.argv):
    """MAURILIA DRIVER"""

    CFP = configure(argv)
    fpp = Maurilia()
    files_in = glob(CFP['FILE_IN'])
    files_in.sort()
    fpp.set_input_files(files_in)
    fpp.set_output_file(CFP['FILE_OUT'],
                        compression=CFP['COMPRESSION'])

    t0 = time()
    fpp.run()
    t1 = time()
    dt = t1 - t0

    print("run in {} s".format(dt))


if __name__ == "__main__":
    MAURILIA(sys.argv)
