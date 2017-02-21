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

import numpy  as np
import tables as tb

import invisible_cities.core.tbl_functions as tbl
from   invisible_cities.core.configure import configure, print_configuration
from   invisible_cities.core.nh5 import MCTrack
from   invisible_cities.cities.base_cities import City

class Maurilia(City):
    """
    The city of MAURILIA extracts MC truth PyTable from HDF5 files.
    """
    def __init__(self, run_number=0, files_in=None, file_out=None):
        """
        Initialize with the specified run number and input filess
        """
        City.__init__(self,
                      run_number = run_number,
                      files_in   = files_in,
                      file_out   = file_out)


    def display_IO_info(self):
        print("""
                 {} will run with
                 Input Files = {}
                 Output File = {}
                          """.format(self.__class__.__name__,
                                     self.input_files, self.output_file))

    # TODO: add nmax
    def run(self):
        """
        Run the machine
        """
        n_events_tot = 0

        self.check_files()
        self.display_IO_info()

        with tb.open_file(self.output_file, "w",
                          filters = tbl.filters(self.compression)) as h5out:

            # create the main pytable
            MC = h5out.create_group(h5out.root, "MC")
            self.mc_table = h5out.create_table(MC, "MCTracks",
                                               description = MCTrack,
                                               title       = "MCTracks",
                                               filters     = tbl.filters("NOCOMPR"))

            # loop over input files
            for ffile in self.input_files:

                print("Opening file {}".format(ffile))
                filename = ffile
                with tb.open_file(filename, "r") as h5in:
                    mctbl  = h5in.root.MC.MCTracks

                    # add all the rows to the main table
                    for r in mctbl.iterrows():
                        self.mc_table.append([r[:]])

            self.mc_table.flush()



def MAURILIA(argv=sys.argv):
    """MAURILIA DRIVER"""

    CFP = configure(argv)
    fpp = Maurilia()
    files_in = glob(CFP['FILE_IN'])
    files_in.sort()
    fpp.set_input_files(files_in)
    fpp.set_output_file(CFP['FILE_OUT'])
    fpp.set_compression(CFP['COMPRESSION'])

    t0 = time()
    fpp.run()
    t1 = time()
    dt = t1 - t0

    print("run in {} s".format(dt))


if __name__ == "__main__":
    MAURILIA(sys.argv)
