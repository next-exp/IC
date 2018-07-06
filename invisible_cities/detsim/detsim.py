"""
code: detsim.py
Simulation of sensor responses starting from Nexus output.

An HDF5 file containing Nexus output is given as input, and the simulated
detector response resulting from the Geant4 ionization tracks stored in this
file is produced.
"""

import numpy as np

from argparse import Namespace

from .. cities.base_cities           import City
from .. io.mcinfo_io                 import load_mchits

from .. io.pmaps_io                  import pmap_writer
from .. io.run_and_event_io          import run_and_event_writer
from .. io.voxels_io                 import voxels_writer

from .. reco.paolina_functions       import voxelize_hits
from .. detsim.detsim_functions      import diffuse_and_smear_hits

class Detsim(City):
    """Simulates detector response for events produced by Nexus"""

    parameters = tuple("""zmin
      zmax zmax diff_transv diff_long resolution_FWHM
      Qbb write_true_voxels true_voxel_dimensions A_sipm d_sipm
      ze_sipm ze_pmt slice_width_sipm E_to_Q_sipm uniformlight_frac_sipm
      s2_threshold_sipm slice_width_pmt E_to_Q_pmt uniformlight_frac_pmt
      s2_threshold_pmt peak_space""".split())

    def __init__(self, **kwds):
        """actions:
        1. inits base city
        2. inits event counter

        """
        super().__init__(**kwds)

        self.cnt.init(n_events_tot = 0)


    def file_loop(self):
        """
        The file loop of TDetSim:
        1. read the input Nexus files
        2. pass the hits to the event loop

        """
        for filename in self.input_files:
            mchits_dict = load_mchits(filename, self.conf.event_range)
            self.event_loop(mchits_dict)


    def event_loop(self, mchits_dict):
        """
        The event loop of TDetSim:
        1. diffuse and apply energy smearing to all hits in each event
        2. create true voxels from the diffused/smeared hits

        """
        write = self.writers

        for evt_number, mchits in mchits_dict.items():

            print("Event {}".format(evt_number))

            dmchits,zdrift = diffuse_and_smear_hits(mchits, self.conf.zmin,
                                                    self.conf.zmax,
                                                    self.conf.diff_transv,
                                                    self.conf.diff_long,
                                                    self.conf.resolution_FWHM,
                                                    self.conf.Qbb)

            voxels = voxelize_hits(dmchits, self.conf.true_voxel_dimensions)
            vc = VoxelCollection(evt_number, 0.)
            vc.voxels = voxels_b

            write.true_voxels(evt_number,vc)
            write.run_and_event(self.run_number, evt_number, 0)

        self.cnt.n_events_tot += len(mchits_dict)


    def get_writers(self, h5out):
        writers = Namespace(
        run_and_event = run_and_event_writer(h5out),
        true_voxels = voxels_writer(h5out)
        )
        return writers


    def write_parameters(self, h5out):
        pass
