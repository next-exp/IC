import os


from .  detsim         import Detsim
from .. core.configure import configure

def test_detsim(mc_particle_and_hits_nexus_data, config_tmpdir):
    hfile, name, vi, vf, p, Ep, nhits, X, Y, Z, E, t = mc_particle_and_hits_nexus_data

    PATH_IN   = hfile
    PATH_OUT  = os.path.join(config_tmpdir,'detsim_true_voxels.h5')
    conf      = configure('dummy invisible_cities/config/detsim.conf'.split())
    nevt_req  = 1

    conf.update(dict(files_in              = PATH_IN,
                     file_out              = PATH_OUT,
                     event_range           = (nevt_req,),
                     zmin                  = 50,
                     zmax                  = 500,
                     diff_transv           = 1.0,
                     diff_long             = 0.3,
                     resolution_FWHM       = 0.8,
                     Qbb                   = 2.45784,
                     true_voxel_dimensions = [5,5,5]))

    cdetsim = Detsim(**conf)
    cdetsim.run()
    cnt         = cdetsim.end()
    assert cnt.n_events_tot      == nevt_req
