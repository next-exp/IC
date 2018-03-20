from . detsim            import Detsim

def test_detsim_SE(SE_nexus_filename, config_tmpdir):
    PATH_IN   = SE_nexus_filename
    PATH_OUT  = os.path.join(config_tmpdir,'SE_true_voxels.h5')
    conf      = configure('dummy invisible_cities/config/detsim.conf'.split())
    nevt_req  = 1

    conf.update(dict(files_in      = PATH_IN,
                     file_out      = PATH_OUT,
                     event_range   = (nevt_req,),
                     **KrMC_hdst.config))

    cdetsim = Detsim(**conf)
    cdetsim.run()
    cnt         = cdetsim.end()
    assert cnt.n_events_tot      == nevt_req

    #df_penthesilea = dio.load_dst(PATH_OUT , 'RECO', 'Events')
    #assert_dataframes_close(df_penthesilea, DF_TRUE, check_types=False)
