from . import load_db as DB


def test_pmts_pd():
    """Check that we retrieve the correct number of PMTs."""
    pmts = DB.DataPMT()
    columns =['SensorID', 'ChannelID', 'PmtID', 'Active', 'X', 'Y',
              'coeff_blr', 'coeff_c', 'adc_to_pes', 'noise_rms', 'Sigma']
    assert columns == list(pmts)
    assert pmts['PmtID'].str.startswith('PMT').all()
    assert pmts.shape[0] == 12

def test_pmts_MC_pd():
    """Check that we retrieve the correct number of PMTs."""
    mc_run = 0
    pmts = DB.DataPMT(mc_run)
    columns =['SensorID', 'ChannelID', 'PmtID', 'Active', 'X', 'Y',
              'coeff_blr', 'coeff_c', 'adc_to_pes', 'noise_rms', 'Sigma']
    assert columns == list(pmts)
    assert pmts['PmtID'].str.startswith('PMT').all()
    assert pmts.shape[0] == 12

def test_sipm_pd():
    """Check that we retrieve the correct number of SiPMs."""
    sipms = DB.DataSiPM()
    columns = ['SensorID', 'ChannelID', 'Active', 'X', 'Y', 'adc_to_pes']
    assert columns == list(sipms)
    assert sipms.shape[0] == 1792

def test_SiPMNoise():
    """Check we have noise for all SiPMs and energy of each bin."""
    noise, energy, baseline = DB.SiPMNoise()
    assert noise.shape[0] == baseline.shape[0]
    assert noise.shape[0] == 1792
    assert noise.shape[1] == energy.shape[0]


def test_DetectorGeometry():
    """Check Detector Geometry."""
    geo = DB.DetectorGeo()
    assert geo['XMIN'][0] == -198
    assert geo['XMAX'][0] ==  198
    assert geo['YMIN'][0] == -198
    assert geo['YMAX'][0] ==  198
    assert geo['ZMIN'][0] ==    0
    assert geo['ZMAX'][0] ==  532
    assert geo['RMAX'][0] ==  198
