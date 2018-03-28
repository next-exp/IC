from . import load_db as DB

import sqlite3
import numpy as np
from pytest  import fixture
from os.path import join

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
    columns = ['SensorID', 'ChannelID', 'Active', 'X', 'Y', 'adc_to_pes', 'Sigma']
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


def test_mc_runs_equal_data_runs():
    assert (DB.DataPMT (-3550).values == DB.DataPMT (3550).values).all()
    assert (DB.DataSiPM(-3550).values == DB.DataSiPM(3550).values).all()


@fixture(scope='module')
def test_db(tmpdir_factory):
    temp_dir = tmpdir_factory.mktemp('output_files')
    dbfile = join(temp_dir, 'db.sqlite3')
    connSql3 = sqlite3.connect(dbfile)
    cursorSql3 = connSql3.cursor()

    cursorSql3.execute('''CREATE TABLE IF NOT EXISTS `SipmBaseline` (
`MinRun` integer NOT NULL
,  `MaxRun` integer DEFAULT NULL
,  `SensorID` integer NOT NULL
,  `Energy` float NOT NULL
);''')

    cursorSql3.execute('''CREATE TABLE IF NOT EXISTS `SipmNoisePDF` (
    `MinRun` integer NOT NULL
    ,  `MaxRun` integer DEFAULT NULL
    ,  `SensorID` integer NOT NULL
    ,  `BinEnergyPes` float NOT NULL
    ,  `Probability` float NOT NULL
);''')

    #Insert sample data
    sql = 'INSERT INTO SipmBaseline (MinRun, MaxRun, SensorID, Energy) VALUES ({})'
    cursorSql3.execute(sql.format('0,NULL,1,0'))
    sql = 'INSERT INTO SipmNoisePDF (MinRun, MaxRun, SensorID, BinEnergyPes, Probability) VALUES ({})'
    cursorSql3.execute(sql.format('0,NULL,1,5,0.1'))
    cursorSql3.execute(sql.format('0,NULL,1,3,0.3'))
    cursorSql3.execute(sql.format('0,NULL,1,4,0.2'))
    cursorSql3.execute(sql.format('0,NULL,1,1,0.5'))
    cursorSql3.execute(sql.format('0,NULL,1,2,0.4'))
    connSql3.commit()
    connSql3.close()

    noise_true     = np.array([[ 0.5,  0.4,  0.3,  0.2,  0.1]])
    bins_true      = np.array([ 1.,  2.,  3.,  4.,  5.])
    baselines_true = np.array([ 0.])
    sipm_noise = noise_true, bins_true, baselines_true

    return dbfile, sipm_noise

def test_sipm_noise_order(test_db):
    #Read from DB
    dbfile = test_db[0]
    noise, bins, baselines = DB.SiPMNoise(1, dbfile)

    #'True' values
    sipm_noise     = test_db[1]
    noise_true     = sipm_noise[0]
    bins_true      = sipm_noise[1]
    baselines_true = sipm_noise[2]

    np.testing.assert_allclose(noise,     noise_true)
    np.testing.assert_allclose(bins,      bins_true)
    np.testing.assert_allclose(baselines, baselines_true)
