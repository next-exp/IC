import time
import sqlite3

from os.path import join

import numpy as np

from pytest  import fixture
from pytest  import mark

from . import load_db as DB

def test_pmts_pd(dbnew):
    """Check that we retrieve the correct number of PMTs."""
    pmts = DB.DataPMT(dbnew)
    columns =['SensorID', 'ChannelID', 'PmtID', 'Active', 'X', 'Y',
              'coeff_blr', 'coeff_c', 'adc_to_pes', 'noise_rms', 'Sigma']
    assert columns == list(pmts)
    assert pmts['PmtID'].str.startswith('PMT').all()
    assert pmts.shape[0] == 12


def test_pmts_MC_pd(dbnew):
    """Check that we retrieve the correct number of PMTs."""
    mc_run = 0
    pmts = DB.DataPMT(dbnew, mc_run)
    columns =['SensorID', 'ChannelID', 'PmtID', 'Active', 'X', 'Y',
              'coeff_blr', 'coeff_c', 'adc_to_pes', 'noise_rms', 'Sigma']
    assert columns == list(pmts)
    assert pmts['PmtID'].str.startswith('PMT').all()
    assert pmts.shape[0] == 12


def test_sipm_pd(dbnew):
    """Check that we retrieve the correct number of SiPMs."""
    sipms = DB.DataSiPM(dbnew)
    columns = ['SensorID', 'ChannelID', 'Active', 'X', 'Y', 'adc_to_pes', 'Sigma']
    assert columns == list(sipms)
    assert sipms.shape[0] == 1792


def test_SiPMNoise(dbnew):
    """Check we have noise for all SiPMs and energy of each bin."""
    noise, energy, baseline = DB.SiPMNoise(dbnew)
    assert noise.shape[0] == baseline.shape[0]
    assert noise.shape[0] == 1792
    assert noise.shape[1] == energy.shape[0]


def test_DetectorGeometry(dbnew):
    """Check Detector Geometry."""
    geo = DB.DetectorGeo(dbnew)
    assert geo['XMIN'][0] == -198
    assert geo['XMAX'][0] ==  198
    assert geo['YMIN'][0] == -198
    assert geo['YMAX'][0] ==  198
    assert geo['ZMIN'][0] ==    0
    assert geo['ZMAX'][0] ==  532
    assert geo['RMAX'][0] ==  198


def test_mc_runs_equal_data_runs(dbnew):
    assert (DB.DataPMT (dbnew, -3550).values == DB.DataPMT (dbnew, 3550).values).all()
    assert (DB.DataSiPM(dbnew, -3550).values == DB.DataSiPM(dbnew, 3550).values).all()


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
    noise, bins, baselines = DB.SiPMNoise(dbfile, 1)

    #'True' values
    sipm_noise     = test_db[1]
    noise_true     = sipm_noise[0]
    bins_true      = sipm_noise[1]
    baselines_true = sipm_noise[2]

    np.testing.assert_allclose(noise,     noise_true)
    np.testing.assert_allclose(bins,      bins_true)
    np.testing.assert_allclose(baselines, baselines_true)


@mark.parametrize("db_fun", (DB.DataPMT, DB.DataSiPM, DB.SiPMNoise))
def test_database_is_being_cached(db_fun, dbnew):
    run_number = 3333 # a value not used by any other test

    t0 = time.time()
    first_call  = db_fun(dbnew, run_number)
    t1 = time.time()
    second_call = db_fun(dbnew, run_number)
    t2 = time.time()

    time_first_call  = t1 - t0
    time_second_call = t2 - t1

    if db_fun is DB.SiPMNoise:
        for item_first, item_second in zip(first_call, second_call):
            assert np.allclose(item_first, item_second)
    else:
        assert np.all(first_call.values == second_call.values)
    assert time_second_call < 1e6 * time_first_call


def test_frontend_mapping(dbnew):
    """ Check the mapping has the expected shape etc """

    fe_mapping, _ = DB.PMTLowFrequencyNoise(dbnew)

    columns = ['SensorID', 'FEBox']

    assert columns == list(fe_mapping)
    assert fe_mapping.SensorID.nunique() == 12
    assert fe_mapping.FEBox.nunique()    == 3


def test_pmt_noise_frequencies(dbnew):
    """ Check the magnitudes and frequencies
    are of the expected length """
    _, frequencies = DB.PMTLowFrequencyNoise(dbnew)

    ## Currently simulate frequencies in range(312.5, 25000) Hz
    freq_expected = np.arange(1, 80) * 312.5
    ## Expected four columns: frequency, mag FE0, mag FE1, mag FE2
    assert frequencies.shape[0] == 79
    assert frequencies.shape[1] == 4
    assert np.all(frequencies[:, 0] == freq_expected)
