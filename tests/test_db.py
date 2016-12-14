import tables
from Database import download
import Database.loadDB as DB
import os


from pytest import fixture, mark

@fixture
def loadDB():
    download.loadDB()
    return os.environ['ICDIR'] + '/Database/localdb.h5'

@mark.slow
def test_numberOfPMTs(loadDB):
    """
    Check that we retrieve the correct number of PMTs
    """
    pmts = DB.DataPMT()
    assert pmts.shape[0] == 12

@mark.slow
def test_numberOfSiPMs(loadDB):
    """
    Check that we retrieve the correct number of SiPMs
    """
    sipms = DB.DataSiPM()
    assert sipms.shape[0] == 1792

@mark.slow
def test_SiPMNoise(loadDB):
    """
    Check we have noise for all SiPMs and energy of each bin
    """
    noise, energy, baseline = DB.SiPMNoise()
    assert noise.shape[0] == baseline.shape[0]
    assert noise.shape[0] == 1792
    assert noise.shape[1] == energy.shape[0]

@mark.slow
def test_DetectorGeometry(loadDB):
    """
    Check Detector Geometry
    """
    geo = DB.DetectorGeo()
    assert geo['XMIN'][0] == -198
    assert geo['XMAX'][0] ==  198
    assert geo['YMIN'][0] == -198
    assert geo['YMAX'][0] ==  198
    assert geo['ZMIN'][0] ==    0
    assert geo['ZMAX'][0] ==  532
    assert geo['RMAX'][0] ==  198
