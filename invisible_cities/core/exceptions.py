"""
Define IC-specific exceptions
"""

class ICException(Exception):
    """ Base class for IC exceptions hierarchy """

class NoInputFiles(ICException):
    """ Input files list is not defined """

class InvalidInputFileStructure(ICException):
    """There is something wrong with the input file"""

class NoOutputFile(ICException):
    pass

class XYRecoFail(ICException):
    pass

class SipmEmptyList(XYRecoFail):
    pass

class SipmEmptyListAboveQthr(XYRecoFail):
    pass

class ClusterEmptyList(XYRecoFail):
    pass

class SipmZeroCharge(XYRecoFail):
    pass

class SipmZeroChargeAboveQthr(XYRecoFail):
    pass

class NoHits(ICException):
    pass

class NoVoxels(ICException):
    pass

class SensorBinningNotFound(ICException):
    pass

class NoParticleInfoInFile(ICException):
    pass

class TableMismatch(ICException):
    pass

class TimeEvolutionTableMissing(ICException):
    pass

class MCEventNotFound(ICException):
    """ Requested event missing from input file """

class SensorIDMismatch(ICException):
    pass
