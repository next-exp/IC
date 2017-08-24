"""
Define IC-specific exceptions
"""

class ICException(Exception):
    """ Base class for IC exceptions hierarchy """
    pass

class NoInputFiles(ICException):
    """ Input files list is not defined """
    pass

class FileLoopMethodNotSet(ICException):
    """ File loop method no defined by cities"""
    pass

class EventLoopMethodNotSet(ICException):
    """ Event loop method no defined by cities"""
    pass

class NoOutputFile(ICException):
    pass

class UnknownRWF(ICException):
    pass

class ParameterNotSet(ICException):
    pass

class PeakNotFound(ICException):
    pass

class XYRecoFail(ICException):
    pass

class SipmEmptyList(XYRecoFail):
    pass

class SipmEmptyListAboveQthr(XYRecoFail):
    pass

class ClusterEmptyList(XYRecoFail):
    pass

class SipmNotFound(XYRecoFail):
    pass

class SipmZeroCharge(XYRecoFail):
    pass

class SipmZeroChargeAboveQthr(XYRecoFail):
    pass

class NoHits(ICException):
    pass

class NoVoxels(ICException):
    pass

class InconsistentS12dPmtsd(ICException):
    pass
