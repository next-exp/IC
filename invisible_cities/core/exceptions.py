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

class SipmEmptyList(ICException):
    pass

class SipmNotFound(ICException):
    pass

class SipmZeroCharge(ICException):
    pass

class NoHits(ICException):
    pass

class NoVoxels(ICException):
    pass

class InconsistentS12dPmtsd(ICException):
    pass
