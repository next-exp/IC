"""
Define IC-specific exceptions
"""

class ICException(Exception):
    """ Base class for IC exceptions hierarchy """
    pass

class NoInputFiles(ICException):
    """ Input files list is not defined """
    pass

class NoOutputFile(ICException):
    pass

class ParameterNotSet(ICException):
    pass
