from pyqtgraph.Qt import QtCore

from . EventMeta import NEWMeta
from . IOManager import IOManager

class EvdManagerBase(QtCore.QObject):
    """Basic manager class for interfacing the GUI and IOManager
    
    This class works between the GUI and the IOManager.  It inherits
    from QObject to allow it to send and receive signals, so that click
    buttons and such from the GUI can cause it to act.
    
    Extends:
        QtCore.QObject
    
    Variables:
        eventChanged {pyqtSignal} -- signal to announce the event has changed
        _io_manager {IOManager} -- instance of IOManager
        _meta {EventMeta} -- instance of EventMeta
        _drawnClasses {dict} -- dictionary of classes that have been triggered to draw
    """
    

    eventChanged = QtCore.pyqtSignal()


    def __init__(self, config, _file=None):
        super(EvdManagerBase, self).__init__()
        QtCore.QObject.__init__(self)

    def init_manager(self, _file):
        """Initialize the manager class
        
        Initializes the IOManager, goes to the first entry, and 
        creates an instance of the correct meta information
        
        Arguments:
            _file {[type]} -- [description]
        """

        # Initialize the IOManager with the requested file
        self._io_manager = IOManager()
        self._io_manager.set_file(_file)

        self.go_to_entry(0)

        # # Meta keeps track of information about number of planes, visible
        # # regions, etc.:
        self._meta = NEWMeta()


        # Drawn classes is a list of things getting drawn, as well.
        self._drawnClasses = dict()



    def meta(self):
        """Access the meta information

        Returns:
            NEWMeta -- NEWMeta object initialized from the database
        """
        return self._meta

    def io(self):
        """Give access to the io manager to query for run, event, etc
        
        Returns the IOManager for use in updating the GUI, etc
        """
        return self._io_manager

    def n_entries(self):
        if self._io_manager is not None:
            return self._io_manager.get_num_events()
        else:
            return 0


    def next(self):
        """Quick function to iterate to the next entry and redraw
        
        """
        if self.io().entry() + 1 < self.n_entries():
            self.go_to_entry(self.io().entry() + 1)
        else:
            print("On the last event, can't go to next.")

    def prev(self):
        """Quick function to iterate to the previous entry and redraw
        
        """
        if self.io().entry != 0:
            self.go_to_entry(self.io().entry() - 1)
        else:
            print("On the first event, can't go to previous.")

    def go_to_entry(self, entry):
        """Random access function
        
        This function makes sure of two things:
        1 - The io manager goes to the next event
        2 - Anyone who is listening (the GUI) knows the event changed 
        
        Arguments:
            entry {int} -- Requested entry
        """
        self._io_manager.go_to_entry(entry)
        self.eventChanged.emit()
