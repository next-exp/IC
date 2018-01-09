import os

from invisible_cities.io import pmap_io, mchits_io

class IOManager(object):
    """wrapper to IC event interface to allow random access through events
    
    IC doesn't implicitly allow a random access event loop.  This class
    reads an entire file into memory, and then stores the events to allow
    the event viewer to access them randomly.
    """
    def __init__(self):
        super(io_manager, self).__init__()

        # List of event numbers
        self._events = []
        # Current entry in the above list
        self._entry = 0
        # Highest entry available
        self._max_entry = 0
        # Current run (informational)
        self._run = 0
        # Current subrun (informational)
        self._subrun = 0

    def event(self):
        """Get the data from the current event
        
        Returns:
            [type] -- [description]
        """
        return self._events[self._entry]

    def entry(self):
        """Get the currently accessed entry
        
        Returns:
            int -- the active entry
        """
        return self._entry

    def run(self):
        """Get the run number of the current entry
        
        Returns:
            int -- the current run number
        """
        return self._run

    def subrun(self):
        """Get the subrun number of the current entry
        
        Returns:
            int -- the current subrun number
        """
        return self._subrun

    def set_file(self, file_name):
        """Open a new file and read it's data
        
        Read the pmaps from a new file.  Will attempt to read MC as well, though
        it will catch exceptions if any MC is missing.

        Does not yet read reconstructed information, this is a TODO
        
        Arguments:
            file_name {str} -- path to file to open
        """

        # Load the pmaps, catch exception by declaring the presence of pmaps as false
        try:
            self._s1_dict, self._s2_dict, self._s2si_dict = pmap_io.load_pmaps(file_name)
            self._has_pmaps = True
        except:
            self._has_pmaps = False

        # Load MC information, catch exception by declaring the presence of mc info as false
        try:
            self._mc_hits = mchits_io.load_mchits(file_name)
            self._mc_part = mchits_io.load_mcparticles(file_name)
            self._has_mc = True
        except:
            self._has_mc = False
            pass

        # Get the run and subrun information
        # TODO - is there a better way to do this???
        strs = os.path.basename(file_name).split("_")
        i = 0
        for s in strs:
            if s == "pmaps":
                break
            i += 1
        # There must be a way to get run and subrun information...
        self._subrun = int(strs[i+1])
        self._run = int(strs[i+2])

        self._has_reco = False
        if not (self._has_reco or self._has_pmaps or self._has_mc):
            print("Couldn't load file {}.".format(file_name))
            exit(-1)

        # Use the S2_dict as the list of events.
        # This explicitly requires that events have both s2 and s2si available.
        self._events = list(set(self._s1_dict.keys()).intersection(set(self._s2si_dict.keys())))
        # Store the highest available entry to access
        self._max_entry = len(self._events) -1

    def s1(self, event=-1):
        """Return s2 information for an event
        
        If event is specified explicitly, check event is available and return that s2
        Otherwise, return s2 for currently active event
        
        Keyword Arguments:
            event {number} -- event number (default: {-1})
        
        Returns:
            S1 -- S1 object of PMap
        """
        if not self._has_pmaps:
            return None
        if event == -1:
            event = self._events[self._entry]
        if event not in self._events:
            print("Can't go to event {}".format(event))
        return self._s1_dict[event]

    def s2(self, event=-1):
        """Return s2 information for an event
        
        If event is specified explicitly, check event is available and return that s2
        Otherwise, return s2 for currently active event
        
        Keyword Arguments:
            event {number} -- event number (default: {-1})
        
        Returns:
            S2 -- S2 object of PMap
        """
        if not self._has_pmaps:
            return None
        if event == -1:
            event = self._events[self._entry]
        if event not in self._events:
            print("Can't go to event {}".format(event))
        return self._s2_dict[event]

    def s2si(self, event=-1):
        """Return s2si information for an event
        
        If event is specified explicitly, check event is available and return that s2si
        Otherwise, return s2si for currently active event
        
        Keyword Arguments:
            event {number} -- event number (default: {-1})
        
        Returns:
            s2si -- s2si object of PMap
        """
        if not self._has_pmaps:
            return None
        if event == -1:
            event = self._events[self._entry]
        if event not in self._events:
            print("Can't go to event {}".format(event))
        return self._s2si_dict[event]

    def mchits(self, event=-1):
        """Return mchit objects
        
        If event is specified explicitly, check event is available and return that mchits
        Otherwise, return mchis for currently active event
                
        Keyword Arguments:
            event {number} -- [description] (default: {-1})
        
        Returns:
            mchits -- MCHits object
        """
        if not self._has_mc:
            return None
        if event == -1:
            event = self._events[self._entry]
        if event not in self._events:
            print("Can't go to event {}".format(event))
        return self._mc_hits[event]

    def mctracks(self, event=-1):
        """Return mctrack objects
        
        If event is specified explicitly, check event is available and return that mctrack
        Otherwise, return mctrack for currently active event
                
        Keyword Arguments:
            event {number} -- [description] (default: {-1})
        
        Returns:
            mctrack -- mctrack object
        """
        if not self._has_mc:
            return None
        if event == -1:
            event = self._events[self._entry]
        if event not in self._events:
            print("Can't go to event {}".format(event))
        return self._mc_part[event]        

    def get_num_events(self):
        """Query for the total number of events in this file
        
        Returns:
            int -- Total number of events
        """
        return len(self._events)

    def go_to_entry(self,entry):
        """Move the current index to the specified entry
        
        Move the access point to the entry specified.  Does checks to 
        verify the entry is available.
        
        Arguments:
            entry {int} -- Desired entry
        """
        if entry >= 0 and entry < self.get_num_events():
            self._entry = entry
        else:
            print("Can't go to entry {}, entry is out of range.".format(entry))


