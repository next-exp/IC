# This is the class that maintains the list of drawable items.
# If your class isn't here, it can't be drawn
import collections

from . MCHit   import MCHit
from . MCTrack import MCTrack
from . PMap    import PMap

class DrawableItems3D(object):

    """This class exists to enumerate the drawableItems in 3D"""
    # If you make a new drawing class, add it here

    def __init__(self):
        super(DrawableItems3D, self).__init__()
        # items are stored as pointers to the classes (not instances)
        self._drawableClasses = collections.OrderedDict()
        self._drawableClasses.update({'MCHits' : MCHit})
        self._drawableClasses.update({'MCTracks' : MCTrack})
        self._drawableClasses.update({'PMaps' : PMap})

    def get_list_of_titles(self):
        return self._drawableClasses.keys()

    def get_list_of_items(self):
        return zip(*self._drawableClasses.values())[1]

    def get_dict(self):
        return self._drawableClasses


