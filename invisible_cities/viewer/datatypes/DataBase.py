# from ROOT import evd
import numpy

# These classes provide the basic interfaces for drawing objects
# It's meant to be inherited from for specific instances

class DataBase(object):

    """Basic class from which data objects inherit
    
    This class exists on it's own to leave open the possibility for non 3D drawing
    """

    def __init__(self):
        super(DataBase, self).__init__()
        self._product_name = "null"

    def product_name(self):
        return self._product_name


class RecoBase3D(DataBase):

    """Core class for 3D object drawing
    
    Contains shells for main functions to be overloaded:
     - clear_drawn_objects
     - get_drawn_objects
     - draw_objects
     - refresh
     - min
     - max
    """

    def __init__(self):
        super(RecoBase3D, self).__init__()
        self._drawnObjects = []
        self._min_coords = numpy.asarray((0,0,0))
        self._max_coords = numpy.asarray((0,0,0))

    def min(self):
        """Get the minimum (x,y,z) location
        
        Returns:
            ndarray -- min coordinates, shape is (3,)
        """
        return self._min_coords
    
    def max(self):
        """Get the maximum (x,y,z) location
        
        Returns:
            ndarray -- max coordinates, shape is (3,)
        """
        return self._max_coords

    def clear_drawn_objects(self, view_manager):
        """Clear all drawn objects
        
        Remove all objects this class draws from the view.
        Unless the class is doing something special, it's unlikely
        to be necessary to overload this
        
        Arguments:
            view_manager {ViewManager3D} -- The view manager
        """
        view = view_manager.get_view()
        for item in self._drawnObjects:
            view.removeItem(item)
        # clear the list:
        self._drawnObjects = []

    def get_drawn_objects(self):
        """Return the list of drawn objects
        
        Returns:
            list -- All objects drawn in the view
        """
        return self._drawnObjects

    def draw_objects(self, view_manager):
        """ "Virtual" function for drawing objects.
        
        Function to draw objects on the view_manager.  Raises 
        NotImplementedError if called directly - instead, override in the base class
        
        Arguments:
            view_manager {ViewManager3D} -- The view manager
        """
        raise NotImplementedError("draw_objects can not be called directly from recoBase3D")

    def refresh(self, view_manager):
        """ "Virtual" function for drawing objects.
        
        Function to redraw the objects on the screen
        If objects are not meant to be refreshed, this function does not need
        to be overridden

        Arguments:
            view_manager {ViewManager3D} -- The view manager
        """
        pass
