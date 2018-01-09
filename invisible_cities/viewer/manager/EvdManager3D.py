from pyqtgraph.Qt import QtCore
import numpy

import datatypes
from . EvdManagerBase import EvdManagerBase
from . EventMeta      import NEWMeta


class EvdManager3D(EvdManagerBase):
    """Extension of general manager for specific 3D case
    
    This class handles file I/O and drawing for 3D viewer
    Many methods are inherited from EvdManagerBase.  Here, 
    specific methods for managing drawing and autozoom are maintained.
    
    Extends:
        EvdManagerBase
    """


    def __init__(self, _file=None):
        super(EvdManager3D, self).__init__(_file)

        # This dictionary is the list of items that the viewer knows how to draw
        self._drawableItems = datatypes.drawableItems3D()
        
        # The manager initialization ofo the base class does not call init_manager
        self.init_manager(_file[0])

    # this function is meant for the first request to draw an object or
    # when the producer changes
    def redraw_product(self, product, view_manager, draw):
        """Draw, or redraw, a product
        
        Handles the calling of a class to draw a product.  Product is specified
        by the caller, but is a string.  view_manager is an instance of ViewManager3D,
        which is passed from the GUI.

        Draw can be True or False - True draws (or redraws) while False clears the object
        
        Arguments:
            product {str} -- Product to draw ("PMaps" for example)
            view_manager {ViewManager3D} -- Instance of graphics manager class
            draw {bool} -- True to ensure product is draw, false to ensure it's not
        """
        
        # If requesting to not draw this product, remove it:
        if draw is False:
            # Make sure it's already draw before trying to clear:
            if product in self._drawnClasses:
                self._drawnClasses[product].clear_drawn_objects(view_manager)
                # Remove it from the list of draw classes
                self._drawnClasses.pop(product)
            return

        # Determine if there is already a drawing process for this product:          
        if product in self._drawnClasses:
            # If so, clear it and redraw
            self._drawnClasses[product].clear_drawn_objects(view_manager)
            self._drawnClasses[product].draw_objects(view_manager, self._io_manager, self.meta())
            return

        # Otherwise, drawing for the first time.
        # Make sure this is actually drawable:
        if product in self._drawableItems.get_list_of_titles():
            # drawable items contains a reference to the class, so instantiate it
            drawingClass=self._drawableItems.get_dict()[product]()

            # Add this instance to the list of drawnclasses
            self._drawnClasses.update({product: drawingClass})

            # Draw this object:
            drawingClass.draw_objects(view_manager, self._io_manager, self.meta())

    # def getProducers(self, product):
    #     return ["mc"]

    def clear_all(self, view_manager):
        """Remove all objects from view
        
        Clear all known objects from the view, but maintains the list of classes to draw
        
        Arguments:
            view_manager {ViewManager3D} -- instance of a ViewManager3D
        """
        for recoProduct in self._drawnClasses:
            self._drawnClasses[recoProduct].clear_drawn_objects(view_manager)

    def draw_fresh(self, view_manager):
        """Redraw everything in the view
        
        Clear every object from the view, and then redraw them in order.
        Order is determined by DrawableItems
        
        Arguments:
            view_manager {ViewManager3D} -- instance of a ViewManager3D
        """
        # Clear everything:
        self.clear_all(view_manager)
        # Draw objects in a specific order defined by drawableItems
        order = self._drawableItems.get_list_of_titles()

        for item in order:
            # If the item is requested to draw, draw it:
            if item in self._drawnClasses:
                self._drawnClasses[item].draw_objects(view_manager, self._io_manager, self.meta())


    def refresh_colors(self, view_manager):
        """Update color scale on the fly
        
        Call the refresh function for each class that is actively drawn
        
        Arguments:
            view_manager {ViewManager3D} -- instance of a ViewManager3D
        """
        # As in draw_fresh, do this in order
        order = self._drawableItems.get_list_of_titles()
        for item in order:
            if item in self._drawnClasses:
                self._drawnClasses[item].refresh(view_manager)

    def get_min_max_coords(self):
        """Get the min/max coordinates of drawn items
        
        Loop over drawn classes and ask each class for the min and max values

        If no classes are drawn, use the detector min/max
        
        Returns:
            list -- numpy array of mins (x,y,z) and numpy array of maxes (x,y,z)
        """
        if len(self._drawnClasses) == 0:
            return [numpy.asarray([self.meta().min_x(), 
                     self.meta().min_y(),
                     self.meta().min_z()]),
                    numpy.asarray([self.meta().max_x(), 
                     self.meta().max_y(),
                     self.meta().max_z()])]
        else:
            mins = []
            maxs = []
            for name, _cls in self._drawnClasses.items():
                mins.append(_cls.min())
                maxs.append(_cls.max())

            return numpy.min(mins, axis=0), numpy.max(maxs, axis=0)
