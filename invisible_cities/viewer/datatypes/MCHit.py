import numpy

from pyqtgraph    import opengl as gl
from . DataBase   import recoBase3D

class MCHit(RecoBase3D):

    """Class for drawing MC Hits
    
    Draws depositions of charge (mchit) as white spheres on the view
    """

    def __init__(self):
        super(mchit, self).__init__()
        self._product_name = 'mchit'
        # OpenGL Object to draw:
        self._gl_points_collection = None
        # List of points to draw (cached for refreshing)
        self._points = None
        # List of values to draw (cached for refreshing)
        self._vals = None


    # this is the function that actually draws the cluster.
    def draw_objects(self, view_manager, io, meta):
        """Override draw_objects for mchits
        
        Gather the MCHits from the io, and call a worker function to put them 
        on the screen.
        
        Arguments:
            view_manager {ViewManager3D} -- The view manager
            io {IOManager} -- Instance of IOManager
            meta {EventMeta} -- Instance of EventMeta
        """

        # Get the data from the file:
        mc_hits = io.mchits()



        self._points = numpy.ndarray((len(mc_hits),3))
        self._vals   = numpy.ndarray((len(mc_hits)))
        self._colors = numpy.ndarray((len(mc_hits),4))
        

        i = 0
        for hit in mc_hits:
            self._points[i][0] = mc_hits[i].X
            self._points[i][1] = mc_hits[i].Y
            self._points[i][2] = mc_hits[i].Z
            self._vals[i] = mc_hits[i].E

            i += 1

        
        self._min_coords = numpy.min(self._points, axis=0)
        self._max_coords = numpy.max(self._points, axis=0)

        self.redraw(view_manager)


    def redraw(self, view_manager):
        """This function actually puts the objects on the screen
        
        Take the numpy arrays from draw_objects, build the color scale fresh, and draw 
        the objects
        
        Arguments:
            view_manager {ViewManager3D} -- The view manager
        """

        if self._gl_points_collection is not None:
            view_manager.get_view().removeItem(self._gl_points_collection)
            self._gl_points_collection = None

        i = 0
        for val in self._vals:
            this_color = self.get_color(view_manager.get_lookup_table(),
                                        view_manager.get_levels(),
                                        val)
            self._colors[i] = this_color
            i += 1

        #make a mesh item: 
        mesh = gl.GLScatterPlotItem(pos=self._points,
                                    color=self._colors,
                                    size=1,
                                    pxMode=False)

        # mesh.setGLOptions("opaque")        
        self._gl_points_collection = mesh
        view_manager.get_view().addItem(self._gl_points_collection)

    def get_color(self, lookupTable, levels, value ):
        """Use the lookup table and levels to interpolate a color
        
        Finds the value of the lookup table that is closest to the 
        value specified.  Colors above threshold are set to the max
        value.  Below threhold is set to (0,0,0,0)
        
        Arguments:
            lookupTable {} -- [Color lookup table]
            levels {list} -- Min and max values of the table
            value {[type]} -- Value in question 
        
        Returns:
            [type] -- [description]
        """
        lmin = levels[0]
        lmax = levels[1]

        if value >= lmax:
            return lookupTable[-1]
        elif value < min:
            return (0,0,0,0)
        else:
            # Map this value to the closest in the lookup table (255 items)
            index = 255*(value - min) / (lmax - min)
            return lookupTable[int(index)]


    def clear_drawn_objects(self, view_manager):
        """Override clear drawn objects
        
        Remove objects from view, and delete the local cache of data
        
        Arguments:
            view_manager {ViewManager3D} -- The view manager
        """

        if self._gl_points_collection is not None:
            view_manager.getView().removeItem(self._gl_points_collection)

        self._gl_points_collection = None
        self._points = None
        self._vals = None
        self._colors = None

    def refresh(self, view_manager):
        self.redraw(view_manager)