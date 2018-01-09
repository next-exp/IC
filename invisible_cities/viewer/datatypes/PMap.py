import numpy

from . DataBase import RecoBase3D
from pyqtgraph  import opengl as gl

class PMap(RecoBase3D):

    """Class for drawing PMaps
    
    Computes active voxels using S1 and S2si to draw pmaps
    """

    def __init__(self):
        super(PMap, self).__init__()
        self._product_name = 'PMap'
        self._gl_voxel_mesh = None
        # X/Y/Z/Val information is cached for fast redrawing
        self._x = None
        self._y = None
        self._z = None
        self._vals = None
        self._meta = None

        # A template of corners defining vertices of a box
        self._box_template = numpy.array([[ 0 , 0, 0],
                                          [ 1 , 0, 0],
                                          [ 1 , 1, 0],
                                          [ 0 , 1, 0],
                                          [ 0 , 0, 1],
                                          [ 1 , 0, 1],
                                          [ 1 , 1, 1],
                                          [ 0 , 1, 1]],
                                         dtype=float)

        # A template of the 12 triangles (in reference to the above 
        # vertexes) needed to draw the faces of a cube
        self._faces_template = numpy.array([[0, 1, 2],
                                            [0, 2, 3],
                                            [0, 1, 4],
                                            [1, 5, 4],
                                            [1, 2, 5],
                                            [2, 5, 6],
                                            [2, 3, 6],
                                            [3, 6, 7],
                                            [0, 3, 7],
                                            [0, 4, 7],
                                            [4, 5, 7],
                                            [5, 6, 7]])

    # this is the function that actually draws the cluster.
    def draw_objects(self, view_manager, io, meta):
        """Get pmaps from the data and draw on the screen
        
        Fetches S1 and S2 and draws the values on the screen.
        
        Arguments:
            view_manager {ViewManager3D} -- The view manager
            io {IOManager} -- Instance of IOManager
            meta {EventMeta} -- Instance of EventMeta
        """


        # Store a reference to the meta object:
        self._meta = meta

        # Z position has to be calculated based on the difference in time between s1 ands s2

        # Get the data from the file

        # Timing of the event
        t0 = 1e-3*io.s1().peaks[0].tpeak

        # Placeholders for x/y/z
        self._x = []
        self._y = []
        self._z = []
        self._vals = []

        i = 0
        for i_peak in range(io.s2si().number_of_peaks):
            for sipm in io.s2si().sipms_in_peak(i_peak):
                wfm = io.s2si().sipm_waveform(i_peak,sipm)
                # Fill the variables as needed:
                for t, e in zip(wfm.t, wfm.E):
                    if e != 0:
                        self._x.append(meta.sipm_data().X[sipm])
                        self._y.append(meta.sipm_data().Y[sipm])
                        self._z.append((1e-3*t - t0))
                        self._vals.append(e)

            i += 1

        self._min_coords = numpy.asarray((numpy.min(self._x), 
                                         numpy.min(self._y), 
                                         numpy.min(self._z)))
        self._max_coords = numpy.asarray((numpy.max(self._x), 
                                         numpy.max(self._y), 
                                         numpy.max(self._z)))

        self.redraw(view_manager)


    def redraw(self, view_manager):
        """Redraw the objects on the screen
        
        Take the cached data and render it
        
        Arguments:
            view_manager {ViewManager3D} -- view manager
        """

        if self._gl_voxel_mesh is not None:
            view_manager.getView().removeItem(self._gl_voxel_mesh)
            self._gl_voxel_mesh = None


        verts, faces, colors = self.build_triangle_array(view_manager)


        #make a mesh item: 
        mesh = gl.GLMeshItem(vertexes=verts,
                             faces=faces,
                             faceColors=colors,
                             smooth=False)

        mesh.setGLOptions("translucent")        
        self._gl_voxel_mesh = mesh
        view_manager.get_view().addItem(self._gl_voxel_mesh)


    def build_triangle_array(self, view_manager):
        """Build an array in the proper format for a gl mesh
        
        Each x/y/z/value point creates 8 vertexes, 12 faces, and 12 face colors
        
        Arguments:
            view_manager {ViewManager3D} -- the view manager
        
        Returns:
            list -- vertexs, faces, and colors all as numpy ndarray
        """
        verts = None
        faces = None
        colors = None


        i = 0
        for x, y, z, val in zip(self._x, self._y, self._z, self._vals):

            # Don't draw this pixel if it's below the threshold:
            if val < view_manager.get_levels()[0]:
                continue


            this_color = self.get_color(view_manager.get_lookup_table(),
                                        view_manager.get_levels(),
                                        val)

            if colors is None:
                colors = numpy.asarray([this_color]*12)
            else:
                colors = numpy.append(colors,
                                      numpy.asarray([this_color]*12),
                                      axis=0)

            # print "({}, {}, {})".format(_pos[0], _pos[1], _pos[2])
            this_verts = self.make_box(x, y, z)

            if faces is None:
                faces = self._faces_template
            else:
                faces = numpy.append(faces, 
                                     self._faces_template + 8*i, 
                                     axis=0)
            if verts is None:
                verts = this_verts
            else:
                verts = numpy.append(verts, 
                                     this_verts, axis=0)

            i += 1

        return verts, faces, colors

    def make_box(self, x, y, z):
        """Build the correct box for X/Y/Z given the meta
        
        Since the meta is voxelized, the X/Y/Z location needs to 
        be mapped to the corresponding voxel.  This function makes 
        that mapping and builds the 8 coordinate box correspondingly
        
        Arguments:
            x {float} -- x location
            y {float} -- y location
            z {float} -- z location
        
        Returns:
            [type] -- [description]
        """
        verts_box = numpy.copy(self._box_template)
        #Scale all the points of the box to the right voxel size:
        verts_box[:,0] *= self._meta.size_voxel_x()
        verts_box[:,1] *= self._meta.size_voxel_y()
        verts_box[:,2] *= self._meta.size_voxel_z()

        #Shift the points to put the center of the cube at (0,0,0)
        verts_box[:,0] -= 0.5*self._meta.size_voxel_x()
        verts_box[:,1] -= 0.5*self._meta.size_voxel_y()
        verts_box[:,2] -= 0.5*self._meta.size_voxel_z()
        
        #Move the points to the right coordinate in this space
        verts_box[:,0] += x
        verts_box[:,1] += y
        verts_box[:,2] += z


        return verts_box


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
        elif value < lmin:
            return (0,0,0,0)
        else:
            # Map this value to the closest in the lookup table (255 items)
            index = 255*(value - lmin) / (lmax - lmin)
            return lookupTable[int(index)]


    def clear_drawn_objects(self, view_manager):
        """Override clear drawn objects
        
        Remove objects from view, and delete the local cache of data
        
        Arguments:
            view_manager {ViewManager3D} -- The view manager
        """

        if self._gl_voxel_mesh is not None:
            view_manager.get_view().removeItem(self._gl_voxel_mesh)

        self._gl_voxel_mesh = None
        self._points = None
        self._vals = None
        self._colors = None
        self._x = []
        self._y = []
        self._z = []
        self._vals = []

    def refresh(self, view_manager):
        self.redraw(view_manager)