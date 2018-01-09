import math
import numpy

from pyqtgraph import opengl as gl
from pyqtgraph.Qt import QtGui, QtCore

class Viewport3D(gl.GLViewWidget):
    """Class to wrap a widget that can display open GL objects
    
    Inherit the GLViewWidget to let this view draw open GL objects,
    but also provide functions to easily do things requested by the GUI
    
    Extends:
        gl.GLViewWidget
    
    Variables:
        quitRequested {pyqtsignal} -- Signal to say a quit was requested
        keyPressSignal {pyqtsignal} -- Signal to echo to main gui key presses
        viewChanged {pyqtsignal} -- Signal to tell main gui the view changed

    """
    
    

    quitRequested = QtCore.pyqtSignal()
    keyPressSignal = QtCore.pyqtSignal(QtGui.QKeyEvent)
    viewChanged = QtCore.pyqtSignal()


    def __init__(self):
        super(Viewport3D, self).__init__()
        # add a view box, which is a widget that allows an image to be shown
        # add an image item which handles drawing (and refreshing) the image
        self.setBackgroundColor((0,0,0,255))
        # self.setBackgroundColor((255,255,255,0))

        # Dims stores the size in x/y/z
        self._dims = None

        # Background items are gl items that are drawn but don't change between events
        # (like detector renderings)
        self._background_items = []
        
        # Show this widget
        self.show()

    def draw_detector(self, meta):
        """Create detector background items
        
        Render the items like PMTs, SIPMs, and outer boundary
        
        Arguments:
            meta {EventMeta} -- Geometry information 
        """
        len_x = meta.len_x()
        len_y = meta.len_y()
        len_z = meta.len_z()
        self._dims = [len_x, len_y, len_z]

        # # Draw a cylinder for the detector:
        cylinderPoints = gl.MeshData.cylinder(2, 10, 
            radius=[1.5*meta.radius(), 1.5*meta.radius()], 
            length=meta.len_z())
        cylinder = gl.GLMeshItem(meshdata=cylinderPoints,
                                 drawEdges=True,
                                 drawFaces=False,
                                 smooth=False,
                                 glOptions='translucent')
        self.addItem(cylinder)

        # Draw locations of all sipms and PMTs

        # Using points to draw pmts and sipms, since a scatter plot is
        # easy and cheap to draw
        n_pmts = len(meta.pmt_data().index)

        pmt_pts = numpy.ndarray((n_pmts, 3))
        pmt_pts[:,0] = meta.pmt_data().X
        pmt_pts[:,1] = meta.pmt_data().Y
        pmt_pts[:,2] = meta.max_z()
        
        pmtPointsCollection = gl.GLScatterPlotItem(pos=pmt_pts,
                                                   size=20.32,
                                                   color=[0,0,1.0,1.0], 
                                                   pxMode=False)
        self.addItem(pmtPointsCollection)

        sipm_pts = numpy.ndarray((len(meta.sipm_data().index), 3))
        sipm_pts[:,0] = meta.sipm_data().X
        sipm_pts[:,1] = meta.sipm_data().Y
        sipm_pts[:,2] = meta.min_z()
        sipmPointsCollection = gl.GLScatterPlotItem(pos=sipm_pts,
                                                    size=2,
                                                    color=[0,1.0,1.0,1.0],
                                                    pxMode=False)
        self.addItem(sipmPointsCollection)

        # Move the center to the middle of the detector
        self.setCenter((0,0,0.5*len_z))
        

    # def updateMeta(self,meta):



    #     self.setCenter((0,0,0))
    
    #     for item in self._background_items:
    #         self.removeItem(item)
    #         self._background_items = []
    
    #     # This section prepares the 3D environment:
    #     # Add an axis orientation item:
    #     self._axis = gl.GLAxisItem()
    #     # self._axis.setSize(x=_len_x, y=_len_y, z=_len_z)
    #     self._axis.setSize(x=_len_x, y=0.25*_len_y, z=0.25*_len_z)
    #     self.addItem(self._axis)
    
    #     # Add a set of grids along x, y, z:
    #     self._xy_grid = gl.GLGridItem()
    #     self._xy_grid.setSize(x=_len_x, y=_len_y, z=0.0)
    #     self._xy_grid.setSpacing(x=meta.size_voxel_x(), y=meta.size_voxel_y(), z=0.0)
    #     self._xy_grid.translate(_len_x*0.5, _len_y * 0.5, 0.0)

    #     self._yz_grid = gl.GLGridItem()
    #     self._yz_grid.setSize(x=_len_z, y=_len_y)
    #     self._yz_grid.setSpacing(x=meta.size_voxel_x(), y=meta.size_voxel_y(), z=0.0)
    #     self._yz_grid.rotate(-90, 0, 1, 0)
    #     self._yz_grid.translate(0, _len_y*0.5, _len_z*0.5)

    #     self._xz_grid = gl.GLGridItem()
    #     self._xz_grid.setSize(x=_len_x, y=_len_z)
    #     self._xz_grid.setSpacing(x=meta.size_voxel_x(), y=meta.size_voxel_y(), z=0.0)
    #     self._xz_grid.rotate(90, 1, 0, 0)
    #     self._xz_grid.translate(_len_x*0.5, 0, _len_z*0.5)
    
    #     self.addItem(self._xy_grid)
    #     self.addItem(self._yz_grid)
    #     self.addItem(self._xz_grid)
    
    #     self._background_items.append(self._axis)
    #     self._background_items.append(self._xy_grid)
    #     self._background_items.append(self._yz_grid)
    #     self._background_items.append(self._xz_grid)
    

    def dims(self):
        return self._dims


    def setCenter(self, center):
        """Set the center of the view
        
        Move the world's center to the specified location
        
        Arguments:
            center {list} -- Length 3 list
        """
        if len(center) != 3:
            return
        cVec = QtGui.QVector3D(center[0],center[1],center[2])
        self.opts['center'] = cVec
        self.update()


    def worldCenter(self):
        """Return the world center point
        
        Returns:
            QVector3D -- World center location
        """
        return self.opts['center']

    def getAzimuth(self):
        return self.opts['azimuth']

    def getElevation(self):
        return self.opts['elevation']

    def setCameraPos(self,pos):
        """Set the camera position
        
        Take the 3D position and set the camera position with respect to the world center
        
        Arguments:
            pos {list} -- length 3 list of position
        """
        # Convert to spherical coordinates:
        if pos is not None and len(pos) == 3:
            # Convert to relative coordinates to always 
            # leave the world center as the center point
            worldCenter = self.opts['center']
            # Check the type:
            if type(worldCenter) is QtGui.QVector3D:
                X = pos[0] - worldCenter.x()
                Y = pos[1] - worldCenter.y()
                Z = pos[2] - worldCenter.z()
            else:
                X = pos[0] - worldCenter[0]
                Y = pos[1] - worldCenter[1]
                Z = pos[2] - worldCenter[2]

            distance = X**2 + Y**2 + Z**2
            distance = math.sqrt(distance)
            if X != 0:
                azimuth = math.atan2(Y,X)
            else:
                azimuth = math.pi
                if Y < 0:
                    azimuth = -1 * azimuth
            if distance != 0:
                elevation = math.asin(Z / distance)
            else:
                elevation = math.copysign(Z)
            azimuth *= 180./math.pi
            elevation *= 180./math.pi
            # Use the underlying function to set the camera position:
            self.setCameraPosition(distance=distance,elevation=elevation,azimuth=azimuth)
            self.viewChanged.emit()

    def keyPressEvent(self,e):
        """Wrap key press events
        
        Handle ctrl+C as well as shift + arrow or +/- keys, which pan the view
        
        Arguments:
            e {event} -- the key press event
        """
        if e.key() == QtCore.Qt.Key_C:
            # print "C was pressed"
            if e.modifiers() and QtCore.Qt.ControlModifier :
                self.quitRequested.emit()
                return
        elif e.modifiers():
            if QtCore.Qt.ShiftModifier :
                if e.key() == QtCore.Qt.Key_Up:
                    self.pan(0,20,0,True)
                if e.key() == QtCore.Qt.Key_Down:
                    self.pan(0,-20,0,True)
                if e.key() == QtCore.Qt.Key_Left:
                    self.pan(20,0,0,True)
                if e.key() == QtCore.Qt.Key_Right:
                    self.pan(-20,0,0,True)
                # # Z direction requires shift and ctrl:
                # if QtCore.QtControlModifier and e.key() == QtCore.Qt.Key_Up:
                #     self.pan(0,0,20,True)
                # if QtCore.QtControlModifier and e.key() == QtCore.Qt.Key_Down:
                #     self.pan(0,0,-20,True)

        else:
            super(Viewport3D,self).keyPressEvent(e)
       

        # Pass this signal to the main gui, too
        self.keyPressSignal.emit(e)

    def orbit(self, azim, elev):
        """Wrap the orbit function to enable viewChanged.emit"""
        super(Viewport3D, self).orbit(azim, elev)
        self.viewChanged.emit()

    def pan(self, dx, dy, dz, relative=False):
        """Wrap the pan function to enable viewChanged.emit"""
        super(Viewport3D, self).pan(dx, dy, dz, relative)
        self.viewChanged.emit()

    def wheelEvent(self, ev):
        """Wrap the wheelEvent function to enable viewChanged.emit"""
        super(Viewport3D, self).wheelEvent(ev)
        self.viewChanged.emit()

