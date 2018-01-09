
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

from . Viewport3D import Viewport3D

# Default color scale:
colorMap = {'ticks': [(1, (151, 30, 22, 125)),
                      (0.791, (0, 181, 226, 125)),
                      (0.645, (76, 140, 43, 125)),
                      (0.47, (0, 206, 24, 125)),
                      (0.33333, (254, 209, 65, 125)),
                      (0, (255, 255, 255, 125))],
            'mode': 'rgb'}

class ViewManager3D(QtCore.QObject):
    """This class manages the viewport 3D"""

    refreshColors = QtCore.pyqtSignal()

    def __init__(self):
        super(ViewManager3D, self).__init__()
        # Add a view to this widget:
        self._view = Viewport3D()


        # Define some color collections:
        self._cmap = pg.GradientWidget(orientation='right')
        self._cmap.restoreState(colorMap)
        self._cmap.sigGradientChanged.connect(self.gradient_change_finished)
        self._cmap.resize(1, 1)

        # Color map lookup table:
        self._lookupTable = self._cmap.getLookupTable(255, alpha=0.75)

        # These boxes control the levels.
        self._upperLevel = QtGui.QLineEdit()
        self._lowerLevel = QtGui.QLineEdit()

        self._upperLevel.returnPressed.connect(self.colors_changed)
        self._lowerLevel.returnPressed.connect(self.colors_changed)

        self._lowerLevel.setText(str(0.0))
        self._upperLevel.setText(str(10.0))

        # Fix the maximum width of the widgets:
        self._upperLevel.setMaximumWidth(35)
        self._cmap.setMaximumWidth(25)
        self._lowerLevel.setMaximumWidth(35)


        # Set up the layout of this widget:
        self._layout = QtGui.QHBoxLayout()
        self._layout.addWidget(self._view)

        colors = QtGui.QVBoxLayout()
        colors.addWidget(self._upperLevel)
        colors.addWidget(self._cmap)
        colors.addWidget(self._lowerLevel)

        self._layout.addLayout(colors)

    def gradient_change_finished(self):
        """Update the lookup table when the gradient changes
        
        """
        self._lookupTable = self._cmap.getLookupTable(255, alpha=0.75)
        self.refreshColors.emit()


    def get_lookup_table(self):
        """Return the lookup table
        
        Returns:
            [list] -- The lookup table as a list
        """
        return self._lookupTable*(1./255)


    def colors_changed(self):
        """Notify when colors are changed
        
        """
        self.refreshColors.emit() 



    def get_layout(self):
        """Return the layout of this widget
        
        Returns:
            [QLayout] -- This widget's layout
        """
        return self._layout


    def set_range_to_max(self):
        """Set the camera position to the maximum recommended value
        
        """
        dims = self._view.dims()
        self._view.setCenter((0.0,0.0,0.0))
        self._view.setCameraPos((1.5*dims[0], 0, 1.0*dims[2]))
        # Move the center of the camera to the center of the view:
        # self._view.pan(dims[0]*0.5, dims[1] * 0.5, dims[2]*0.5)

    def get_view(self):
        """ Return the view"""
        return self._view

    def get_levels(self):
        """Return the current levels
        
        """
        lmax = float(self._upperLevel.text())
        lmin = float(self._lowerLevel.text())
        return (lmin, lmax)


    def update(self):
        self._view.update()

    def restoreDefaults(self):
        print("restoreDefaults called but not implemented")
