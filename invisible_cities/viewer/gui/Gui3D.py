import numpy as np

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

from . ViewManager3D import ViewManager3D

# 

class ConnectedSpinBox(QtGui.QSpinBox):
    """Wrap the spin box class to allow key signals to pass to the Gui
    
    [description]
    
    Extends:
        QtGui.QSpinBox
    
    Variables:
        quit_requested {pyqtsignal} -- signal to emit when quit has been requested
        and this class has focus

    """
    quit_requested = QtCore.pyqtSignal()
    def __init__(self):
        super(ConnectedSpinBox, self).__init__()
        

    def keyPressEvent(self, e):
        """Catch ctrl + C calls
        
        Intercept keyPressEvents for ctrl + c.  Other keyPressEvents
        are passed to the default handler for QSpinBox
        
        Arguments:
            e {} -- Key event
        """
        if e.key() == QtCore.Qt.Key_C:
            if e.modifiers() and QtCore.Qt.ControlModifier :
                self.quit_requested.emit()
                return
        else:
            super(ConnectedSpinBox,self).keyPressEvent(e )



class Gui3D(QtGui.QWidget):
    """Basic 3D window interface
    
    This class creates a 3D view interface including camera controls, a widget
    that has OpenGL rendering ability (member variable self._view_manager.view()), 
    and basic event interface buttons (next, previous, go_to_event, etc.)
    
    Extends:
        QtGui.QWidget
    
    Variables:
        self._view_manager {[type]} -- [description]
        
    """

    def __init__(self, manager):
        super(Gui3D, self).__init__()

        # init_ui should not do ANY data handling, it should only get the interface loaded
        self._view_manager = ViewManager3D()
        self._view_manager.get_view().draw_detector(manager.meta())
        self._event_manager = manager

    def closeEvent(self, event):
        """Wrap close Events to call quit

        Arguments:
            event {[type]} -- [description]

        """
        self.quit()  

    def quit(self):
        """Close the qapplication
        
        """
        QtCore.QCoreApplication.instance().quit()


    def update(self):
        """Keeps informational text up to date
        
        Keeps run, event, etc up to date when things change
        Also triggers a fresh draw of items to be rendered
        
        """

        # set the text boxes correctly:
        self._entryBox.setText(str(self._event_manager.io().entry()))

        eventLabel = "Ev: " + str(self._event_manager.io().event())
        self._eventLabel.setText(eventLabel)

        runLabel = "Run: " + str(self._event_manager.io().run())
        self._runLabel.setText(runLabel)

        subrunLabel = "Subrun: " + str(self._event_manager.io().subrun())
        self._subrunLabel.setText(subrunLabel)
        
        self._event_manager.draw_fresh(self._view_manager)


    def update_camera_info(self, cameraPos=None,worldPos=None):
        """Update camera text when things change
        
        This function is not for changing the view when the text boxes are changed
        It is for keeping the text boxes in sync with the view as the user changes the view.
        
        Keyword Arguments:
            cameraPos {[type]} -- [description] (default: {None})
            worldPos {[type]} -- [description] (default: {None})

        """
        # 

        # UPdate all of the camera text entries:
        if cameraPos is None:
            cameraPos = self._view_manager.get_view().cameraPosition()
        if worldPos is None:
            worldPos = self._view_manager.get_view().worldCenter()

        # To actually update these things corrently, we have to unplug 
        # the signals from the slots, update the fields, and then plug everything back in
        

        try:
            self._cameraCenterX.valueChanged.disconnect()
        except:
            pass
        try:
            self._cameraCenterY.valueChanged.disconnect()
        except:
            pass
        try:
            self._cameraCenterZ.valueChanged.disconnect()
        except:
            pass
        try:
            self._worldCenterX.valueChanged.disconnect()
        except:
            pass
        try:
            self._worldCenterY.valueChanged.disconnect()
        except:
            pass
        try:
            self._worldCenterZ.valueChanged.disconnect()
        except:
            pass

        self._cameraCenterX.setValue(cameraPos.x())
        self._cameraCenterY.setValue(cameraPos.y())
        self._cameraCenterZ.setValue(cameraPos.z())
        self._worldCenterX.setValue(worldPos.x())
        self._worldCenterY.setValue(worldPos.y())
        self._worldCenterZ.setValue(worldPos.z())

        self._cameraCenterX.valueChanged.connect(self.camera_center_worker)
        self._cameraCenterY.valueChanged.connect(self.camera_center_worker)
        self._cameraCenterZ.valueChanged.connect(self.camera_center_worker)
        self._worldCenterX.valueChanged.connect(self.world_center_worker)
        self._worldCenterY.valueChanged.connect(self.world_center_worker)
        self._worldCenterZ.valueChanged.connect(self.world_center_worker)


    # This function prepares the buttons such as prev, next, etc and returns a layout
    def get_event_control_buttons(self):
        """Prepares the buttons such as prev, next, etc and returns a layout
        
        """
        # This is a box to allow users to enter an event
        self._goToLabel = QtGui.QLabel("Go to: ")
        self._entryBox = QtGui.QLineEdit()
        self._entryBox.setToolTip("Enter an event to skip to that event")
        self._entryBox.returnPressed.connect(self.go_to_event_worker)

        # # These labels display current events
        self._runLabel = QtGui.QLabel("Run: 0")
        self._eventLabel = QtGui.QLabel("Ev.: 0")
        self._subrunLabel = QtGui.QLabel("Subrun: 0")

        # Jump to the next event
        self._nextButton = QtGui.QPushButton("Next")
        # self._nextButton.setStyleSheet("background-color: red")
        self._nextButton.clicked.connect(self._event_manager.next)
        self._nextButton.setToolTip("Move to the next event.")
        
        # Go to the previous event
        self._prevButton = QtGui.QPushButton("Previous")
        self._prevButton.clicked.connect(self._event_manager.prev)
        self._prevButton.setToolTip("Move to the previous event.")

        
        # pack the buttons into a box
        self._eventControlBox = QtGui.QVBoxLayout()

        # Make a horiztontal box for the event entry and label:
        self._eventGrid = QtGui.QHBoxLayout()
        self._eventGrid.addWidget(self._goToLabel)
        self._eventGrid.addWidget(self._entryBox)

        # Pack it all together
        self._eventControlBox.addLayout(self._eventGrid)
        self._eventControlBox.addWidget(self._eventLabel)
        self._eventControlBox.addWidget(self._runLabel)
        self._eventControlBox.addWidget(self._subrunLabel)
        self._eventControlBox.addWidget(self._nextButton)
        self._eventControlBox.addWidget(self._prevButton)

        return self._eventControlBox
  

    def go_to_event_worker(self):
        """Pass the entry of line edit to the event control
        
        """
        try:
            event = int(self._entryBox.text())
        except:
            print("Error, must enter an integer")
            self._entryBox.setText(str(self._event_manager.entry()))
            return
        self._event_manager.go_to_entry(event)

    def get_drawing_control_buttons(self):
        """prepares the range controlling options and returns a layout
        
        """
        # Button to set range to max
        self._autoRangeButton = QtGui.QPushButton("Auto Range")
        self._autoRangeButton.setToolTip("Set the range of the viewers to show the whole event")
        self._autoRangeButton.clicked.connect(self.auto_range_worker)


        # add a box to restore the drawing defaults:
        self._restoreDefaults = QtGui.QPushButton("Restore Defaults")
        self._restoreDefaults.setToolTip("Restore the drawing defaults of the views.")
        self._restoreDefaults.clicked.connect(self.restore_defaults_worker)


        # Add some controls to manage the camera
        self._cameraControlLayout = QtGui.QHBoxLayout()

        # Get the min and max values for height, length, width:

        width  = self._event_manager.meta().max_x() - self._event_manager.meta().min_x()
        height = self._event_manager.meta().max_y() - self._event_manager.meta().min_y()
        length = self._event_manager.meta().max_z() - self._event_manager.meta().min_z()
        

        # Define the x,y,z location of the camera and world center
        self._cameraCenterLayout = QtGui.QVBoxLayout()
        self._cameraLabel = QtGui.QLabel("Camera")
        self._cameraCenterLayout.addWidget(self._cameraLabel)
        self._cameraCenterXLayout = QtGui.QHBoxLayout()
        self._cameraCenterXLabel = QtGui.QLabel("X:")
        self._cameraCenterX = ConnectedSpinBox()
        self._cameraCenterX.setValue(0)
        self._cameraCenterX.setRange(-10*width,10*width)
        self._cameraCenterX.quit_requested.connect(self.quit)
        self._cameraCenterX.valueChanged.connect(self.camera_center_worker)
        self._cameraCenterXLayout.addWidget(self._cameraCenterXLabel)
        self._cameraCenterXLayout.addWidget(self._cameraCenterX)

        self._cameraCenterLayout.addLayout(self._cameraCenterXLayout)
        self._cameraCenterYLayout = QtGui.QHBoxLayout()
        self._cameraCenterYLabel = QtGui.QLabel("Y:")
        self._cameraCenterY = ConnectedSpinBox()
        self._cameraCenterY.setValue(0)
        self._cameraCenterY.setRange(-10*height,10*height)
        self._cameraCenterY.quit_requested.connect(self.quit)
        self._cameraCenterY.valueChanged.connect(self.camera_center_worker)
        self._cameraCenterYLayout.addWidget(self._cameraCenterYLabel)
        self._cameraCenterYLayout.addWidget(self._cameraCenterY)

        self._cameraCenterLayout.addLayout(self._cameraCenterYLayout)
        self._cameraCenterZLayout = QtGui.QHBoxLayout()
        self._cameraCenterZLabel = QtGui.QLabel("Z:")
        self._cameraCenterZ = ConnectedSpinBox()
        self._cameraCenterZ.setValue(0)
        self._cameraCenterZ.setRange(-10*length,10*length)   
        self._cameraCenterZ.quit_requested.connect(self.quit)
        self._cameraCenterZ.valueChanged.connect(self.camera_center_worker)
        self._cameraCenterZLayout.addWidget(self._cameraCenterZLabel)
        self._cameraCenterZLayout.addWidget(self._cameraCenterZ)
        self._cameraCenterLayout.addLayout(self._cameraCenterZLayout)


        self._worldCenterLayout = QtGui.QVBoxLayout()
        self._worldLabel = QtGui.QLabel("world")
        self._worldCenterLayout.addWidget(self._worldLabel)
        self._worldCenterXLayout = QtGui.QHBoxLayout()
        self._worldCenterXLabel = QtGui.QLabel("X:")
        self._worldCenterX = ConnectedSpinBox()
        self._worldCenterX.setValue(0)
        self._worldCenterX.setRange(-10*width,10*width)
        self._worldCenterX.quit_requested.connect(self.quit)
        self._worldCenterX.valueChanged.connect(self.world_center_worker)
        self._worldCenterXLayout.addWidget(self._worldCenterXLabel)
        self._worldCenterXLayout.addWidget(self._worldCenterX)

        self._worldCenterLayout.addLayout(self._worldCenterXLayout)
        self._worldCenterYLayout = QtGui.QHBoxLayout()
        self._worldCenterYLabel = QtGui.QLabel("Y:")
        self._worldCenterY = ConnectedSpinBox()
        self._worldCenterY.setValue(0)
        self._worldCenterY.setRange(-10*height,10*height)
        self._worldCenterY.quit_requested.connect(self.quit)
        self._worldCenterY.valueChanged.connect(self.world_center_worker)
        self._worldCenterYLayout.addWidget(self._worldCenterYLabel)
        self._worldCenterYLayout.addWidget(self._worldCenterY)

        self._worldCenterLayout.addLayout(self._worldCenterYLayout)
        self._worldCenterZLayout = QtGui.QHBoxLayout()
        self._worldCenterZLabel = QtGui.QLabel("Z:")
        self._worldCenterZ = ConnectedSpinBox()
        self._worldCenterZ.setValue(0)
        self._worldCenterZ.setRange(-10*length,10*length)   
        self._worldCenterZ.quit_requested.connect(self.quit)
        self._worldCenterZ.valueChanged.connect(self.world_center_worker)
        self._worldCenterZLayout.addWidget(self._worldCenterZLabel)
        self._worldCenterZLayout.addWidget(self._worldCenterZ)
        self._worldCenterLayout.addLayout(self._worldCenterZLayout)



        # Pack the stuff into a layout

        self._drawingControlBox = QtGui.QVBoxLayout()
        self._drawingControlBox.addWidget(self._restoreDefaults)
        self._drawingControlBox.addWidget(self._autoRangeButton)
        self._drawingControlBox.addLayout(self._cameraControlLayout)

        self._drawingControlBox.addLayout(self._cameraCenterLayout)
        self._drawingControlBox.addLayout(self._worldCenterLayout)

        return self._drawingControlBox

    def world_center_worker(self):
        """Worker function for updating the world center
        """
        x = float(self._worldCenterX.text())
        y = float(self._worldCenterY.text())
        z = float(self._worldCenterZ.text())
        self._view_manager.get_view().setCenter((x,y,z))

    def camera_center_worker(self):
        """Worker function for updating the camera position
        """
        # assemble the camera position:
        x = float(self._cameraCenterX.text())
        y = float(self._cameraCenterY.text())
        z = float(self._cameraCenterZ.text())
        self._view_manager.get_view().setCameraPos(pos = (x,y,z) )

    def auto_range_worker(self):
        """Worker function to handle clickinng the autorange button
        
        """
        # Get the list of min/max coordinates:
        cmin, cmax = self._event_manager.get_min_max_coords()
        ctr = 0.5*(cmin + cmax)
        diag = (cmax - cmin) * np.asarray((1.5, 1.5, 1.0))
        self._view_manager.get_view().setCenter(ctr)
        self._view_manager.get_view().setCameraPos(ctr + diag)

    def restore_defaults_worker(self):
        """Restore the default values for the view"""
        self._view_manager.set_range_to_max()
    
    def get_quit_layout(self):
        """Prepare the layout for the quit button"""
        self._quitButton = QtGui.QPushButton("Quit")
        self._quitButton.setToolTip("Close the viewer.")
        self._quitButton.clicked.connect(self.quit)
        return self._quitButton

      # 
    def get_west_layout(self):
        """This function combines the control button layouts, range layouts, and quit button
        
        """
        event_control = self.get_event_control_buttons()
        draw_control = self.get_drawing_control_buttons()


        # Add the quit button?
        quit_control = self.get_quit_layout()
        
        self._westLayout = QtGui.QVBoxLayout()
        self._westLayout.addLayout(event_control)
        self._westLayout.addStretch(1)
        self._westLayout.addLayout(draw_control)
        self._westLayout.addStretch(1)


        self._westLayout.addStretch(1)

        self._westLayout.addWidget(quit_control)
        self._westWidget = QtGui.QWidget()
        self._westWidget.setLayout(self._westLayout)
        self._westWidget.setMaximumWidth(150)
        self._westWidget.setMinimumWidth(100)
        return self._westWidget


    def get_south_layout(self):
        """This layout contains the status bar and the capture screen buttons"""

        # The screen capture button:
        self._screenCaptureButton = QtGui.QPushButton("Capture Screen")
        self._screenCaptureButton.setToolTip("Capture the entire screen to file")
        self._screenCaptureButton.clicked.connect(self.screenCapture)

        self._southWidget = QtGui.QWidget()
        self._southLayout = QtGui.QHBoxLayout()
        # Add a status bar
        self._statusBar = QtGui.QStatusBar()
        self._statusBar.showMessage("Test message")
        self._southLayout.addWidget(self._statusBar)
        # self._southLayout.addStretch(1)
        self._southLayout.addWidget(self._screenCaptureButton)
        self._southWidget.setLayout(self._southLayout)

        return self._southWidget

    def get_east_layout(self):
        """This function just makes a dummy eastern layout to use."""
        label = QtGui.QLabel("Dummy")
        self._eastWidget = QtGui.QWidget()
        self._eastLayout = QtGui.QVBoxLayout()
        self._eastLayout.addWidget(label)
        self._eastLayout.addStretch(1)
        self._eastWidget.setLayout(self._eastLayout)
        self._eastWidget.setMaximumWidth(200)
        self._eastWidget.setMinimumWidth(100)
        return self._eastWidget



    def init_ui(self):
        """Main initialization function

        Build the entire gui with this function, and pack in into the main layout
        which is this class's layout
        """


        # Get all of the widgets:
        self.eastWidget  = self.get_east_layout()
        self.westWidget  = self.get_west_layout()
        self.southLayout = self.get_south_layout()

        # Area to hold data:
        self._view = self._view_manager.get_view()
        self._view.keyPressSignal.connect(self.keyPressEvent)
        self.centerLayout = self._view_manager.get_layout()
        self._view.quitRequested.connect(self.quit)
        self._view.viewChanged.connect(self.update_camera_info)

        # Put the layout together
        self.master = QtGui.QVBoxLayout()
        self.slave = QtGui.QHBoxLayout()
        self.slave.addWidget(self.westWidget)
        self.slave.addLayout(self.centerLayout)
        self.slave.addWidget(self.eastWidget)
        self.master.addLayout(self.slave)
        self.master.addWidget(self.southLayout)

        self.setLayout(self.master)    

        self.update_camera_info()

        self.setGeometry(0, 0, 2400, 1600)
        self.setWindowTitle('Event Display')    
        self.setFocus()
        self.show()

    def keyPressEvent(self,e):
        """Intercept key press events

        Allows control of N/P for next/prev and ctrl + C
        """
        if e.key() == QtCore.Qt.Key_N:
            self._event_manager.next()
            return
        if e.key() == QtCore.Qt.Key_P:
            self._event_manager.prev()
            return
        if e.key() == QtCore.Qt.Key_C:
            # print("C was pressed")
            if e.modifiers() and QtCore.Qt.ControlModifier :
                self.quit()
                return

        # Pass unused key press items up
        super(Gui3D, self).keyPressEvent(e)

    def screenCapture(self):
        """Capture the screen viewed
        
        """
        print("Screen Capture!")
        dialog = QtGui.QFileDialog()
        r = self._event_manager.run()
        e = self._event_manager.event()
        s = self._event_manager.subrun()
        name = "larcv_3D_" + "R" + str(r)
        name = name + "_S" + str(s)
        name = name + "_E" + str(e) + ".png"
        f = dialog.getSaveFileName(self,"Save File",name,
            "PNG (*.png);;JPG (*.jpg);;All Files (*)")

        if (pg.Qt.QtVersion.startswith('4')):
              pixmapImage = QtGui.QPixmap.grabWidget(self)
              pixmapImage.save(f,"PNG")
        else:
              pixmapImage = super(Gui3D, self).grab()
              pixmapImage.save(f[0],"PNG")



