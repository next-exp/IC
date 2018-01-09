from pyqtgraph import opengl as gl

from . Gui3D import Gui3D
from pyqtgraph.Qt import QtGui


# Inherit the basic gui to extend it
# override the gui to give the display special features:


class EvdGui3D(Gui3D):

    """Specialized GUI for next EVD 3D"""

    def __init__(self, mgr):
        super(EvdGui3D, self).__init__(mgr)
        # Connect the manager functions as necessary:
        self._event_manager.eventChanged.connect(self.update)
        self._view_manager.refreshColors.connect(self.refresh_colors)


    # override the init_ui function to change things:
    def init_ui(self):
        super(EvdGui3D, self).init_ui()
        self._view_manager.set_range_to_max()
        self.update()

    def refresh_colors(self):
        """Worker to pass refresh colors signals"""
        self._event_manager.refresh_colors(self._view_manager)

    # This function sets up the eastern widget
    def get_east_layout(self):
        # This function just makes a dummy eastern layout to use.
        label1 = QtGui.QLabel("NEXT")
        label2 = QtGui.QLabel("EVD 3D")
        font = label1.font()
        font.setBold(True)
        label1.setFont(font)
        label2.setFont(font)

        self._eastWidget = QtGui.QWidget()
        # This is the total layout
        self._eastLayout = QtGui.QVBoxLayout()
        # add the information sections:
        self._eastLayout.addWidget(label1)
        self._eastLayout.addWidget(label2)
        self._eastLayout.addStretch(1)
        

        # In this case, many things are not made with different producers
        # but just exist.  So use check boxes to toggle them on and off.

        # Now we get the list of items that are drawable:
        drawableProducts = self._event_manager.get_drawable_items()

        for product in drawableProducts:
            thisBox = QtGui.QCheckBox(product)
            thisBox.stateChanged.connect(self.checkBoxHandler)
            self._eastLayout.addWidget(thisBox)


        self._eastLayout.addStretch(2)

        self._eastWidget.setLayout(self._eastLayout)
        self._eastWidget.setMaximumWidth(150)
        self._eastWidget.setMinimumWidth(100)
        return self._eastWidget


    def checkBoxHandler(self, state):
        sender = self.sender()
        if not sender.isChecked():
            self._event_manager.redraw_product(str(sender.text()), self._view_manager, draw=False)
            return
        else:
            self._event_manager.redraw_product(str(sender.text()), self._view_manager, draw=True)

