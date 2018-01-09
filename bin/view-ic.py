#!/usr/bin/env python

import argparse
import sys
import signal
from pyqtgraph.Qt import QtGui, QtCore

from . gui     import evdgui3D
from . manager import evd_manager_3D


def sigintHandler(*args):
    """Handler for the SIGINT signal."""
    sys.stderr.write('\r')
    sys.exit()


def main():
    """Main executed function for the viewer
    
    Creates the data interface manager, the GUI, and starts the event loop
    """
    
    # Parse command line arguments.  Only the file to open is included, but more 
    # could come in the future
    parser = argparse.ArgumentParser(description='Python based event display.')
    parser.add_argument('file', nargs='*', help="Optional input file to use")

    args = parser.parse_args()

    # Create an instance of a QApplication
    app = QtGui.QApplication(sys.argv)


    # Initialize the manager and pass it the file to open
    manager = EvdManager3D(args.file)

    # Initialize the GUI, which knows about the manager
    thisgui = evdgui3D(manager)
    
    #init_ui builds the PyQt user interface
    thisgui.init_ui()

    # Create a signal to catch ctrl+C in the interpreter
    signal.signal(signal.SIGINT, sigintHandler)
    timer = QtCore.QTimer()
    timer.start(500)  # You may change this if you wish.
    timer.timeout.connect(lambda: None)  # Let the interpreter run each 500 ms.

    # Launch the application:
    app.exec_()


if __name__ == '__main__':
    main()
