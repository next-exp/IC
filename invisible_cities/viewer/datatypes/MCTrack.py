import pyqtgraph as pg
import numpy
import pyqtgraph.opengl as gl

from . DataBase import RecoBase3D

class MCTrack(RecoBase3D):

    """docstring for mctrack"""

    def __init__(self):
        super(mctrack, self).__init__()
        self._product_name = 'mctrack'
        self._points = None
        self._vals = None



    # this is the function that actually draws the cluster.
    def draw_objects(self, view_manager, event, meta):
        """Override draw_objects for mctracks
        
        Gather the MCTracks from the io, and put them on the screen
        
        Arguments:
            view_manager {ViewManager3D} -- The view manager
            io {IOManager} -- Instance of IOManager
            meta {EventMeta} -- Instance of EventMeta
        """
        # Get the data from the file:
        mc_tracks = event.mctracks()
        print(mc_tracks.keys())

        running_min = None
        running_max = None
        for track_id in mc_tracks:
            track = mc_tracks[track_id]
            print(type(track))
            # construct a line for this track:
            points = track.hits
            x = numpy.zeros(len(points))
            y = numpy.zeros(len(points))
            z = numpy.zeros(len(points))

            i = 0
            for point in points:
                x[i] = point.X
                y[i] = point.Y
                z[i] = point.Z
                i+= 1

            pts = numpy.vstack([x,y,z]).transpose()

            this_min = numpy.min(pts, axis=0)
            this_max = numpy.max(pts, axis=0)

            if running_min is None:
                running_min = this_min
            else:
                running_min = numpy.minimum(this_min, running_min)
            if running_max is None:
                running_max = this_max
            else:
                running_max = numpy.maximum(this_max, running_max)


            pen = pg.mkPen((255,0,0), width=8)
            line = gl.GLLinePlotItem(pos=pts,color=(255,0,0,255),width=8)
            view_manager.get_view().addItem(line)
            self._drawnObjects.append(line)

        self._min_coords = running_min
        self._max_coords = running_max
