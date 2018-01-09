from invisible_cities.database import load_db



class NEWMeta(object):
    """Wrapper for geometry interface for NEW
    
    Contains easy access functions for NEW geometry properties, particularly
    detector dimensions, pmt and sipm locations
    """
    def __init__(self):
        super(NEWMeta, self).__init__()


        # Load the database objects
        self._det_geo   = load_db.DetectorGeo()
        self._pmt_data  = load_db.DataPMT()
        self._sipm_data = load_db.DataSiPM()

        self.skim_detector_info(self._det_geo)


    def skim_detector_info(self, det_geo):
        """Capture and store basic geometry information
        
        Gets basic geometry information from the database (x min/max, etc)
        Also specifies the size of pixels and number of pixels in each direction

        Units are implicitly in millimeters
        
        Arguments:
            det_geo {DetectorGeo Database} -- the output of load_db.DetectorGeo()
        """
        self._x_min   = det_geo.XMIN[0]
        self._y_min   = det_geo.YMIN[0]
        self._z_min   = det_geo.ZMIN[0]
        self._x_max   = det_geo.XMAX[0]
        self._y_max   = det_geo.YMAX[0]
        self._z_max   = det_geo.ZMAX[0]
        self._r_max   = det_geo.RMAX[0]
        # Use 1 pixel per sipm
        self._x_n_pixels = (self._x_max - self._x_min) / 10.
        self._y_n_pixels = (self._y_max - self._y_min) / 10.
        # Use 1 pixel per mm of drift
        self._z_n_pixels = self._z_max - self._z_min
        self._size_voxel_x = 10
        self._size_voxel_y = 10
        self._size_voxel_z = 1

    def pmt_data(self):
        return self._pmt_data

    def sipm_data(self):
        return self._sipm_data

    def size_voxel_x(self):
        return self._size_voxel_x

    def size_voxel_y(self):
        return self._size_voxel_y

    def size_voxel_z(self):
        return self._size_voxel_z

    def n_voxels_x(self):
        return self._x_n_pixels

    def n_voxels_y(self):
        return self._y_n_pixels

    def n_voxels_z(self):
        return self._y_n_pixels

    def len_x(self):
        return self._x_max - self._x_min

    def len_y(self):
        return self._y_max - self._y_min

    def len_z(self):
        return self._z_max - self._z_min

    def radius(self):
        return self._r_max

    def width(self):
        return self.len_x()

    def height(self):
        return self.len_y()

    def length(self):
        return self.len_z()

    def min_y(self):
        return self._y_min

    def max_y(self):
        return self._y_max

    def min_x(self):
        return self._x_min

    def max_x(self):
        return self._x_max

    def min_z(self):
        return self._z_min    

    def max_z(self):
        return self._z_max
