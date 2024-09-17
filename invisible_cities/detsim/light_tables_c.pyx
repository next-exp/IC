"""
This is a module to read the light tables files. Since this depends on the format of the table itself,
the table-specific class should inherit from the base LightTable class, providing:

 - get_values_ method, that depends on the x, y position in the EL plane and the internal sensor_id
   the output of the method is a pointer to array of all the values across EL gap for a given x, y bin
   if there are no values for a given x, y, sensor the result is a NULL pointer

 - zbins_  property,  is an  array of positions corresponding to the EL gap partitions

 - num_sensors attribute,  total number of sensors


LT_SiPM and LT_PMT are designed for currently availabel light tables for sipms and pmts;
should be revisited once the format for the lt is fixed and standardized.

"""


import numpy  as np
import pandas as pd

cimport cython
cimport numpy as np

from libc.math cimport            sqrt
from libc.math cimport           floor
from libc.math cimport round as cround



from ..         core import system_of_units as   units
from  . light_tables import read_lighttable as read_lt

cdef class LightTable:
    """
    Base abstract class to be inherited from for all LightTables classes.
    It needs get_values_ cython method implemented, as well as zbins_ and sensor_ids_ attributes.
    """

    cdef double* get_values_(self, const double x, const double y, const int sensor_id):
        raise NotImplementedError

    @property
    def zbins(self):
        """ Array of z positions """
        return np.asarray(self.zbins_)

    def get_values(self, const double x, const double y, const int sns_id):
        """
        Returns array of light table values over EL gap for x, y position
        of the electron and internal sensor id
        """
        cdef double* pointer
        pointer = self.get_values_(x, y, sns_id)
        if pointer!=NULL:
            return np.asarray(<np.double_t[:self.zbins_.shape[0]]> pointer)
        else:
            return np.zeros(self.zbins_.shape[0])


def get_el_bins(el_pitch, el_gap):
    """
    Returns the array of bins position given the bin distance and the total gap
    """
    return np.arange(el_pitch/2., el_gap, el_pitch).astype(np.double)

cdef class LT_SiPM(LightTable):
    """
    A class to handle reading of sipm distance-based light table. Inherits from base class LightTable

    Attributes:
    -----------
       el_gap_width  : double
             width of the EL gap
       active_radius : double
             active radius of full detector volume
       num_sensors   : int
             number of sipm sensors
       snsx          : numpy array of doubles
             x position of sensors
       snsy          : numpy array of doubles
             y position of sensors

    Parameters (keyword):
    -----------
        fname         : string
              filename of the light table
        sipm_database : pandas dataframe
              dataframe containing information about sensor positions
        el_gap_width  : float
              optionally set new EL gap width
        active_radius : float
              optionally set new active radius

    """

    cdef readonly:
        double [:] snsx
        double [:] snsy
    cdef:
        double [:, ::1] values
        double psf_bin
        double max_zel
        double max_psf
        double max_psf2
        double inv_bin
        double active_r2

    def __init__(self, *, fname, sipm_database, el_gap_width=None, active_radius=None, data_mc_ratio=1):
        if data_mc_ratio <= 0: raise ValueError("LT_SiPM: data_mc_ratio must be greater than 0")

        lt_df, config_df, el_gap, active_r = read_lt(fname, 'PSF', el_gap_width, active_radius)
        lt_df.set_index('dist_xy', inplace=True)
        self.el_gap_width  = el_gap
        self.active_radius = active_r
        self.active_r2 = active_r**2 # compute this once to speed up the get_values_ calls

        el_pitch  = float(config_df.loc["pitch_z"].value) * units.mm
        self.zbins_    = get_el_bins(el_pitch, el_gap)
        self.values    = np.array(lt_df.values/len(self.zbins_) * data_mc_ratio, order='C', dtype=np.double)
        self.psf_bin   = float(lt_df.index[1]-lt_df.index[0]) * units.mm #index of psf is the distance to the sensor in mm
        self.inv_bin   = 1./self.psf_bin # compute this once to speed up the get_values_ calls

        self.snsx        = sipm_database.X.values.astype(np.double)
        self.snsy        = sipm_database.Y.values.astype(np.double)
        self.max_zel     = el_gap
        self.max_psf     = max(lt_df.index.values)
        self.max_psf2    = self.max_psf**2
        self.num_sensors = len(sipm_database)

    @cython.wraparound(False)
    cdef double* get_values_(self, const double x, const double y, const int sns_id):
        cdef:
            double dist
            double aux
            unsigned int psf_bin_id
            double xsipm
            double ysipm
            double tmp_x
            double tmp_y
            double*  values
        if sns_id >= self.num_sensors:
            return NULL
        xsipm = self.snsx[sns_id]
        ysipm = self.snsy[sns_id]
        tmp_x = x-xsipm; tmp_y = y-ysipm
        dist = tmp_x*tmp_x + tmp_y*tmp_y
        if dist>self.max_psf2:
            return NULL
        if x*x+y*y>=self.active_r2:
            return NULL
        aux = sqrt(dist)*self.inv_bin
        bin_id = <int> floor(aux)
        values = &self.values[bin_id, 0]
        return values

    def get_values(self, const double x, const double y, const int sns_id):
        """
        Retrive values from the light tables for all z partitions.

        Parameters:
        -----------
        x, y   : doubles
            electron position at EL plane
        sns_id : int
            internal sensor id in range [0, num_sensors)
            sensors are ordered by sipm_database raws

        Returns:
        --------
        array of values over EL gap partitions
        """
        return super().get_values(x, y, sns_id)


cdef class LT_PMT(LightTable):
    """
    A class to handle reading of PMTs light table. Inherits from base class LightTable

    Attributes:
    -----------
       el_gap_width  : double
             width of the EL gap
       active_radius : double
             active radius of full detector volume
       num_sensors   : int
             number of sipm sensors

    Parameters (keyword):
    -----------
        fname         : string
              filename of the light table
        el_gap_width  : float
              optionally set new EL gap width
        active_radius : float
              optionally set new active radius

    """

    cdef:
        double [:, :, :, ::1] values
        double max_zel
        double max_psf
        double max_psf2
        double inv_binx
        double inv_biny
        double xmin
        double ymin
        double active_r2

    def __init__(self, *, fname, el_gap_width=None, active_radius=None, data_mc_ratio=1):
        if data_mc_ratio <= 0: raise ValueError("LT_PMT: data_mc_ratio must be greater than 0")

        lt_df, config_df, el_gap, active_r = read_lt(fname, 'LT', el_gap_width, active_radius)
        self.el_gap_width  = el_gap
        self.active_radius = active_r
        self.active_r2 = active_r**2 # compute this once to speed up the get_values_ calls

        sensor = config_df.loc["sensor"].value
        #remove column total from the list of columns
        columns = [col for col in lt_df.columns if ((sensor in col) and ("total" not in col))]
        el_pitch    = el_gap #hardcoded for this specific table
        bin_x = float(config_df.loc["pitch_x"].value) * units.mm
        bin_y = float(config_df.loc["pitch_y"].value) * units.mm

        self.zbins_ = get_el_bins(el_pitch, el_gap)
        values_aux, (xmin, xmax), (ymin, ymax)  = self.__extend_lt_bounds(lt_df, config_df, columns, bin_x, bin_y)
        lenz = len(self.zbins)
        # add dimension for z partitions (1 in case of this table)
        self.values = np.asarray(np.repeat(values_aux, lenz, axis=-1) * data_mc_ratio, dtype=np.double, order='C')
        self.xmin = xmin
        self.ymin = ymin
        # calculate inverse to speed up calls of get_values_
        self.inv_binx    = 1./bin_x
        self.inv_biny    = 1./bin_y
        self.num_sensors = len(columns)

    def __extend_lt_bounds(self, lt_df, config_df, columns, bin_x, bin_y):
        """
        Extend light tables values up to a full active_radius volume, using nearest interpolation method.
        The resulting tensor has shape of num_bins_x, num_bins_y.
        """
        from scipy.interpolate import griddata
        xtable   = lt_df.x.values
        ytable   = lt_df.y.values
        xmin_, xmax_ = xtable.min(), xtable.max()
        ymin_, ymax_ = ytable.min(), ytable.max()
        # extend min, max to go one bin-width over the active volume
        xmin, xmax = xmin_-np.ceil((self.active_radius-np.abs(xmin_))/bin_x)*bin_x, xmax_+np.ceil((self.active_radius-np.abs(xmax_))/bin_x)*bin_x
        ymin, ymax = ymin_-np.ceil((self.active_radius-np.abs(ymin_))/bin_y)*bin_y, ymax_+np.ceil((self.active_radius-np.abs(ymax_))/bin_y)*bin_y
        #create new centers that extend over full active volume
        x          = np.arange(xmin, xmax+bin_x/2., bin_x).astype(np.double)
        y          = np.arange(ymin, ymax+bin_y/2., bin_y).astype(np.double)
        #interpolate missing values using nearest method from scipy
        xx, yy     = np.meshgrid(x, y)
        values_aux = (np.concatenate([griddata((xtable, ytable), lt_df[column], (yy, xx), method='nearest')[..., None]
                                      for column in columns],axis=-1)[..., None]).astype(np.double)
        return values_aux, (xmin, xmax), (ymin, ymax)

    @cython.wraparound(False)
    cdef double* get_values_(self, const double x, const double y, const int sns_id):
        cdef:
            double*  values
            int xindx, yindx
        if (x*x+y*y)>=self.active_r2 :
            return NULL
        if sns_id >= self.num_sensors:
            return NULL
        xindx = <int> cround((x-self.xmin)*self.inv_binx)
        yindx = <int> cround((y-self.ymin)*self.inv_biny)
        values = &self.values[xindx, yindx, sns_id, 0]
        return values

    def get_values(self, const double x, const double y, const int sns_id):
        """
        Retrive values from the light tables for all z partitions.

        Parameters:
        -----------
        x, y   : doubles
            electron position at EL plane
        sns_id : int
            internal sensor id in range [0, num_sensors)
            sensors are ordered by columns of light table file

        Returns:
        --------
        array of values over EL gap partitions
        """
        return super().get_values(x, y, sns_id)
