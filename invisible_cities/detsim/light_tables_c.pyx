import numpy  as np

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
