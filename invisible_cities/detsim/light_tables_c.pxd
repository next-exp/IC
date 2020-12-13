cdef class LightTable:
    cdef readonly:
        double el_gap_width
        double active_radius
        int    num_sensors
    cdef:
        double [:] zbins_
    cdef double* get_values_(self, const double x, const double y, const int sensor_id)
