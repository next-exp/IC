import numpy  as np
import tables as tb

from typing  import Callable

from .. reco import tbl_functions as tbl


def rwf_writer(h5out           : tb.file.File          ,
               *,
               group_name      :          str          ,
               table_name      :          str          ,
               compression     :          str = 'ZLIB4',
               n_sensors       :          int          ,
               waveform_length :          int          ) -> Callable:
    """
    Defines group and table where raw waveforms
    will be written.

    h5out           : pytables file
                      Output file where waveforms to be saved
    group_name      : str
                      Name of the group in h5in.root
                      Known options: RD, BLR
                      Setting to root will save directly in root
    table_name      : str
                      Name of the table
                      Known options: pmtrwf, pmtcwf, sipmrwf
    compression     : str
                      file compression
    n_sensors       : int
                      number of sensors in the table (shape[0])
    waveform_length : int
                      Number of samples per sensor
    """
    if group_name is 'root':
        rwf_group = h5out.root
    else:
        try:
            rwf_group = getattr           (h5out.root, group_name)
        except tb.NoSuchNodeError:
            rwf_group = h5out.create_group(h5out.root, group_name)

    rwf_table = h5out.create_earray(rwf_group,
                                   table_name,
                                   atom    = tb.Int16Atom(),
                                   shape   = (0, n_sensors, waveform_length),
                                   filters = tbl.filters(compression))
    def write_rwf(waveform : np.ndarray) -> None:
        """
        Writes raw waveform arrays to file.
        waveform : np.ndarray
                   shape = (n_sensors, waveform_length) array
                   of sensor charge.
        """
        rwf_table.append(waveform.reshape(1, n_sensors, waveform_length))
    return write_rwf
