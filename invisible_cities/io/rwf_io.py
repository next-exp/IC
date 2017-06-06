import tables as tb

from .. reco import tbl_functions as tbl


def rwf_writer(file,
               *,
               group_name      : 'options: RD, BLR',
               table_name      : 'options: pmtrwf, pmtcwf, sipmrwf',
               compression     = 'ZLIB4',
               n_sensors       : 'number of pmts or sipms',
               waveform_length : 'length of pmt or sipm waveform_length'):
    try:                       rwf_group = getattr          (file.root, group_name)
    except tb.NoSuchNodeError: rwf_group = file.create_group(file.root, group_name)

    rwf_table = file.create_earray(rwf_group,
                                   table_name,
                                   atom    = tb.Int16Atom(),
                                   shape   = (0, n_sensors, waveform_length),
                                   filters = tbl.filters(compression))
    def write_rwf(waveform : 'np.array: RWF, CWF, SiPM'):
        rwf_table.append(waveform.reshape(1, n_sensors, waveform_length))
    return write_rwf
