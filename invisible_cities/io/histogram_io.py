import tables as tb

from .. reco import tbl_functions as tbl


def hist_writer(file,
                *,
                group_name  : 'options: HIST, HIST2D',
                table_name  : 'options: pmt, pmtMAU, sipm, sipmMAU',
                compression = 'ZLIB4',
                n_sensors   : 'number of pmts or sipms',
                bin_centres : 'np.array of bin centres'):
    try:                       hist_group = getattr          (file.root, group_name)
    except tb.NoSuchNodeError: hist_group = file.create_group(file.root, group_name)

    n_bins = len(bin_centres)

    hist_table = file.create_earray(hist_group,
                                    table_name,
                                    atom    = tb.Int32Atom(),
                                    shape   = (0, n_sensors, n_bins),
                                    filters = tbl.filters(compression))

    ## The bins can be written just once at definition of the writer
    file.create_array(hist_group, table_name+'_bins', bin_centres)

    def write_hist(histo : 'np.array of histograms, one for each sensor'):
        hist_table.append(histo.reshape(1, n_sensors, n_bins))

    return write_hist
