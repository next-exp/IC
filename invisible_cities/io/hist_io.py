import numpy  as np
import tables as tb

from hypothesis    import settings
from hypothesis    import HealthCheck
from hypothesis    import given

from .. reco       import tbl_functions as tbl

from .. evm.histos import HistoManager
from .. evm.histos import Histogram


def hist_writer(file,
                *,
                group_name  : 'options: HIST, HIST2D',
                table_name  : 'options: pmt, pmtMAU, sipm, sipmMAU',
                compression = 'ZLIB4',
                n_sensors   : 'number of pmts or sipms',
                n_bins      : 'length of bin range used',
                bin_centres : 'np.array of bin centres'):
    try:                       hist_group = getattr          (file.root, group_name)
    except tb.NoSuchNodeError: hist_group = file.create_group(file.root, group_name)

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

def hist_writer_var(file, *, compression='ZLIB4'):

    def write_hist(group_name  : 'string with folder name to save histograms',
                   table_name  : 'histogram name'                            ,
                   entries     : 'np.array with bin content'                 ,
                   bins        : 'list of np.array of bins'                  ,
                   out_of_range: 'np.array lenght=2 with events out of range',
                   errors      : 'np.array with bins uncertainties'          ,
                   labels      : 'list with labels of the histogram'         ):

        try:                       hist_group = getattr          (file.root, group_name)
        except tb.NoSuchNodeError: hist_group = file.create_group(file.root, group_name)

        if table_name in hist_group:
            raise ValueError(f"Histogram {table_name} already exists")

        vlarray = file.create_vlarray(hist_group, table_name + '_bins',
                                      atom    = tb.Float64Atom(shape=()),
                                      filters = tbl.filters(compression))
        for ibin in bins:
            vlarray.append(ibin)
        add_carray       (hist_group, table_name              , entries     )
        add_carray       (hist_group, table_name + '_outRange', out_of_range)
        add_carray       (hist_group, table_name + '_errors'  , errors      )
        file.create_array(hist_group, table_name + '_labels'  , labels      )

    def add_carray(hist_group, table_name, var):
        array_atom  = tb  .Atom.from_dtype(var.dtype)
        array_shape = var .shape
        entry       = file.create_carray(hist_group, table_name,
                                         atom    = array_atom,
                                         shape   = array_shape,
                                         filters = tbl.filters(compression))
        entry[:]    = var

    return write_hist


def save_histomanager_to_file(histogram_manager, file_out, mode='w', group='HIST'):
    """
    Saves the HistoManager and its contained Histograms to a file.

    Arguments:
    file_out = String with the path of the file were the HistoManager will
               be written.
    mode     = Writting mode. By default a new file will be created.
    group    = Group name to save the histograms in the file.
    """
    if mode not in 'wa':
        raise ValueError(f"Incompatible mode ({mode}) of writting, please use 'w' (write) or 'a' (append).")
    with tb.open_file(file_out, mode, filters=tbl.filters('ZLIB4')) as h5out:
        writer = hist_writer_var(h5out)
        for histoname, histo in histogram_manager.histos.items():
            writer(group, histoname,
                   histo.data, histo.bins, histo.out_range,
                   histo.errors, histo.labels)


def get_histograms_from_file(file_input, group_name='HIST'):
    histo_manager = HistoManager()

    def name_selection(x):
        selection = (   ('bins'     not in x)
                    and ('labels'   not in x)
                    and ('errors'   not in x)
                    and ('outRange' not in x))
        return selection

    with tb.open_file(file_input, "r") as h5in:
        histogram_list = []
        group = getattr(h5in.root, group_name)
        for histoname in filter(name_selection, group._v_children):
            entries   = np.array(getattr(group, histoname              )[:])
            bins      =          getattr(group, histoname + '_bins'    )[:]
            out_range =          getattr(group, histoname + '_outRange')[:]
            errors    = np.array(getattr(group, histoname + '_errors'  )[:])
            labels    =          getattr(group, histoname + '_labels'  )[:]
            labels    = [str(lab)[2:-1].replace('\\\\', '\\') for lab in labels]

            histogram           = Histogram(histoname, bins, labels)
            histogram.data      = entries
            histogram.out_range = out_range
            histogram.errors    = errors

            histogram_list.append(histogram)

    return HistoManager(histogram_list)
