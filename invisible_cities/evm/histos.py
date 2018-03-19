import numpy  as np
import tables as tb

from .. reco  import tbl_functions as tbl


class Histogram:
    def __init__(self, title, bins, labels, values=None):
        """
        This class represents a histogram with is a parameter holder that
        contains data grouped by bins.

        Attributes:
        title     = String with the histogram title
        bins      = List with the histogram binning.
        data      = Array with the accumulated entries on each bin.
        out_range = Array with the Accumulated counts out of the bin range.
                    Values are n-dim arrays of lenght 2 (first element is
                    underflow, second oveflow).
        errors    = Array with the assigned uncertanties to each bin.
        labels    = List with the axis labels.

        Arguments:
        bins   = List containing the histogram binning.
        values = Array with initial values, optional.
                 If not passed, then the initial bin content is set to zero.
        """
        self.title     = title
        self.bins      = bins
        self.data      = self.init_from_bins()
        self.errors    = self.init_from_bins()
        self.out_range = np.zeros(shape=(2, len(self.bins)))
        self.labels    = labels

        if values is not None:
            self.fill(np.asarray(values))

    def init_from_bins(self):
        "Encapsulation for histogram initialization to 0"
        return np.zeros(shape=tuple(len(x) - 1 for x in self.bins))

    def fill(self, additive, data_weights=None):
        """
        Given datapoints, bins and adds thems to the stored bin content.

        Arguments:
        additive     = Array or list with data to fill the histogram.
        data_weights = Array or list with weights of the data.
        """
        additive     = np.array(additive)
        data_weights = np.ones(len(additive.T)) if data_weights is None else np.array(data_weights)
        if len(data_weights) != len(additive.T):
            raise ValueError("Dimensions of data and weights is not compatible.")

        binnedData, outRange = self.bin_data(additive, data_weights)

        self.data      += binnedData
        self.out_range += outRange
        self.update_errors()

    def bin_data(self, data, data_weights):
        """
        Bins the given data and computes the events out of range.

        Arguments:
        data         = Array with the data to be binned.
        data_weights = Array with weights for the data points.
        """
        binned_data, *_ = np.histogramdd(data.T, self.bins, weights=data_weights)
        out_of_range    = self.count_out_of_range(np.array(data, ndmin=2))

        return binned_data, out_of_range

    def count_out_of_range(self, data):
        """
        Returns an array with the number of events out of the Histogram's bin
        range of the given data.

        Arguments:
        data = Array with the data.
        """
        out_of_range = []
        for i, bins in enumerate(self.bins):
            lower_limit = bins[0]
            upper_limit = bins[-1]
            out_of_range.append([np.count_nonzero(data[i] < lower_limit),
                                 np.count_nonzero(data[i] > upper_limit)])

        return np.asarray(out_of_range).T

    def update_errors(self, errors=None):
        """
        Updates the errors with the passed list/array. If nothing is passed,
        then the square root of the counts is computed and assigned as error.

        Arguments:
        errors = List or array of errors.
        """
        self.errors = np.asarray(errors) if errors is not None else np.sqrt(self.data)

    def _check_valid_binning(self, bins):
        if len(self.bins) != len(bins) or not np.all(a == b for a, b in zip(self.bins, bins)):
            raise ValueError("Histogram binning is not compatible")

    def __radd__(self, other):
        return self + other

    def __add__ (self, other):
        if other is None:
            return self
        self._check_valid_binning(other.bins)
        if self.title  !=  other.title:
            print(f"""Warning: Histogram titles are different.
                      {self.title}, {other.title}""")
        if self.labels != other.labels:
            print(f"""Warning: Histogram titles are different.
                      {self.labels}, {other.labels}""")
        new_histogram           = Histogram(self.title, self.bins, self.labels)
        new_histogram.data      =           self.data        + other.data
        new_histogram.out_range =           self.out_range   + other.out_range
        new_histogram.errors    = np.sqrt  (self.errors ** 2 + other.errors ** 2)
        return new_histogram


class HistoManager:
    def __init__(self, histograms=None):
        """
        This class is a parameter holder that contains a dictionary
        of Histogram objects.

        Attributes:
        histos = Dictionary holding Histogram objects.
                 Keys are the Histograms' name.

        Arguments:
        histograms = List with the initial Histogram objects.
        """
        self.histos = {}

        if histograms is not None:
            values = histograms.values() if isinstance(histograms, dict) else iter(histograms)
            for histogram in values:
                self.new_histogram(histogram)

    def new_histogram(self, histogram):
        """
        Adds a new Histogram to the HistoManager.

        Arguments:

        histogram = Histogram object.
        """
        self[histogram.title] = histogram

    def fill_histograms(self, additives):
        """
        Fills several Histograms of the Histomanager.

        Arguments:
        additives: Dictionary with keys equal to the Histograms names.
                   Values are the data to fill the Histogram.
        """
        for histoname, additive in additives.items():
            if histoname in self.histos:
                self[histoname].fill(np.asarray(additive))
            else:
                print(f"Histogram with name {histoname} does not exist")

    def __getitem__(self, histoname):
        return self.histos[histoname]

    def __setitem__(self, histoname, histogram):
        self.histos[histoname] = histogram
