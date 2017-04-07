import abc
import tables as tb

import invisible_cities.reco.nh5 as table_formats
import invisible_cities.reco.tbl_functions as tbf


class DST_writer:

    def __init__(self,
                 filename,
                 group       = "DST",
                 mode        = "w",
                 compression = "ZLIB4"):
        self._hdf5_file  = tb.open_file(filename, mode)
        self.group       = group
        self.mode        = mode
        self.compression = compression

    @abc.abstractmethod
    def __call__(self, *args):
        pass

    def close(self):
        self._hdf5_file.close()

    @property
    def file(self):
        return self._hdf5_file

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class Kr_writer(DST_writer):
    def __init__(self,
                 filename,
                 group = "DST",
                 mode = "w",
                 compression = "ZLIB4",

                 table_name = "Events",
                 table_doc  = None):
        DST_writer.__init__(self,
                            filename,
                            group,
                            mode,
                            compression)

        self.table_name = table_name
        self.table_doc  = table_name if table_doc is None else table_doc
        self.table      = self._make_table()
        self.row        = self.table.row


    def _make_table(self):
        return _make_table(self.file,
                           self.group,
                           self.table_name,
                           table_formats.KrTable,
                           self.compression,
                           self.table_doc)

    def __call__(self, evt):
        KrEvent(evt).store(self.row)


def _make_table(hdf5_file, group, name, format, compression, description):

    dst_group  = hdf5_file.create_group(hdf5_file.root, group) 
    table = hdf5_file.create_table(dst_group,
                                   name,
                                   format,
                                   description,
                                   tbf.filters(compression))
    table.cols.event.create_index()

    return table


class PointLikeEvent:
    def __init__(self, other = None):
        if other is not None:
            self.copy(other)
            return
        self.evt   = -1
        self.T     = -1

        self.nS1   = -1
        self.S1w   = []
        self.S1h   = []
        self.S1e   = []
        self.S1t   = []

        self.nS2   = -1
        self.S2w   = []
        self.S2h   = []
        self.S2e   = []
        self.S2q   = []
        self.S2t   = []

        self.Nsipm = []
        self.DT    = []
        self.Z     = []
        self.X     = []
        self.Y     = []
        self.R     = []
        self.Phi   = []
        self.Xrms  = []
        self.Yrms  = []

    def copy(self, other):
        assert isinstance(other, PointLikeEvent)
        for attr in other.__dict__:
            setattr(self, attr, getattr(other, attr))

    @abc.abstractmethod
    def store(self, *args, **kwargs):
        pass


class KrEvent(PointLikeEvent):
    def store(self, row):
        for i in range(self.nS2):
            row["event"] = self.event
            row["time" ] = self.time
            row["peak" ] = i
            row["nS2"  ] = self.nS2

            row["S1w"  ] = self.S1w  [0]
            row["S1h"  ] = self.S1h  [0]
            row["S1e"  ] = self.S1e  [0]
            row["S1t"  ] = self.S1t  [0]

            row["S2w"  ] = self.S2w  [i]
            row["S2h"  ] = self.S2h  [i]
            row["S2e"  ] = self.S2e  [i]
            row["S2q"  ] = self.S2q  [i]
            row["S2t"  ] = self.S2t  [i]

            row["Nsipm"] = self.Nsipm[i]
            row["DT"   ] = self.DT   [i]
            row["Z"    ] = self.Z    [i]
            row["X"    ] = self.X    [i]
            row["Y"    ] = self.Y    [i]
            row["R"    ] = self.R    [i]
            row["Phi"  ] = self.Phi  [i]
            row["Xrms" ] = self.Xrms [i]
            row["Yrms" ] = self.Yrms [i]
            row.append()
