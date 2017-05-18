import abc
import copy

import tables as tb

from . import nh5           as table_formats
from . import tbl_functions as tbf


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
                 mode  = "w",
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
        self.table.cols.event.create_index()
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


class XYcorr_writer(DST_writer):
    def __init__(self,
                 filename,
                 group = "Corrections",
                 mode  = "w",
                 compression = "ZLIB4"):
        DST_writer.__init__(self,
                            filename,
                            group,
                            mode,
                            compression)

        self.table = self._make_table()

    def _make_table(self):
        xy_table = _make_table(self.file,
                               self.group,
                               "XYcorrections",
                               table_formats.XYfactors,
                               self.compression,
                               "Correction in the x,y coordinates")
        return xy_table

    def __call__(self, xs, ys, fs, us, ns):
        row = self.table.row
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                row["x"]           = x
                row["y"]           = y
                row["factor"]      = fs[i,j]
                row["uncertainty"] = us[i,j]
                row["nevt"]        = ns[i,j]
                row.append()



class HitCollection_writer(DST_writer):
    def __init__(self,
                 filename,
                 group = "DST",
                 mode  = "w",
                 compression = "ZLIB4",

                 table_name = "Tracks",
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
                           table_formats.TrackTable,
                           self.compression,
                           self.table_doc)

    def __call__(self, track):
        track.store(self.row)


def _make_table(hdf5_file, group, name, format, compression, description):
    if group not in hdf5_file.root:
        hdf5_file.create_group(hdf5_file.root, group)
    table = hdf5_file.create_table(getattr(hdf5_file.root, group),
                                   name,
                                   format,
                                   description,
                                   tbf.filters(compression))
    return table


class Event:
    def __init__(self):
        self.evt  = -1
        self.time = -1

    def __str__(self):
        s = "{0}Event\n{0}".format("#"*20 + "\n")
        for attr in self.__dict__:
            s += "{}: {}\n".format(attr, getattr(self, attr))
        return s

    def copy(self, other):
        assert isinstance(other, Event)
        for attr in other.__dict__:
            setattr(self, attr, copy.deepcopy(getattr(other, attr)))

    @abc.abstractmethod
    def store(self, *args, **kwargs):
        pass


class PointLikeEvent(Event):
    def __init__(self):
        Event.__init__(self)

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



class KrEvent(PointLikeEvent):
    def __init__(self, other = None):
        if other is not None:
            self.copy(other)
            return
        PointLikeEvent.__init__(self)

    def store(self, row):
        for i in range(int(self.nS2)):
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


class Hit:
    def __init__(self):
        self.Npeak = -1
        self.X     = -1e12
        self.Y     = -1e12
        self.R     = -1e12
        self.Phi   = -1e12
        self.Z     = -1
        self.E     = -1
        self.Q     = -1
        self.Nsipm = -1


class HitCollection(list, Event):
    def __init__(self):
        list .__init__(self)
        Event.__init__(self)

    def store(self, row):
        for hit in self:
            row["event"] = self.evt
            row["time" ] = self.time

            row["npeak"] = hit.Npeak
            row["X"    ] = hit.X
            row["Y"    ] = hit.Y
            row["Z"    ] = hit.Z
            row["R"    ] = hit.R
            row["Phi"  ] = hit.Phi
            row["Nsipm"] = hit.Nsipm
            row["Q"    ] = hit.Q
            row["E"    ] = hit.E
            row["Ecorr"] = hit.Ecorr

            row.append()


def write_test_dst(df, filename, group, node):
    with tb.open_file(filename, "w") as h5in:
        group = h5in.create_group(h5in.root, group)
        table = h5in.create_table(group,
                                  "data",
                                  table_formats.KrTable,
                                  "Test data",
                                  tbf.filters("ZLIB4"))

        tablerow = table.row
        for index, row in df.iterrows():
            for name, value in row.items():
                tablerow[name] = value
            tablerow.append()
        table.flush()
