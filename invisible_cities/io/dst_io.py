import abc

import tables as tb

from .. reco import tbl_functions as tbl
from .. reco import nh5           as table_formats


def kr_writer(file, *, compression='ZLIB4'):
    kr_table = _make_kr_tables(file, compression)
    def write_kr(kr_event):
        kr_event.store(kr_table)
    return write_kr

def _make_kr_tables(hdf5_file, compression):
    c = tbl.filters(compression)
    dst_group = hdf5_file.create_group(hdf5_file.root, 'DST')
    events_table = hdf5_file.create_table(
        dst_group, 'Events', table_formats.KrTable, 'Events Table', c)
    return events_table


def xy_writer(file, *, compression='ZLIB4'):
    xy_table = _make_xy_tables(file, compression)
    def write_xy(xs, ys, fs, us, ns):
        row = xy_table.row
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                row["x"]           = x
                row["y"]           = y
                row["factor"]      = fs[i,j]
                row["uncertainty"] = us[i,j]
                row["nevt"]        = ns[i,j]
                row.append()
    return write_xy

def _make_xy_tables(hdf5_file, compression):
    c = tbl.filters(compression)
    xy_group = hdf5_file.create_group(hdf5_file.root, 'Corrections')
    xy_table = hdf5_file.create_table(
        xy_group, 'XYcorrections', table_formats.XYfactors, 'Correction in the x,y coordinates', c)
    return xy_table



def _make_table(hdf5_file, group, name, format, compression, description):
    if group not in hdf5_file.root:
        hdf5_file.create_group(hdf5_file.root, group)
    table = hdf5_file.create_table(getattr(hdf5_file.root, group),
                                   name,
                                   format,
                                   description,
                                   tbl.filters(compression))
    return table


class PointLikeEvent:
    def __init__(self):
        self.evt   = -1
        self.T     = -1

        self.nS1   = -1
        # Consider replacing with a list of namedtuples or a
        # structured array
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

    def __str__(self):
        s = "{0}Event\n{0}".format("#"*20 + "\n")
        for attr in self.__dict__:
            s += "{}: {}\n".format(attr, getattr(self, attr))
        return s


class KrEvent(PointLikeEvent):
    def store(self, table):
        row = table.row
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

def write_test_dst(df, filename, group, node):
    with tb.open_file(filename, "w") as h5in:
        group = h5in.create_group(h5in.root, group)
        table = h5in.create_table(group,
                                  "data",
                                  table_formats.KrTable,
                                  "Test data",
                                  tbl.filters("ZLIB4"))

        tablerow = table.row
        for index, row in df.iterrows():
            for name, value in row.items():
                tablerow[name] = value
            tablerow.append()
        table.flush()
