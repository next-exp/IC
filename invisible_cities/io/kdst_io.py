from .  table_io import make_table
from .. reco.nh5 import KrTable
from .. reco.nh5 import XYfactors


def kr_writer(hdf5_file, *, compression='ZLIB4'):
    kr_table = make_table(hdf5_file,
                          group       = 'DST',
                          name        = 'Events',
                          fformat     = KrTable,
                          description = 'KDST Events',
                          compression = compression)

    def write_kr(kr_event):
        kr_event.store(kr_table)
    return write_kr


def xy_writer(hdf5_file, *, compression='ZLIB4'):
    xy_table = make_table(hdf5_file,
                          group       = 'Corrections',
                          name        = 'XYcorrections',
                          fformat     = XYfactors,
                          description = 'x,y corrections',
                          compression = compression)

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
