from .  table_io import make_table
from .. evm.nh5  import KrTable
from .. evm.nh5  import XYfactors
from .. evm.nh5  import PSFfactors


def kr_writer(hdf5_file, *, compression='ZLIB4'):
    kr_table = make_table(hdf5_file,
                          group       = 'DST',
                          name        = 'Events',
                          fformat     = KrTable,
                          description = 'KDST Events',
                          compression = compression)
    # Mark column to index after populating table
    kr_table.set_attr('columns_to_index', ['event'])

    def write_kr(kr_event):
        kr_event.store(kr_table)
    return write_kr


def xy_writer(hdf5_file, **kwargs):
    xy_table = make_table(hdf5_file,
                          fformat = XYfactors,
                          **kwargs)

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


def xy_correction_writer(hdf5_file, * ,
                         group       = "Corrections",
                         table_name  = "XYcorrections",
                         compression = 'ZLIB4'):
    return xy_writer(hdf5_file,
                     group        = group,
                     name         = table_name,
                     description  = "XY corrections",
                     compression  = compression)


def xy_lifetime_writer(hdf5_file, * ,
                       group       = "Corrections",
                       table_name  = "LifetimeXY",
                       compression = 'ZLIB4'):
    return xy_writer(hdf5_file,
                     group        = group,
                     name         = table_name,
                     description  = "XY-dependent lifetime values",
                     compression  = compression)

def psf_writer(hdf5_file, **kwargs):
    psf_table = make_table(hdf5_file,
                           group       = "PSF",
                           name        = "PSFs",
                           fformat     = PSFfactors,
                           description = "XYZ dependent point spread functions",
                           compression = 'ZLIB4')

    def write_psf(xr, yr, zr, xp, yp, zp, fr, nr):
        row = psf_table.row
        for i, x in enumerate(xr):
            for j, y in enumerate(yr):
                for k, z in enumerate(zr):
                    row["xr"    ] = x
                    row["yr"    ] = y
                    row["zr"    ] = z
                    row["x"     ] = xp
                    row["y"     ] = yp
                    row["z"     ] = zp
                    row["factor"] = fr[i,j,k]
                    row["nevt"  ] = nr[i,j,k]
                    row.append()
    return write_psf
