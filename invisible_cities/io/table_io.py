
from .. reco import tbl_functions as tbl

def make_table(hdf5_file,
               group, name, fformat, description, compression):
    if group not in hdf5_file.root:
        hdf5_file.create_group(hdf5_file.root, group)
    table = hdf5_file.create_table(getattr(hdf5_file.root, group),
                                   name,
                                   fformat,
                                   description,
                                   tbl.filters(compression))
    return table
