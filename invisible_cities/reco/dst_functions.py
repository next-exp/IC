import pandas as pd
import tables as tb

from params import Correction

def load_dst(filename, group, node):
    with tb.open_file(filename) as h5in:
        table = getattr(getattr(h5in.root, group), node).read()
        return pd.DataFrame.from_records(table)


def load_dsts(dst_list, group, node):
    dsts = [load_dst(filename, group, node) for filename in dst_list]
    return pd.concat(dsts)


def load_z_corrections(filename):
    dst = load_dst(filename, "Corrections", "Zcorrections")
    return Correction(dst.z, dst.factor, dst.uncertainty)


def load_xy_corrections(filename):
    dst = load_dst(filename, "Corrections", "XYcorrections")
    return Correction((dst.x, dst.y), dst.factor, dst.uncertainty)
