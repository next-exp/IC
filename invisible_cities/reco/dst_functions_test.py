from pytest import fixture, mark

import numpy  as np
import pandas as pd
import tables as tb

from invisible_cities.reco.dst_functions import load_dst, load_dsts
from invisible_cities.reco.tbl_functions import filters
from invisible_cities.reco.nh5           import KrTable


@mark.skip
def write_test_dst(df, filename, group, node):
    with tb.open_file(filename, "w") as h5in:
        group = h5in.create_group(h5in.root, group)
        table = h5in.create_table(group,
                                  "data",
                                  KrTable,
                                  "Test data",
                                  filters("ZLIB4"))

        tablerow = table.row
        for index, row in df.iterrows():
            for name, value in row.items():
                tablerow[name] = value
            tablerow.append()
        table.flush()


@fixture
def Kr_dst_data(ICDIR):
    data = {}
    data["event"] = np.array   ([1  ] * 3 + [2  ] + [6   ] * 2)
    data["time" ] = np.array   ([1e7] * 3 + [2e7] + [3e7 ] * 2)
    data["peak" ] = np.array   ([0, 1, 2] + [0  ] + [0, 1]    )
    data["nS2"  ] = np.array   ([3  ] * 3 + [1  ] + [2   ] * 2)
    data["S1w"  ] = np.linspace(100, 200, 6)
    data["S1h"  ] = np.linspace( 10,  60, 6)
    data["S1e"  ] = np.linspace(  0, 100, 6)
    data["S1t"  ] = np.linspace(100, 800, 6)

    data["S2w"  ] = np.linspace( 10,  17, 6)
    data["S2h"  ] = np.linspace(150, 850, 6)
    data["S2e"  ] = np.linspace(1e3, 8e3, 6)
    data["S2q"  ] = np.linspace(  0, 700, 6)
    data["S2t"  ] = np.linspace(200, 900, 6)

    data["Nsipm"] = np.arange  (  1,   7, 1)
    data["DT"   ] = np.linspace(100, 107, 6)
    data["Z"    ] = np.linspace(200, 207, 6)
    data["X"    ] = np.linspace(-55, +55, 6)
    data["Y"    ] = np.linspace(-95, +95, 6)
    data["R"    ] = (data["X"]**2 + data["Y"]**2)**0.5
    data["Phi"  ] = np.arctan2 (data["Y"], data["X"])
    data["Xrms" ] = np.linspace( 10,  70, 6)
    data["Yrms" ] = np.linspace( 20,  90, 6)
    df = pd.DataFrame(data)

    
    return (ICDIR + "/database/test_data/Kr_dst.h5", "DST", "data"), df


def test_load_dst(Kr_dst_data):
    (filename, group, node), df = Kr_dst_data
    load_dst(filename, group, node) == df


def test_load_dsts_single_file(Kr_dst_data):
    (filename, group, node), df = Kr_dst_data
    load_dsts([filename], group, node) == df


def test_load_dsts_double_file(Kr_dst_data):
    (filename, group, node), df = Kr_dst_data
    load_dsts([filename, filename], group, node) == pd.concat([df, df])
