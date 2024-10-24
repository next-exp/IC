from pytest import mark
from pytest import raises

from . import tbl_functions as tbl


@mark.parametrize("name", "NOCOMPR ZLIB1 ZLIB4 ZLIB5 ZLIB9 BLOSC5 BLZ4HC5".split())
def test_filters(name):
    assert tbl.filters(name) is not None


def test_filters_None():
    assert tbl.filters(None) is None


def test_filters_raises():
    with raises(ValueError):
        tbl.filters("this_filter_does_not_exist")


"""
    if name is None: return None

    try:
        level, lib = {"NOCOMPR": (0,  None)        ,
                      "ZLIB1"  : (1, 'zlib')       ,
                      "ZLIB4"  : (4, 'zlib')       ,
                      "ZLIB5"  : (5, 'zlib')       ,
                      "ZLIB9"  : (9, 'zlib')       ,
                      "BLOSC5" : (5, 'blosc')      ,
                      "BLZ4HC5": (5, 'blosc:lz4hc'),
                      }[name]
        return tb.Filters(complevel=level, complib=lib)
    except KeyError:
        raise ValueError("Compression option {} not found.".format(name))
"""

@mark.parametrize('filename first_evt'.split(),
                  (('dst_NEXT_v0_08_09_Co56_INTERNALPORTANODE_74_0_7bar_MCRD_10000.root.h5', 740000),
                   ('NEXT_v0_08_09_Co56_2_0_7bar_MCRD_1000.root.h5'                        ,   2000),
                   ('electrons_40keV_z250_MCRD.h5'                                         ,      0)))
def test_event_number_from_input_file_name(filename, first_evt):
    assert tbl.event_number_from_input_file_name(filename) == first_evt
