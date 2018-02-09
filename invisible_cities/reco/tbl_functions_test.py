from pytest import mark

from . import tbl_functions as tbl


@mark.parametrize('filename first_evt'.split(),
                  (('dst_NEXT_v0_08_09_Co56_INTERNALPORTANODE_74_0_7bar_MCRD_10000.root.h5', 740000),
                   ('NEXT_v0_08_09_Co56_2_0_7bar_MCRD_1000.root.h5'                        ,   2000),
                   ('electrons_40keV_z250_MCRD.h5'                                         ,      0)))
def test_event_number_from_input_file_name(filename, first_evt):
    assert tbl.event_number_from_input_file_name(filename) == first_evt
