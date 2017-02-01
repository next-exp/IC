from __future__ import absolute_import

from pytest import mark
import sys, os

import invisible_cities.core.configure as conf

@mark.xfail
def test_configure():
    """Test configure function. Read from conf file.
    NB: this test will crash if file irene.conf in /config is modified.
    BUT, file should not be modified without a good reason since
    is is a default configuration.
    """
    CONF_file_def = os.environ['ICDIR']   + '/config/irene.conf'
    n = 10
    s = 0
    p = 1
    CFP = conf.configure(['program_name',
                     '-c', CONF_file_def,
                     '-i', 'rwf_file',
                     '-o', 'pmp_file',
                     '-n', n,
                     '-s', s,
                     '-p', p])

    assert CFP["COMPRESSION"] == 'ZLIB4'
    assert CFP["FILE_IN"] == 'rwf_file'
    assert CFP["FILE_OUT"] == 'pmp_file'
    assert CFP["INFO"] ==                   False
    assert CFP["NBASELINE"] ==              28000
    assert CFP["NEVENTS"] ==                10
    assert CFP["NMAU"] ==                   100
    assert CFP["NPRINT"] ==                 1
    assert CFP["PRINT_EMPTY_EVENTS"] ==     1
    assert CFP["RUN_ALL"] ==                False
    assert CFP["RUN_NUMBER"] ==             0
    assert CFP["S1_LMAX"] ==                20
    assert CFP["S1_LMIN"] ==                6
    assert CFP["S1_STRIDE"] ==              4
    assert CFP["S1_TMAX"] ==                590
    assert CFP["S1_TMIN"] ==                10
    assert CFP["S2_LMAX"] ==                100000
    assert CFP["S2_LMIN"] ==                100
    assert CFP["S2_STRIDE"] ==              40
    assert CFP["S2_TMAX"] ==                1190
    assert CFP["S2_TMIN"] ==                0
    assert CFP["SKIP"] ==                   0
    assert CFP["THR_CSUM"] ==               0.5
    assert CFP["THR_MAU"] ==                3
    assert CFP["THR_SIPM_S2"] ==            30
    assert CFP["THR_TRIGGER"] ==            5
    assert CFP["THR_ZS"] ==                 20
    assert CFP["VERBOSITY"] ==              20
