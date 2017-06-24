"""
code: isidora_test.py
description: test suite for isidora
author: J.J. Gomez-Cadenas
IC core team: Jacek Generowicz, JJGC,
G. Martinez, J.A. Hernando, J.M Benlloch
package: invisible cities. See release notes and licence
last changed: 01-12-2017
"""

from   pytest import raises

from .. core.exceptions import NoInputFiles
from .. core.exceptions import NoOutputFile
from .  base_cities     import City


def test_no_input_files_raises_NoInputFiles():
    with raises(NoInputFiles):
        City(file_out = 'dummy/output/file')

def test_no_output_files_raises_NoOutptFiles():
    with raises(NoOutputFile):
        City(files_in = 'dummy/input/files')
