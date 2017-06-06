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
from .  isidora import Isidora


def test_isidora_no_input():
    """Test that Isidora throws NoInputFiles exceptions if no input files
       are defined. """

    with raises(NoInputFiles):
        fpp = Isidora(run_number = 0)
        fpp.run(nmax=1)
