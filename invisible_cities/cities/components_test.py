import os

import numpy as np

from argparse  import Namespace
from functools import partial

from pytest import mark
from pytest import raises

from .. core.configure  import EventRange as ER
from .. core.exceptions import InvalidInputFileStructure
from .. core.exceptions import ClusterEmptyList

from .. core.system_of_units_c import units

from .  components import event_range
from .  components import WfType
from .  components import   wf_from_files
from .  components import pmap_from_files
from .  components import compute_xy_position

from .. database import load_db


def _create_dummy_conf_with_event_range(value):
    return Namespace(event_range = value)


@mark.parametrize("given expected".split(),
                  ((       9          , (   9,     )),
                   ( (     9,        ), (   9,     )),
                   ( (     5,       9), (   5,    9)),
                   ( (     5, ER.last), (   5, None)),
                   (  ER.all          , (None,     )),
                   ( (ER.all,        ), (None,     ))))
def test_event_range_valid_options(given, expected):
    conf = _create_dummy_conf_with_event_range(given)
    assert event_range(conf) == expected


@mark.parametrize("given",
                  ( ER.last    ,
                   (ER.last,)  ,
                   (ER.last, 4),
                   (ER.all , 4),
                   ( 1,  2,  3)))

def test_event_range_invalid_options_raises_ValueError(given):
    conf = _create_dummy_conf_with_event_range(given)
    with raises(ValueError):
        event_range(conf)


_rwf_from_files = partial(wf_from_files, wf_type=WfType.rwf)
@mark.parametrize("source filename".split(),
                  ((_rwf_from_files, "defective_rwf_rd_pmtrwf.h5"      ),
                   (_rwf_from_files, "defective_rwf_rd_sipmrwf.h5"     ),
                   (_rwf_from_files, "defective_rwf_run_events.h5"     ),
                   (_rwf_from_files, "defective_rwf_trigger_events.h5" ),
                   (_rwf_from_files, "defective_rwf_trigger_trigger.h5"),
                   (pmap_from_files, "defective_pmp_pmap_all.h5"       ),
                   (pmap_from_files, "defective_pmp_run_events.h5"     )))
def test_sources_invalid_input_raises_InvalidInputFileStructure(ICDATADIR, source, filename):
    full_filename = os.path.join(ICDATADIR, "defective_files", filename)
    s = source((full_filename,))
    with raises(InvalidInputFileStructure):
        next(s)


def test_compute_xy_position_depend_on_masked_channels():
    minimum_seed_charge = 6*units.pes
    reco_parameters = {'Qthr': 2*units.pes,
                       'Qlm': minimum_seed_charge,
                       'lm_radius': 0*units.mm,
                       'new_lm_radius': 15 * units.mm,
                       'msipm': 9,
                       'consider_masked': True}
    run_number = 6977
    find_xy_pos = compute_xy_position('new', run_number, **reco_parameters)

    # masked_channel = 11009, x_pos = -65, y_pos = 15
    # the channels entering the reco algorithm are the ones in a square of 3x3
    # that includes the masked channel. The test is supposed to fail if the
    # compute_xy_position function doesn't use the run_number parameter.
    xs_to_test  = np.array([-65, -65, -55, -55, -55, -45, -45, -45])
    ys_to_test  = np.array([  5,  25,   5,  15,  25,   5,  15,  25])
    xys_to_test = np.stack((xs_to_test, ys_to_test), axis=1)

    charge         = minimum_seed_charge - 1
    seed_charge    = minimum_seed_charge + 1
    charge_to_test = np.array([charge, charge, charge, seed_charge, charge, charge, charge, charge])

    try:
        find_xy_pos(xys_to_test, charge_to_test)
    except(ClusterEmptyList):
        assert False
