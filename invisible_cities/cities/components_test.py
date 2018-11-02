import os

from argparse  import Namespace
from functools import partial

from pytest import mark
from pytest import raises

from .. core.configure  import EventRange as ER
from .. core.exceptions import InvalidInputFileStructure

from .  components import event_range
from .  components import WfType
from .  components import   wf_from_files
from .  components import pmap_from_files


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
