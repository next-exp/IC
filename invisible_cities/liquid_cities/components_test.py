from argparse import Namespace

from pytest import mark
from pytest import raises

from ..core.configure import EventRange as ER
from . components import event_range


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
