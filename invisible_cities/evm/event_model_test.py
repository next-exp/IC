from pytest import mark

from .. types.symbols import HitEnergy


@mark.parametrize("value", "E Ec Ep".split())
def test_hitenergy_value(value):
    assert getattr(HitEnergy, value).value == value
