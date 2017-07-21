
from .  corrections        import Correction
from .  dst_functions      import load_xy_corrections


def test_load_xy_corrections(corr_toy_data):
    filename, true_data = corr_toy_data
    x, y, E, U, _ = true_data
    corr          = load_xy_corrections(filename)
    assert corr == Correction((x,y), E, U)
