from hypothesis import given
from hypothesis.extra.numpy import arrays, np

from .wfmFunctions import subtract_baseline

@given(arrays(float, (10,20)))
def test_subtract_baseline(waveforms):
    subtract_baseline(waveforms)
    assert 'Did not crash'
