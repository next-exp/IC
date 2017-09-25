import numpy as np

from hypothesis             import given
from hypothesis.strategies  import integers
from hypothesis.strategies  import booleans
from hypothesis.strategies  import dictionaries
from hypothesis.strategies  import composite

from .  s1s2_filter import S12SelectorOutput

@composite
def random_filter_output(draw):
    ok = draw(booleans())
    s1 = draw(dictionaries(keys     = integers(min_value=0, max_value=10),
                           values   = booleans(),
                           max_size = 5))
    s2 = draw(dictionaries(keys     = integers(min_value=0, max_value=10),
                           values   = booleans(),
                           max_size = 5))
    return S12SelectorOutput(ok, s1, s2)


@composite
def anded_filter(draw):
    filter_1 = draw(random_filter_output())
    filter_2 = draw(random_filter_output())

    ok_anded = filter_1.passed and filter_2.passed
    s1_anded = {}
    for peak_no in set(filter_1.s1_peaks) | set(filter_2.s1_peaks):
        s1_anded[peak_no] = (filter_1.s1_peaks.get(peak_no, False) and
                             filter_2.s1_peaks.get(peak_no, False))

    s2_anded = {}
    for peak_no in set(filter_1.s2_peaks) | set(filter_2.s2_peaks):
        s2_anded[peak_no] = (filter_1.s2_peaks.get(peak_no, False) and
                             filter_2.s2_peaks.get(peak_no, False))

    filter_anded = S12SelectorOutput(ok_anded, s1_anded, s2_anded)
    return filter_1, filter_2, filter_anded


@composite
def ored_filter(draw):
    filter_1 = draw(random_filter_output())
    filter_2 = draw(random_filter_output())

    ok_ored = filter_1.passed or filter_2.passed
    s1_ored = {}
    for peak_no in set(filter_1.s1_peaks) | set(filter_2.s1_peaks):
        s1_ored[peak_no] = (filter_1.s1_peaks.get(peak_no, False) or
                            filter_2.s1_peaks.get(peak_no, False))

    s2_ored = {}
    for peak_no in set(filter_1.s2_peaks) | set(filter_2.s2_peaks):
        s2_ored[peak_no] = (filter_1.s2_peaks.get(peak_no, False) or
                            filter_2.s2_peaks.get(peak_no, False))

    filter_ored = S12SelectorOutput(ok_ored, s1_ored, s2_ored)
    return filter_1, filter_2, filter_ored


@given(anded_filter())
def test_s12selectoroutput_and(filters):
    one, two, true = filters
    f_1 = S12SelectorOutput(one.passed, one.s1_peaks, one.s2_peaks)
    f_2 = S12SelectorOutput(two.passed, two.s1_peaks, two.s2_peaks)

    f_anded = f_1 & f_2
    assert true.passed   == f_anded.passed
    assert true.s1_peaks == f_anded.s1_peaks
    assert true.s2_peaks == f_anded.s2_peaks


@given(ored_filter())
def test_s12selectoroutput_or(filters):
    one, two, true = filters
    f_1 = S12SelectorOutput(one.passed, one.s1_peaks, one.s2_peaks)
    f_2 = S12SelectorOutput(two.passed, two.s1_peaks, two.s2_peaks)

    f_ored = f_1 | f_2
    assert true.passed   == f_ored.passed
    assert true.s1_peaks == f_ored.s1_peaks
    assert true.s2_peaks == f_ored.s2_peaks
