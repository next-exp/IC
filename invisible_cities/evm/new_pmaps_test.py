from hypothesis             import assume
from hypothesis.strategies  import integers
from hypothesis.strategies  import floats
from hypothesis.strategies  import sampled_from
from hypothesis.strategies  import composite
from hypothesis.extra.numpy import arrays

from . new_pmaps import  PMTResponses
from . new_pmaps import SiPMResponses
from . new_pmaps import S1
from . new_pmaps import S2
from . new_pmaps import PMap


wf_min =   0
wf_max = 100


@composite
def sensor_responses(draw, nsamples=None, type_=None):
    nsensors    = draw(integers(1,  5))
    nsamples    = draw(integers(1, 50)) if nsamples is None else nsamples
    shape       = nsensors, nsamples
    ids         = draw(arrays(  int, nsensors, integers(0, 1e3), unique=True))
    all_wfs     = draw(arrays(float,    shape, floats  (wf_min, wf_max)))
    PMT_or_SiPM =(draw(sampled_from((PMTResponses, SiPMResponses)))
                  if type_ is None else type_)
    args        = ids, all_wfs
    return args, PMT_or_SiPM(*args)


@composite
def peaks(draw, type_=None):
    nsamples  = draw(integers(1, 50))
    _, pmt_r  = draw(sensor_responses(nsamples,  PMTResponses))
    _, sipm_r = draw(sensor_responses(nsamples, SiPMResponses))
    assume(pmt_r.sum_over_sensors[ 0] != 0)
    assume(pmt_r.sum_over_sensors[-1] != 0)
    times     = draw(arrays(float, nsamples,
                            floats(min_value=0, max_value=1e3),
                            unique = True))
    S1_or_S2  = draw(sampled_from((S1, S2))) if type_ is None else type_
    args      = times, pmt_r, sipm_r
    return args, S1_or_S2(*args)


@composite
def pmaps(draw):
    n_s1 = draw(integers(0, 3))
    n_s2 = draw(integers(0, 3))
    s1s  = tuple(draw(peaks(S1))[1] for i in range(n_s1))
    s2s  = tuple(draw(peaks(S2))[1] for i in range(n_s2))
    args = s1s, s2s
    return args, PMap(*args)
