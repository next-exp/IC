import numpy as np

from . detsim_waveforms import generate_s1_waveform

from ..core.system_of_units import ns

def test_generate_s1_waveform_pes():
    "Test that total pes are conserved in s1 waveform"
    pes_at_pmts = np.array([[1, 2, 3, 4]])
    time = np.zeros(pes_at_pmts.shape[-1])
    buffer_length = 1000 * ns
    bin_width     = 25  * ns
    start_time    = 0

    wfs = generate_s1_waveform(pes_at_pmts, time, buffer_length, bin_width, start_time)
    np.testing.assert_allclose(np.sum(pes_at_pmts, axis=1), np.sum(wfs, axis=1), rtol=0.001)


def test_generate_s1_waveform_multi():
    "Test waveform creation for multiple s1s"
    pes_at_pmts = np.array([[1, 2, 3, 4, 1, 2, 3, 4]])
    time = np.zeros(pes_at_pmts.shape[-1])
    time[:-4] = 1000 * ns

    buffer_length = 2000 * ns
    bin_width     = 25  * ns
    start_time    = 0

    wfs = generate_s1_waveform(pes_at_pmts, time, buffer_length, bin_width, start_time)
    np.testing.assert_allclose( wfs[:, :int(buffer_length/bin_width/2)]
                              , wfs[:,  int(buffer_length/bin_width/2):], rtol=0.001)
