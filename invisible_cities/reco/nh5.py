"""Tables defining the DM."""
import tables as tb


class RunInfo(tb.IsDescription):
    run_number = tb.Int32Col(shape=(), pos=0)


class EventInfo(tb.IsDescription):
    evt_number = tb. Int32Col(shape=(), pos=0)
    timestamp  = tb.UInt64Col(shape=(), pos=1)


class DetectorGeometry(tb.IsDescription):
    """Store geometry information for the detector."""
    x_det = tb.Float32Col(pos=1, shape=2)  # xmin, xmax
    y_det = tb.Float32Col(pos=2, shape=2)  # ymin, ymax
    z_det = tb.Float32Col(pos=3, shape=2)  # zmin, zmax
    r_det = tb.Float32Col(pos=4)  # radius


class DataSensor(tb.IsDescription):
    """Store metadata information for the SiPMs (position, gain,
    calibration-constant, mask).
    """
    channel    = tb.  Int32Col(pos=0) # electronic channel
    position   = tb.Float32Col(pos=1, shape=3)
    coeff      = tb.Float64Col(pos=2)
    adc_to_pes = tb.Float32Col(pos=3)
    noise_rms  = tb.Float32Col(pos=4)


class MCTrack(tb.IsDescription):
    """Stores. the parameters used by the simulation as metadata using
    Pytables.
    """
    event_indx     = tb.  Int32Col(    pos= 1)
    mctrk_indx     = tb.  Int16Col(    pos= 2)
    particle_name  = tb. StringCol(10, pos= 3)
    pdg_code       = tb.  Int16Col(    pos= 4)
    initial_vertex = tb.Float32Col(    pos= 5, shape=3)
    final_vertex   = tb.Float32Col(    pos= 6, shape=3)
    momentum       = tb.Float32Col(    pos= 7, shape=3)
    energy         = tb.Float32Col(    pos= 8)
    nof_hits       = tb.  Int16Col(    pos= 9)
    hit_indx       = tb.  Int16Col(    pos=10)
    hit_position   = tb.Float32Col(    pos=11, shape=3)
    hit_time       = tb.Float32Col(    pos=12)
    hit_energy     = tb.Float32Col(    pos=13)


class SENSOR_WF(tb.IsDescription):
    """Describe a true waveform (zero supressed)."""
    event    = tb.UInt32Col (pos=0)
    ID       = tb.UInt32Col (pos=1)
    time_mus = tb.Float32Col(pos=2)
    ene_pes  = tb.Float32Col(pos=3)


class FEE(tb.IsDescription):
    """Store the parameters used by the EP simulation as metadata."""
    OFFSET        = tb.  Int16Col(pos= 1)  # displaces the baseline (e.g, 700)
    CEILING       = tb.  Int16Col(pos= 2)  # adc top count (4096)
    PMT_GAIN      = tb.Float32Col(pos= 3)  # Gain of PMT (4.5e6)
    FEE_GAIN      = tb.Float32Col(pos= 4)  # FE gain (250*ohm)
    R1            = tb.Float32Col(pos= 5)  # resistor in Ohms (2350*ohm)
    C1            = tb.Float32Col(pos= 6)  # Capacitor C1 in nF
    C2            = tb.Float32Col(pos= 7)  # Capacitor C2 in nF
    ZIN           = tb.Float32Col(pos= 8)  # equivalent impedence
    DAQ_GAIN      = tb.Float32Col(pos= 9)
    NBITS         = tb.Float32Col(pos=10)  # number of bits ADC
    LSB           = tb.Float32Col(pos=11)  # LSB (adc count)
    NOISE_I       = tb.Float32Col(pos=12)  # Noise at the input
    NOISE_DAQ     = tb.Float32Col(pos=13)  # Noise at DAQ
    t_sample      = tb.Float32Col(pos=14)  # sampling time
    f_sample      = tb.Float32Col(pos=15)  # sampling frequency
    f_mc          = tb.Float32Col(pos=16)  # sampling frequency in MC (1ns)
    f_LPF1        = tb.Float32Col(pos=17)  # LPF
    f_LPF2        = tb.Float32Col(pos=18)  # LPF
    coeff_c       = tb.Float64Col(pos=19, shape=12)  # cleaning coeff
    coeff_blr     = tb.Float64Col(pos=20, shape=12)  # COEFF BLR
    adc_to_pes    = tb.Float32Col(pos=21, shape=12)  # CALIB CONST
    pmt_noise_rms = tb.Float32Col(pos=22, shape=12)  # rms noise


class DECONV_PARAM(tb.IsDescription):
    N_BASELINE            = tb.Int32Col(pos=0)
    THR_TRIGGER           = tb.Int16Col(pos=1)
    ACUM_DISCHARGE_LENGTH = tb.Int16Col(pos=2)


class S12(tb.IsDescription):
    """Store for a S1/S2
    The table maps a S12:
    peak is the index of the S12 dictionary, running over the number of peaks found
    time and energy of the peak.
    """
    event  = tb.  Int32Col(pos=0)
    peak   = tb.  UInt8Col(pos=2) # peak number
    time   = tb.Float32Col(pos=3) # time in ns
    ene    = tb.Float32Col(pos=4) # energy in pes


class S2Si(tb.IsDescription):
    """Store for a S2Si
    The table maps a S2Si
    peak is the same than the S2 peak
    nsipm gives the SiPM number
    only energies are stored (times are defined in S2)
    """
    event = tb.  Int32Col(pos=0)
    peak  = tb.  UInt8Col(pos=2) # peak number
    nsipm = tb.  Int16Col(pos=3) # sipm number
    ene   = tb.Float32Col(pos=5) # energy in pes


class KrTable(tb.IsDescription):
    event = tb.  Int32Col(pos= 0)
    time  = tb.Float64Col(pos= 1)
    peak  = tb. UInt16Col(pos= 2)
    nS2   = tb. UInt16Col(pos= 3)

    S1w   = tb.Float64Col(pos= 4)
    S1h   = tb.Float64Col(pos= 5)
    S1e   = tb.Float64Col(pos= 6)
    S1t   = tb.Float64Col(pos= 7)

    S2w   = tb.Float64Col(pos= 8)
    S2h   = tb.Float64Col(pos= 9)
    S2e   = tb.Float64Col(pos=10)
    S2q   = tb.Float64Col(pos=11)
    S2t   = tb.Float64Col(pos=12)

    Nsipm = tb. UInt16Col(pos=13)
    DT    = tb.Float64Col(pos=14)
    Z     = tb.Float64Col(pos=15)
    X     = tb.Float64Col(pos=16)
    Y     = tb.Float64Col(pos=17)
    R     = tb.Float64Col(pos=18)
    Phi   = tb.Float64Col(pos=19)
    Xrms  = tb.Float64Col(pos=20)
    Yrms  = tb.Float64Col(pos=21)


class XYfactors(tb.IsDescription):
    x            = tb.Float32Col(pos=0)
    y            = tb.Float32Col(pos=1)
    factor       = tb.Float32Col(pos=2)
    uncertainty  = tb.Float32Col(pos=3)
    nevt         = tb. UInt32Col(pos=4)


class Zfactors(tb.IsDescription):
    z            = tb.Float32Col(pos=0)
    factor       = tb.Float32Col(pos=1)
    uncertainty  = tb.Float32Col(pos=2)


class Tfactors(tb.IsDescription):
    t            = tb.Float32Col(pos=0)
    factor       = tb.Float32Col(pos=1)
    uncertainty  = tb.Float32Col(pos=2)


class TrackTable(tb.IsDescription):
    event = tb.  Int32Col(pos= 0)
    time  = tb.Float64Col(pos= 1)

    S1w   = tb.Float64Col(pos= 2)
    S1h   = tb.Float64Col(pos= 3)
    S1e   = tb.Float64Col(pos= 4)
    S1t   = tb.Float64Col(pos= 5)

    npeak = tb. UInt16Col(pos= 6)
    X     = tb.Float64Col(pos= 7)
    Y     = tb.Float64Col(pos= 8)
    Z     = tb.Float64Col(pos= 9)
    R     = tb.Float64Col(pos=10)
    Phi   = tb.Float64Col(pos=11)
    Nsipm = tb. UInt16Col(pos=12)
