"""Tables defining the DM."""
import tables as tb


class RunInfo(tb.IsDescription):
    run_number = tb.Int32Col(shape=(), pos=0)


class TriggerType(tb.IsDescription):
    trigger_type = tb.Int32Col(shape=(), pos=0)


class EventInfo(tb.IsDescription):
    evt_number = tb. Int64Col(shape=(), pos=0)
    timestamp  = tb.UInt64Col(shape=(), pos=1)


class SensorTable(tb.IsDescription):
    """
    Stores the Sensors group, mimicking what is saved
    by the decoder.
    """
    channel  = tb.Int32Col(pos=0)
    sensorID = tb.Int32Col(pos=1)


class MCGeneratorInfo(tb.IsDescription):
    """Store MC generator information as metadata using
    Pytables.
    """
    evt_number    = tb. Int64Col(pos=0)
    atomic_number = tb. Int32Col(pos=1)
    mass_number   = tb. Int32Col(pos=2)
    region        = tb.StringCol(20, pos=3)


class MCExtentInfo(tb.IsDescription):
    """Store the last row of each table as metadata using
    Pytables.
    """
    evt_number    = tb. Int64Col(pos=0)
    last_hit      = tb.UInt64Col(pos=1)
    last_particle = tb.UInt64Col(pos=2)


class MCEventMap(tb.IsDescription):
    """Map between event index and original event."""
    evt_number = tb.Int64Col(shape=(), pos=0)
    nexus_evt  = tb.Int64Col(shape=(), pos=1)


class MCHitInfo(tb.IsDescription):
    """Stores the simulated hits as metadata using Pytables.
    """
    hit_position  = tb.Float32Col(    pos=0, shape=3)
    hit_time      = tb.Float64Col(    pos=1)
    hit_energy    = tb.Float32Col(    pos=2)
    label         = tb. StringCol(20, pos=3)
    particle_indx = tb.  Int16Col(    pos=4)
    hit_indx      = tb.  Int16Col(    pos=5)


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


class S12(tb.IsDescription):
    """Store for a S1/S2
    The table maps a S12:
    peak is the index of the S12 dictionary,
    running over the number of peaks found
    time and energy of the peak.
    """
    event  = tb.  Int64Col(pos=0)
    peak   = tb.  UInt8Col(pos=2) # peak number
    time   = tb.Float32Col(pos=3) # time in ns
    bwidth = tb.Float32Col(pos=4) # bin width in ns
    ene    = tb.Float32Col(pos=5) # energy in pes

class S12Pmt(tb.IsDescription):
    """Store for a S1/S2 of the individual pmts
    The table maps a S12Pmt:
    peak is the index of the S12 dictionary,
    running over the number of peaks found
    npmt gives the pmt sensor id
    time and energy of the peak.
    """
    event  = tb.  Int64Col(pos=0)
    peak   = tb.  UInt8Col(pos=2) # peak number
    npmt   = tb.  UInt8Col(pos=3) # pmt number (in order of IC db 26/8/2017: equal to SensorID)
    ene    = tb.Float32Col(pos=5) # energy in pes


class S2Si(tb.IsDescription):
    """Store for a S2Si
    The table maps a S2Si
    peak is the same than the S2 peak
    nsipm gives the SiPM number
    only energies are stored (times are defined in S2)
    """
    event = tb.  Int64Col(pos=0)
    peak  = tb.  UInt8Col(pos=2) # peak number
    nsipm = tb.  Int16Col(pos=3) # sipm number
    ene   = tb.Float32Col(pos=5) # energy in pes


class KrTable(tb.IsDescription):
    event   = tb.  Int64Col(pos= 0)
    time    = tb.Float64Col(pos= 1)
    s1_peak = tb. UInt16Col(pos= 2)
    s2_peak = tb. UInt16Col(pos= 3)
    nS1     = tb. UInt16Col(pos= 4)
    nS2     = tb. UInt16Col(pos= 5)

    S1w     = tb.Float64Col(pos= 6)
    S1h     = tb.Float64Col(pos= 7)
    S1e     = tb.Float64Col(pos= 8)
    S1t     = tb.Float64Col(pos= 9)

    S2w     = tb.Float64Col(pos=10)
    S2h     = tb.Float64Col(pos=11)
    S2e     = tb.Float64Col(pos=12)
    S2q     = tb.Float64Col(pos=13)
    S2t     = tb.Float64Col(pos=14)
    qmax    = tb.Float64Col(pos=15)

    Nsipm   = tb. UInt16Col(pos=16)
    DT      = tb.Float64Col(pos=17)
    Z       = tb.Float64Col(pos=18)
    Zrms    = tb.Float64Col(pos=19)
    X       = tb.Float64Col(pos=20)
    Y       = tb.Float64Col(pos=21)
    R       = tb.Float64Col(pos=22)
    Phi     = tb.Float64Col(pos=23)
    Xrms    = tb.Float64Col(pos=24)
    Yrms    = tb.Float64Col(pos=25)


class HitsTable(tb.IsDescription):
    event    = tb.  Int64Col(pos=0)
    time     = tb.Float64Col(pos=1)
    npeak    = tb. UInt16Col(pos=2)
    Xpeak    = tb.Float64Col(pos=3)
    Ypeak    = tb.Float64Col(pos=4)
    nsipm    = tb. UInt16Col(pos=5)
    X        = tb.Float64Col(pos=6)
    Y        = tb.Float64Col(pos=7)
    Xrms     = tb.Float64Col(pos=8)
    Yrms     = tb.Float64Col(pos=9)
    Z        = tb.Float64Col(pos=10)
    Q        = tb.Float64Col(pos=11)
    E        = tb.Float64Col(pos=12)
    Qc       = tb.Float64Col(pos=13)
    Ec       = tb.Float64Col(pos=14)
    track_id = tb.  Int32Col(pos=15)
    Ep       = tb.Float64Col(pos=16)


class VoxelsTable(tb.IsDescription):
    event    = tb.  Int64Col(pos=0)
    X        = tb.Float64Col(pos=1)
    Y        = tb.Float64Col(pos=2)
    Z        = tb.Float64Col(pos=3)
    E        = tb.Float64Col(pos=4)
    size     = tb.Float64Col(pos=5, shape=3)


class EventPassedFilter(tb.IsDescription):
    event  = tb.Int64Col(pos=0)
    passed = tb. BoolCol(pos=1)
