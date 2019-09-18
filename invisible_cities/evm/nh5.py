"""Tables defining the DM."""
import tables as tb


class RunInfo(tb.IsDescription):
    run_number = tb.Int32Col(shape=(), pos=0)


class TriggerType(tb.IsDescription):
    trigger_type = tb.Int32Col(shape=(), pos=0)


class EventInfo(tb.IsDescription):
    evt_number = tb.Int32Col(shape=(), pos=0)
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
    evt_number    = tb.Int32Col(pos=0)
    atomic_number = tb.Int32Col(pos=1)
    mass_number   = tb.Int32Col(pos=2)
    region        = tb.StringCol(20, pos=3)


class MCExtentInfo(tb.IsDescription):
    """Store the last row of each table as metadata using
    Pytables.
    """
    evt_number    = tb.Int32Col(pos=0)
    last_hit      = tb.UInt64Col(pos=1)
    last_particle = tb.UInt64Col(pos=2)


class MCHitInfo(tb.IsDescription):
    """Stores the simulated hits as metadata using Pytables.
    """
    hit_position  = tb.Float32Col(    pos=0, shape=3)
    hit_time      = tb.Float64Col(    pos=1)
    hit_energy    = tb.Float32Col(    pos=2)
    label         = tb. StringCol(20, pos=3)
    particle_indx = tb.  Int16Col(    pos=4)
    hit_indx      = tb.  Int16Col(    pos=5)


class MCParticleInfo(tb.IsDescription):
    """Stores the simulated particles as metadata using Pytables.
    """
    particle_indx  = tb.  Int16Col(     pos= 0)
    particle_name  = tb. StringCol( 20, pos= 1)
    primary        = tb.  Int16Col(     pos= 2)
    mother_indx    = tb.  Int16Col(     pos= 3)
    initial_vertex = tb.Float32Col(     pos= 4, shape=4)
    final_vertex   = tb.Float32Col(     pos= 5, shape=4)
    initial_volume = tb. StringCol( 20, pos= 6)
    final_volume   = tb. StringCol( 20, pos= 7)
    momentum       = tb.Float32Col(     pos= 8, shape=3)
    kin_energy     = tb.Float32Col(     pos= 9)
    creator_proc   = tb. StringCol(100, pos=10)


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


class DECONV_PARAM(tb.IsDescription):
    N_BASELINE             = tb.Int32Col(pos=0)
    THR_TRIGGER            = tb.Int16Col(pos=1)
    ACCUM_DISCHARGE_LENGTH = tb.Int16Col(pos=2)


class S12(tb.IsDescription):
    """Store for a S1/S2
    The table maps a S12:
    peak is the index of the S12 dictionary,
    running over the number of peaks found
    time and energy of the peak.
    """
    event  = tb.  Int32Col(pos=0)
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
    event  = tb.  Int32Col(pos=0)
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
    event = tb.  Int32Col(pos=0)
    peak  = tb.  UInt8Col(pos=2) # peak number
    nsipm = tb.  Int16Col(pos=3) # sipm number
    ene   = tb.Float32Col(pos=5) # energy in pes


class KrTable(tb.IsDescription):
    event   = tb.  Int32Col(pos= 0)
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

    Nsipm   = tb. UInt16Col(pos=15)
    DT      = tb.Float64Col(pos=16)
    Z       = tb.Float64Col(pos=17)
    Zrms    = tb.Float64Col(pos=18)
    X       = tb.Float64Col(pos=19)
    Y       = tb.Float64Col(pos=20)
    R       = tb.Float64Col(pos=21)
    Phi     = tb.Float64Col(pos=22)
    Xrms    = tb.Float64Col(pos=23)
    Yrms    = tb.Float64Col(pos=24)


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


class HitsTable(tb.IsDescription):
    event    = tb.  Int32Col(pos=0)
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


class VoxelsTable(tb.IsDescription):
    event    = tb.  Int32Col(pos=0)
    X        = tb.Float64Col(pos=1)
    Y        = tb.Float64Col(pos=2)
    Z        = tb.Float64Col(pos=3)
    E        = tb.Float64Col(pos=4)
    size     = tb.Float64Col(pos=5, shape=3)


class EventPassedFilter(tb.IsDescription):
    event  = tb.Int32Col(pos=0)
    passed = tb. BoolCol(pos=1)


class PSFfactors(tb.IsDescription):
    xr     = tb.Float32Col(pos=0)
    yr     = tb.Float32Col(pos=1)
    zr     = tb.Float32Col(pos=2)
    x      = tb.Float32Col(pos=3)
    y      = tb.Float32Col(pos=4)
    z      = tb.Float32Col(pos=5)
    factor = tb.Float32Col(pos=6)
    nevt   = tb. UInt32Col(pos=7)
