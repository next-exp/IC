"""
Tables defining the DM
"""
import tables as tb


class RunInfo(tb.IsDescription):
    run_number = tb.Int32Col(shape=(), pos=0)


class EventInfo(tb.IsDescription):
    evt_number = tb.Int32Col(shape=(), pos=0)
    timestamp = tb.UInt64Col(shape=(), pos=1)


class DetectorGeometry(tb.IsDescription):
    """
    Stores geometry information for the detector
    """
    x_det = tb.Float32Col(shape=2, pos=1)  # xmin, xmax
    y_det = tb.Float32Col(shape=2, pos=2)  # ymin, ymax
    z_det = tb.Float32Col(shape=2, pos=3)  # zmin, zmax
    r_det = tb.Float32Col(pos=4)  # radius


class DataSensor(tb.IsDescription):
    """
    Stores metadata information for the SiPMs
    (position, gain, calibration-constant, mask)
    """
    channel = tb.Int32Col(pos=0)  # electronic channel
    position = tb.Float32Col(shape=3, pos=1)
    coeff = tb.Float64Col(pos=2)
    adc_to_pes = tb.Float32Col(pos=3)
    noise_rms = tb.Float32Col(pos=4)


class MCTrack(tb.IsDescription):
    """
    Stores the parameters used by the simulation as metadata
    using Pytables
    """
    event_indx = tb.Int16Col(pos=1)
    mctrk_indx = tb.Int16Col(pos=2)
    particle_name = tb.StringCol(10, pos=3)
    pdg_code = tb.Int16Col(pos=4)
    initial_vertex = tb.Float32Col(shape=3, pos=5)
    final_vertex = tb.Float32Col(shape=3, pos=6)
    momentum = tb.Float32Col(shape=3, pos=7)
    energy = tb.Float32Col(pos=8)
    nof_hits = tb.Int16Col(pos=9)
    hit_indx = tb.Int16Col(pos=10)
    hit_position = tb.Float32Col(shape=3, pos=11)
    hit_time = tb.Float32Col(pos=12)
    hit_energy = tb.Float32Col(pos=13)


class SENSOR_WF(tb.IsDescription):
    """
    Describes a true waveform (zero supressed)
    """
    event = tb.UInt32Col(pos=0)
    ID = tb.UInt32Col(pos=1)
    time_mus = tb.Float32Col(pos=2)
    ene_pes = tb.Float32Col(pos=3)


class FEE(tb.IsDescription):
    """
    Stores the parameters used by the EP simulation as metadata
    """
    OFFSET = tb.Int16Col(pos=1)  # displaces the baseline (e.g, 700)
    CEILING = tb.Int16Col(pos=2)  # adc top count (4096)
    PMT_GAIN = tb.Float32Col(pos=3)  # Gain of PMT (4.5e6)
    FEE_GAIN = tb.Float32Col(pos=4)  # FE gain (250*ohm)
    R1 = tb.Float32Col(pos=5)  # resistor in Ohms (2350*ohm)
    C1 = tb.Float32Col(pos=6)  # Capacitor C1 in nF
    C2 = tb.Float32Col(pos=7)  # Capacitor C2 in nF
    ZIN = tb.Float32Col(pos=8)  # equivalent impedence
    DAQ_GAIN = tb.Float32Col(pos=9)
    NBITS = tb.Float32Col(pos=10)  # number of bits ADC
    LSB = tb.Float32Col(pos=11)  # LSB (adc count)
    NOISE_I = tb.Float32Col(pos=12)  # Noise at the input
    NOISE_DAQ = tb.Float32Col(pos=13)  # Noise at DAQ
    t_sample = tb.Float32Col(pos=14)  # sampling time
    f_sample = tb.Float32Col(pos=15)  # sampling frequency
    f_mc = tb.Float32Col(pos=16)  # sampling frequency in MC (1ns)
    f_LPF1 = tb.Float32Col(pos=17)  # LPF
    f_LPF2 = tb.Float32Col(pos=18)  # LPF
    coeff_c = tb.Float64Col(shape=12, pos=19)  # cleaning coeff
    coeff_blr = tb.Float64Col(shape=12, pos=20)  # COEFF BLR
    adc_to_pes = tb.Float32Col(shape=12, pos=21)  # CALIB CONST
    pmt_noise_rms = tb.Float32Col(shape=12, pos=22)  # rms noise


class DECONV_PARAM(tb.IsDescription):
    N_BASELINE = tb.Int16Col(pos=0)
    THR_TRIGGER = tb.Int16Col(pos=1)
    ACUM_DISCHARGE_LENGTH = tb.Int16Col(pos=2)
    ACUM_TAU = tb.Int16Col(pos=3)
    ACUM_COMPRESS = tb.Float32Col(pos=4)


class PMAP(tb.IsDescription):
    """
    Store for a PMap
    """
    event = tb.Int32Col(pos=0)
    peak = tb.UInt8Col(pos=1)
    signal = tb.StringCol(2, pos=2)
    time = tb.Float32Col(pos=3)
    ToT = tb.UInt16Col(pos=4)
    cathode = tb.Float32Col(pos=5)
    anode = tb.Float32Col(pos=6, shape=(1792,))
