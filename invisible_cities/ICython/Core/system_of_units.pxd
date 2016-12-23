# $Id: system_of_units.h,v 1.1 2004/04/16 17:09:03 gomez Exp $
# ----------------------------------------------------------------------
# HEP coherent system of Unitscoulomb
#

cdef class SystemOfUnits:
    cdef readonly double euro, millimeter, millimeter2, millimeter3, centimeter, centimeter2, cm, cm2, cm3, centimeter3, decimeter, decimeter2, decimeter3, liter, l, meter, meter2, meter3, kilometer,  kilometer2,  kilometer3,  micrometer,  nanometer,  angstrom, fermi,  nm,  mum,  micron,  barn,  millibarn,  microbarn,  nanobarn,  picobarn,  mm,  mm2,  mm3, m,  m2,  m3,  km, km2, km3, ft,  radian,  milliradian, mrad, degree,  steradian,  rad, sr,  deg,  nanosecond,  millisecond,  second, year,  day,  minute,  hour,  s,  ms,  ps, mus, ns, picosecond, microsecond, hertz, kilohertz, megahertz,  gigahertz,  MHZ,  kHz, kHZ, GHZ,  eplus,  e_SI,  coulomb,  electronvolt, megaelectronvolt,  milielectronvolt,  kiloelectronvolt, gigaelectronvolt,  teraelectronvolt, petaelectronvolt, meV,  eV,  keV,  MeV,  GeV,  TeV,  PeV, eV2,  joule,  kilogram,  gram,  milligram, ton,  kiloton,  kg,  g,  mg,  watt, newton, hep_pascal,  pascal,  Pa,  kPa,  MPa,  GPa,  bar,  milibar,  atmosphere,  denier,  ampere,  milliampere, microampere, nanoampere,  mA,  muA,  nA,  megavolt,  kilovolt,  volt,  millivolt,  V,  mV,  kV, MV,  ohm, farad,  millifarad,  microfarad,  nanofarad,  picofarad, nF,  pF, weber,  tesla,  gauss,  kilogauss,  henry,  kelvin,  mole,  mol,  becquerel,  curie,  Bq,  mBq,  muBq, cks,  U238ppb, Th232ppb,  gray,  candela,  lumen,  lux, perCent, perThousand,  perMillion,  pes, adc


cpdef double celsius(double tKelvin)
