# $Id: system_of_units.h,v 1.1 2004/04/16 17:09:03 gomez Exp $
# ----------------------------------------------------------------------
# HEP coherent system of Unitscoulomb
#
# This file has been provided by Geant4 (simulation toolkit for HEP).
#
# The basic units are :
#  		millimeter              (millimeter)
# 		nanosecond              (nanosecond)
# 		Mega electron Volt      (MeV)
# 		positron charge         (eplus)
# 		degree Kelvin           (kelvin)
#              the amount of substance (mole)
#              luminous intensity      (candela)
# 		radian                  (radian)
#              steradian               (steradian)
#
# Below is a non exhaustive list of derived and pratical units
# (i.e. mostly the SI units).
# You can add your own units.
#
# The SI numerical value of the positron charge is defined here,
# as it is needed for conversion factor : positron charge = e_SI (coulomb)
#
# The others physical constants are defined in the header file :
# PhysicalConstants.h
#
# Authors: M.Maire, S.Giani
#
# History:
#
# 06.02.96   Created.
# 28.03.96   Added miscellaneous constants.
# 05.12.97   E.Tcherniaev: Redefined pascal (to avoid warnings on WinNT)
# 20.05.98   names: meter, second, gram, radian, degree
#            (from Brian.Lasiuk@yale.edu (STAR)). Added luminous units.
# 05.08.98   angstrom, picobarn, microsecond, picosecond, petaelectronvolt


#
# Length [L]
#


euro = 1.
millimeter = 1.
millimeter2 = millimeter * millimeter
millimeter3 = millimeter * millimeter2

centimeter = 10. * millimeter
centimeter2 = centimeter * centimeter
centimeter3 = centimeter * centimeter2

decimeter = 100. * millimeter
decimeter2 = decimeter * decimeter
decimeter3 = decimeter * decimeter2
liter = decimeter3
l = liter

meter = 1000. * millimeter
meter2 = meter * meter
meter3 = meter * meter2

kilometer = 1000. * meter
kilometer2 = kilometer * kilometer
kilometer3 = kilometer * kilometer2

micrometer = 1.e-6 * meter
nanometer = 1.e-9 * meter
angstrom = 1.e-10 * meter
fermi = 1.e-15 * meter

nm = nanometer
mum = micrometer

micron = micrometer
barn = 1.e-28 * meter2
millibarn = 1.e-3 * barn
microbarn = 1.e-6 * barn
nanobarn = 1.e-9 * barn
picobarn = 1.e-12 * barn

# symbols
mm = millimeter
mm2 = millimeter2
mm3 = millimeter3

cm = centimeter
cm2 = centimeter2
cm3 = centimeter3

m = meter
m2 = meter2
m3 = meter3

km = kilometer
km2 = kilometer2
km3 = kilometer3

ft = 30.48 * cm

#
# Angle
#
radian = 1.
milliradian = 1.e-3 * radian
degree = (3.14159265358979323846/180.0) * radian

steradian = 1.

# symbols
rad = radian
mrad = milliradian
sr = steradian
deg = degree

#
# Time [T]
#
nanosecond = 1.
second = 1.e+9 * nanosecond
millisecond = 1.e-3 * second
microsecond = 1.e-6 * second
picosecond = 1.e-12 * second
year = 3.1536e+7 * second
day = 864e2 * second
minute = 60 * second
hour = 60 * minute

s = second
ms = millisecond
ps = picosecond
mus = microsecond
ns = nanosecond

# new!
hertz = 1./second
kilohertz = 1.e+3 * hertz
megahertz = 1.e+6 * hertz
gigahertz = 1.e+6 * hertz

MHZ = megahertz
kHZ = kilohertz
kHz = kHZ
GHZ = gigahertz


#
# Electric charge [Q]
#
eplus = 1.  # positron charge
e_SI = 1.60217733e-19  # positron charge in coulomb
coulomb = eplus/e_SI  # coulomb = 6.24150 e+18 * eplus


#
# Energy [E]
#
megaelectronvolt = 1.
electronvolt = 1.e-6 * megaelectronvolt
milielectronvolt = 1.e-3 * electronvolt
kiloelectronvolt = 1.e-3 * megaelectronvolt
gigaelectronvolt = 1.e+3 * megaelectronvolt
teraelectronvolt = 1.e+6 * megaelectronvolt
petaelectronvolt = 1.e+9 * megaelectronvolt

meV = milielectronvolt
eV = electronvolt
keV = kiloelectronvolt
MeV = megaelectronvolt
GeV = gigaelectronvolt
TeV = teraelectronvolt
PeV = petaelectronvolt

eV2 = eV*eV

joule = electronvolt/e_SI  # joule = 6.24150 e+12 * MeV

#
# Mass [E][T^2][L^-2]
#
kilogram = joule * second * second / meter2
gram = 1.e-3 * kilogram
milligram = 1.e-3 * gram
ton = 1.e+3 * kilogram
kiloton = 1.e+3 * ton

# symbols
kg = kilogram
g = gram
mg = milligram

#
# Power [E][T^-1]
#
watt = joule/second  # watt = 6.24150 e+3 * MeV/ns

#
# Force [E][L^-1]
#
newton = joule/meter  # newton = 6.24150 e+9 * MeV/mm

#
# Pressure [E][L^-3]
#


hep_pascal = newton/m2  # pascal = 6.24150 e+3 * MeV/mm3
pascal = hep_pascal
Pa = pascal
kPa = 1000 * Pa
MPa = 1e+6 * Pa
GPa = 1e+9 * Pa
bar = 100000 * pascal  # bar = 6.24150 e+8 * MeV/mm3
milibar = 1e-3 * bar

atmosphere = 101325 * pascal  # atm = 6.32420 e+8 * MeV/mm3

denier = gram / (9000 * meter)
#
# Electric current [Q][T^-1]
ampere = coulomb/second  # ampere = 6.24150 e+9 * eplus/ns
milliampere = 1.e-3 * ampere
microampere = 1.e-6 * ampere
nanoampere = 1.e-9 * ampere
mA = milliampere
muA = microampere
nA = nanoampere

#
# Electric potential [E][Q^-1]
#
megavolt = megaelectronvolt/eplus
kilovolt = 1.e-3 * megavolt
volt = 1.e-6 * megavolt
millivolt = 1.e-3 * volt

V = volt
mV = millivolt
kV = kilovolt
MV = megavolt


#
# Electric resistance [E][T][Q^-2]
#
ohm = volt/ampere  # ohm = 1.60217e-16*(MeV/eplus)/(eplus/ns)

#
# Electric capacitance [Q^2][E^-1]
#
farad = coulomb/volt  # farad = 6.24150e+24 * eplus/Megavolt
millifarad = 1.e-3 * farad
microfarad = 1.e-6 * farad
nanofarad = 1.e-9 * farad
picofarad = 1.e-12 * farad

nF = nanofarad
pF = picofarad


#
# Magnetic Flux [T][E][Q^-1]
#
weber = volt * second  # weber = 1000*megavolt*ns

#
# Magnetic Field [T][E][Q^-1][L^-2]
#
tesla = volt*second/meter2  # tesla = 0.001*megavolt*ns/mm2

gauss = 1.e-4 * tesla
kilogauss = 1.e-1 * tesla

#
# Inductance [T^2][E][Q^-2]
#
henry = weber/ampere  # henry = 1.60217e-7*MeV*(ns/eplus)**2

#
# Temperature
#
kelvin = 1.

#
# Amount of substance
#
mole = 1.
mol = mole

#
# Activity [T^-1]
#


becquerel = 1./second

curie = 3.7e+10 * becquerel

Bq = becquerel
mBq = 1e-3 * becquerel
muBq = 1e-6 * becquerel
kBq =  1e+3 * becquerel
MBq =  1e+6 * becquerel
cks = Bq/keV
U238ppb = Bq / 81.
Th232ppb = Bq / 246.
#
# Absorbed dose [L^2][T^-2]
#
gray = joule/kilogram

#
# Luminous intensity [I]
#
candela = 1.

#
# Luminous flux [I]
#
lumen = candela * steradian

#
# Illuminance [I][L^-2]
#
lux = lumen/meter2

#
# Miscellaneous
#
perCent = 1e-2
perThousand = 1e-3
perMillion = 1e-6

pes = 1.
adc = 1


def celsius(tKelvin):
    return tKelvin - 273.15
