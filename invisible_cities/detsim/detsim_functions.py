"""
detsim_functions.py
Defines key functions used in Detsim.
"""

import numpy as np

from .. core.system_of_units_c import units

from .. evm.event_model        import MCHit

from .. reco                   import peak_functions           as pkf

def diffuse_and_smear_hits(mchits, zmin, zmax, diff_transv, diff_long,
                           resolution_FWHM, Qbb):
    """
    Applies diffusion and energy smearing to all MC hits.
    """
    # calculate unscaled variance for energy smearing
    E_evt = sum([hit.E for hit in mchits])
    sigma0 = ((resolution_FWHM/100.) * np.sqrt(Qbb) * np.sqrt(E_evt)) / 2.355
    var0 = sigma0**2

    # calculate drift distance
    zdrift = np.random.uniform(zmin,zmax)

    # apply diffusion and energy smearing
    dmchits = []
    for hit in mchits:

        xh = np.random.normal(hit.X,np.sqrt(zdrift/10.)*diff_transv)
        yh = np.random.normal(hit.Y,np.sqrt(zdrift/10.)*diff_transv)
        zh = np.random.normal(hit.Z+zdrift,np.sqrt(zdrift/10.)*diff_long)
        eh = np.random.normal(hit.E,np.sqrt(var0*hit.E/E_evt))

        dmchits.append(MCHit([xh,yh,zh], hit.T, eh))

    return dmchits,zdrift
