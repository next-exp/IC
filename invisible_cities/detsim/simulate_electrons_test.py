import numpy  as np

from collections import namedtuple
from operator    import itemgetter

from pytest import fixture

from .. core         import system_of_units as units
from .. io.mcinfo_io import load_mchits_df

from . simulate_electrons import generate_ionization_electrons
from . simulate_electrons import drift_electrons
from . simulate_electrons import diffuse_electrons

@fixture(scope="session")
def MChits_and_detsim_params(krypton_MCRD_file):

    #hits
    hits_df    = load_mchits_df(krypton_MCRD_file)
    #choose event
    evt = 0
    hits = hits_df.loc[evt]

    xs = hits["x"].values
    ys = hits["y"].values
    zs = hits["z"].values
    energies = hits["energy"].values

    # electron simulation parameters
    wi = 22.4 * units.eV
    fano_factor = 0.15
    drift_velocity = 1.00 * units.mm / units.mus
    lifetime       = 7.00 * units.ms

    transverse_diffusion   = 1.00 * units.mm / units.cm**0.5
    longitudinal_diffusion = 0.30 * units.mm / units.cm**0.5

    nt = namedtuple(typename    = "MChits_and_detsim_params",
                    field_names = """xs ys zs energies
                                     wi fano_factor
                                     lifetime drift_velocity
                                     transverse_diffusion longitudinal_diffusion""")

    return nt(*itemgetter(*nt._fields)(locals()))


def test_generate_ionization_electrons(MChits_and_detsim_params):

    f = MChits_and_detsim_params

    n_electrons = generate_ionization_electrons(f.energies, f.wi, f.fano_factor)

    # test same lengths and all positive
    assert len(n_electrons) == len(f.energies)
    assert np.all(n_electrons >= 0)

    # test is integer
    assert issubclass(n_electrons.dtype.type, (np.integer, int))


def test_generate_ionization_electrons_null_energies(MChits_and_detsim_params):

    f = MChits_and_detsim_params

    # test for null energies
    n_electrons = generate_ionization_electrons(np.zeros(10), f.wi, f.fano_factor)
    assert np.all(n_electrons == np.zeros(10))


def test_drift_electrons(MChits_and_detsim_params):

    f = MChits_and_detsim_params

    n_ie = np.mean(f.energies / f.wi)
    n_electrons = np.clip(np.random.normal(n_ie, n_ie**0.5, size=len(f.zs)), 0, None).astype(int)
    drifted_electrons = drift_electrons(f.zs, n_electrons, f.lifetime, f.drift_velocity)

    # test all >= 0 and <= n_electrons
    assert np.all(drifted_electrons >= 0)
    assert np.all(drifted_electrons <= n_electrons)

    # test is integer
    assert issubclass(drifted_electrons.dtype.type, (np.integer, int))


def test_drift_electrons_extreme_lifetimes(MChits_and_detsim_params):

    f = MChits_and_detsim_params

    n_ie = np.mean(f.energies / f.wi)
    n_electrons = np.clip(np.random.normal(n_ie, n_ie**0.5, size=len(f.zs)), 0, None).astype(int)

    # test lifetime = 0
    drifted_electrons = drift_electrons(f.zs, n_electrons, 0, f.drift_velocity)
    assert np.all(drifted_electrons == 0)

    # test lifetime = inf
    drifted_electrons = drift_electrons(f.zs, n_electrons, np.inf, f.drift_velocity)
    assert np.all(drifted_electrons == n_electrons)


def test_diffuse_electrons(MChits_and_detsim_params):

    f = MChits_and_detsim_params

    n_ie = np.mean(f.energies / f.wi)
    n_electrons = np.clip(np.random.normal(n_ie, n_ie**0.5, size=len(f.zs)), 0, None).astype(int)

    (dxs, dys, dzs) = diffuse_electrons(f.xs, f.ys, f.zs, n_electrons, f.transverse_diffusion, f.longitudinal_diffusion)

    # test same lengths
    assert len(dxs) == len(dys) == len(dzs) == np.sum(n_electrons)


def test_diffuse_electrons_negative_z(MChits_and_detsim_params):

    f = MChits_and_detsim_params

    n_ie = np.mean(f.energies / f.wi)
    n_electrons = np.clip(np.random.normal(n_ie, n_ie**0.5, size=len(f.zs)), 0, None).astype(int)

    # test z<0 clip
    zs = np.copy(f.zs)
    sel = [0, 2, 5]
    zs[sel] = -1
    (_, _, dzs) = diffuse_electrons(f.xs, f.ys, zs, n_electrons, f.transverse_diffusion, f.longitudinal_diffusion)
    assert np.all(dzs >= 0)


def test_diffuse_electrons_null_diffusions(MChits_and_detsim_params):

    f = MChits_and_detsim_params

    n_ie = np.mean(f.energies / f.wi)
    n_electrons = np.clip(np.random.normal(n_ie, n_ie**0.5, size=len(f.zs)), 0, None).astype(int)

    # test null diffusions
    (dxs, dys, _) = diffuse_electrons(f.xs, f.ys, f.zs, n_electrons, 0, f.longitudinal_diffusion)
    assert np.all(dxs == np.repeat(f.xs, n_electrons))
    assert np.all(dys == np.repeat(f.ys, n_electrons))

    (_, _, dzs) = diffuse_electrons(f.xs, f.ys, f.zs, n_electrons, f.transverse_diffusion, 0)
    assert np.all(dzs == np.repeat(f.zs, n_electrons))
