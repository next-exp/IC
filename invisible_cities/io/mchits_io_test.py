import os
import numpy  as np
from . mchits_io import load_mchits

def test_load_mchits(electron_MCRD_file):

    X = [-0.10718990862369537, -0.16415221989154816, -0.18664051592350006, -0.19431403279304504]
    Y = [-0.01200979482382536, -0.07335199415683746, -0.09059777110815048, -0.09717071801424026,]
    Z = [25.12295150756836, 25.140811920166016, 25.11968994140625, 25.115009307861328]
    E = [0.006218845956027508, 0.014433029107749462, 0.010182539001107216, 0.009165585972368717]
    t = [0.0009834024822339416, 0.0018070531077682972, 0.002247565658763051, 0.002446305239573121]

    mchits_dict = load_mchits(electron_MCRD_file)
    hX = [hit.X for hit in mchits_dict[0]]
    hY = [hit.Y for hit in mchits_dict[0]]
    hZ = [hit.Z for hit in mchits_dict[0]]
    hE = [hit.E for hit in mchits_dict[0]]
    ht = [hit.time for hit in mchits_dict[0]]

    assert np.isclose(X,hX).all()
    assert np.isclose(Y,hY).all()
    assert np.isclose(Z,hZ).all()
    assert np.isclose(E,hE).all()
    assert np.isclose(t,ht).all()
