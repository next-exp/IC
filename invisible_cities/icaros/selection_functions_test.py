import numpy             as np
import pandas            as pd

from pytest import fixture
from invisible_cities.core.core_functions import in_range

from invisible_cities.icaros.selection_functions import eff_of_selection, select_var_inrange, select_1S1_1S2, apply_selections



dtrms2_low = lambda dt: -0.7 + 0.030 * (dt-20) # Gonzalo's
dtrms2_upp = lambda dt: 2.6 + 0.036 * (dt-20) # Gonzalo's2

@fixture
def dummy_select_var():
    return pd.DataFrame({'event': np.linspace(0, 10, 1001),
                         'Nsipm': np.linspace(0, 30, 1001, dtype = int),
                         'DT': np.linspace(0, 1000, 1001),
                         'R': np.linspace(0, 300, 1001),
                         'S2t': np.linspace(1.3e6, 1.5e6, 1001),
                         'time': np.linspace(1e5, 1e6, 1001),
                         'nS1': np.random.randint(0, 5, 1001),
                         'nS2': np.full(1001, 1, dtype = int),
                         'Ec': np.linspace(0, 110, 1001),
                         'Zrms': np.linspace(0, 10, 1001)
                             })



def test_eff_of_selection():
    df_before = pd.DataFrame(data = {'event':[1,2,3,4,5]})
    events_before = df_before.event.nunique()

    df_after = pd.DataFrame(data = {'event':[1,3,4]})
    events_after = df_after.event.nunique()

    assert eff_of_selection(df_before, df_after) == 0.6
    assert len(df_before) > len(df_after)



def test_select_var_inrange_Zrms(dummy_select_var):

    dummy_select_var['Zrms2'] = dummy_select_var.Zrms**2

    kdst_selected, eff = select_var_inrange(dummy_select_var,'Zrms2', dtrms2_low(dummy_select_var.DT), dtrms2_upp(dummy_select_var.DT), 'band selection')

    DT = kdst_selected.DT.values
    Zrms = kdst_selected.Zrms.values

    expected_eff = len(kdst_selected)/len(dummy_select_var)

    assert np.all(Zrms**2 >= dtrms2_low(DT))
    assert np.all(Zrms**2  <= dtrms2_upp(DT))
    assert eff == expected_eff


def test_select_var_inrange_Xrays(dummy_select_var):

    kdst_selected, eff = select_var_inrange(dummy_select_var, 'Ec', 35, 45, 'remove xrays')

    assert np.all(kdst_selected.Ec <= 45)
    assert np.all(kdst_selected.Ec >= 35)



def test_select_1S1_1S2():

    time = np.linspace(1e5, 1e6, 11)
    kdst_test = pd.DataFrame(data =
                             {'event': np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 5]),
                              'time': time,
                              'nS1': np.array([1, 2, 2, 2, 3, 1, 1, 1, 4, 1, 1]),
                              'nS2': np.full_like(time, 1, dtype = int)})

    kdst_selected, eff = select_1S1_1S2(kdst_test)

    kdst_sel = kdst_selected.groupby('event').count()

    assert np.all(kdst_sel.nS1.values == 1)
    assert np.all(kdst_sel.nS2.values == 1)
    assert np.all((kdst_sel.nS1.values) & (kdst_sel.nS2.values)) == 1



def test_select_var_inrange_S2t(dummy_select_var):

    low_S2t = 1.38e6
    high_S2t = 1.44e6

    kdst_sel, eff = select_var_inrange(dummy_select_var,'S2t', low_S2t, high_S2t, 'S2t in trigger time')

    assert np.all(kdst_sel.S2t.values >= low_S2t)
    assert np.all(kdst_sel.S2t.values <= high_S2t)



def test_select_var_inrange_Rmax(dummy_select_var):
    Rmax = 150

    kdst_sel, eff = select_var_inrange(dummy_select_var, 'R', 0, Rmax, f'events with R less than {Rmax}')

    assert np.all(kdst_sel.R.values <= Rmax)


def test_select_var_inrange_DTrange(dummy_select_var):
    low_DT = 20
    high_DT = 800


    kdst_sel, eff = select_var_inrange(dummy_select_var,'DT', low_DT, high_DT, f'events in DT range [{low_DT}, {high_DT}]')

    assert np.all(kdst_sel.DT.values <= high_DT)
    assert np.all(kdst_sel.DT.values >= low_DT)



def test_select_var_inrange_nsipm(dummy_select_var):
    low_nsipm = 0
    high_nsipm = 30

    kdst_sel, eff = select_var_inrange(dummy_select_var,'Nsipm', low_nsipm, high_nsipm, f'events in NSipm range [{low_nsipm},,{high_nsipm}]')

    assert np.all(kdst_sel.Nsipm.values <= high_nsipm)
    assert np.all(kdst_sel.Nsipm.values >= low_nsipm)



def test_apply_selections(dummy_select_var):

    kdst_selected, efficiencies = apply_selections(dummy_select_var, dtrms2_low, dtrms2_upp, 36, 47, 1.38e6, 1.44e6, 450, 20, 1350, 0, 30)

    total_efficiency = np.prod(efficiencies.values[0, :-1])

    assert np.allclose(total_efficiency, efficiencies.total_efficiency)
