import os

from operator    import mul
from operator    import truediv
from collections import namedtuple

import numpy  as np
import pandas as pd

from pytest import fixture
from pytest import mark

from hypothesis              import given
from hypothesis.strategies   import integers
from hypothesis.strategies   import lists
from hypothesis.extra.pandas import columns, data_frames

from .. io.dst_io            import load_dst
from .. io.dst_io            import load_dsts
from .  corrections          import Correction
from .  dst_functions        import load_xy_corrections
from .  dst_functions        import load_lifetime_xy_corrections
from .  dst_functions        import dst_event_id_selection
from .  dst_functions        import load_paolina_summary
from .. core.testing_utils   import assert_dataframes_equal

normalization_data = namedtuple("normalization_data", "node kwargs op")

@fixture(scope  = "session",
         params = [False, True])
def normalization(request):
    if request.param:
        node   = "LifetimeXY_inverse"
        kwargs = {"norm_strategy": "const",
                  "norm_opts"    : {"value": 1}}
        op     = truediv
    else:
        node   = "LifetimeXY"
        kwargs = {}
        op     = mul
    return normalization_data(node, kwargs, op)


def test_load_xy_corrections(corr_toy_data, normalization):
    filename, true_data = corr_toy_data
    x, y, E, U, _ = true_data
    corr          = load_xy_corrections(filename,
                                        node = normalization.node,
                                        **normalization.kwargs)
    assert corr == Correction((x,y), E, U)


@mark.parametrize("scale",
                  (0.5, 1, 2.0))
def test_load_lifetime_xy_corrections(corr_toy_data, normalization, scale):
    filename, true_data = corr_toy_data
    x, y, LT, U, _ = true_data
    corr           = load_lifetime_xy_corrections(filename,
                                                  node  = normalization.node,
                                                  scale = scale,
                                                  **normalization.kwargs)

    LT = normalization.op(LT, scale)
    U  = normalization.op(U , scale)
    for i in np.linspace(0, 2, 5):
        # This should yield exp(i * x/x) = exp(i)
        z_test   = LT.flatten() * i
        x_test   = np.repeat(x, y.size)
        y_test   = np.tile  (y, x.size)
        (f_test,
         u_test) = corr(z_test, x_test, y_test)

        f_true = np.exp(i)
        u_true = z_test * U.flatten()/LT.flatten()**2 * f_test
        assert np.allclose(f_test, f_true)
        assert np.allclose(u_test, u_true)


@given(data_frames(columns=columns(['event'], elements=integers(min_value=-1e5, max_value=1e5))),
       lists(integers(min_value=-1e5, max_value=1e5)))
def test_dst_event_id_selection(dst, events):
    filtered_dst = dst_event_id_selection(dst, events)
    assert set(filtered_dst.event.values) == set(dst.event.values) & set(events)


def test_dst_event_id_selection_2():
    data         = {'event': [1, 1, 3, 6, 7], 'values': [3, 4, 2, 5, 6]}
    filt_data    = {'event': [1, 1, 6], 'values': [3, 4, 5]}

    df_data      = pd.DataFrame(data=data)
    df_filt_data = pd.DataFrame(data=filt_data)
    df_real_filt = dst_event_id_selection(df_data, [1, 2, 6, 10])

    assert np.all(df_real_filt.reset_index(drop=True) == df_filt_data.reset_index(drop=True))


def test_load_paolina_summary_equal_to_old_summary(ICDATADIR):
    old_esmeralda     = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC.h5")
    kdst_esmeralda    = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC_KDST_no_filter.h5")
    summary_df_old    = load_dst            (old_esmeralda, 'PAOLINA', 'Summary')
    summary_df_reader = load_paolina_summary(kdst_esmeralda)
    columns = sorted(summary_df_old.columns)
    assert_dataframes_equal(summary_df_old[columns], summary_df_reader[columns], check_types=False)


def test_load_paolina_summary_loads_multiple_file(ICDATADIR):
    old_esmeralda     = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC.h5")
    kdst_esmeralda    = os.path.join(ICDATADIR, "exact_Kr_tracks_with_MC_KDST_no_filter.h5")
    summary_df_old    = load_dsts           ([old_esmeralda,  old_esmeralda  ], 'PAOLINA', 'Summary')
    summary_df_reader = load_paolina_summary([kdst_esmeralda, kdst_esmeralda])
    columns = sorted(summary_df_old.columns)
    assert_dataframes_equal(summary_df_old[columns], summary_df_reader[columns], check_types=False)
