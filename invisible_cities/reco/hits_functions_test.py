import numpy  as np
import pandas as pd

from pytest        import mark
from pytest        import fixture
from numpy.testing import assert_almost_equal

from   .. core.testing_utils   import assert_dataframes_close
from   .. types.ic_types       import NN
from   .  hits_functions       import e_from_q
from   .  hits_functions       import merge_NN_hits
from   .  hits_functions       import threshold_hits
from   .  hits_functions       import sipms_above_threshold
from   .  hits_functions       import cluster_tagger
from hypothesis                import given
from hypothesis.strategies     import lists
from hypothesis.strategies     import floats
from hypothesis.strategies     import integers
from copy                      import deepcopy
from hypothesis                import assume
from hypothesis.strategies     import composite
from hypothesis.extra.pandas   import data_frames, column, range_indexes

event_numbers = integers(0, np.iinfo(np.int32).max)

@composite
def hit(draw, event=None):
    event = draw(event_numbers) if event is None else event
    Q     = draw(floats  (-10, 100)
                 .map(lambda x: NN if x<=0  else x)
                 .filter(lambda x: abs(x)>.1)
                )

    hit = pd.DataFrame(dict( event    = event
                           , time     = 0
                           , npeak    = 0
                           , Xpeak    = draw(floats  (  1,   5))
                           , Ypeak    = draw(floats  (-10,  10))
                           , X        = draw(floats  (  1,   5))
                           , Y        = draw(floats  (-10,  10))
                           , Z        = draw(floats  ( 50, 100))
                           , Q        = Q
                           , E        = draw(floats  ( 50, 100))
                           , Ec       = draw(floats  ( 50, 100))
                           ), index=[0])
    return hit


@composite
def list_of_hits(draw):
    event  = draw(event_numbers)
    hits   = draw(lists(hit(event), min_size=2, max_size=10))
    hits   = pd.concat(hits, ignore_index=True)
    non_nn = hits.Q[hits.Q != NN]
    assume(non_nn.sum() >  0)
    assume(non_nn.size  >= 1)
    return hits


@composite
def thresholds(draw, min_value=1, max_value=1):
    th1 = draw (integers(  10   ,  20))
    th2 = draw (integers(  th1+1,  30))
    return th1, th2


def test_e_from_q_simple():
    e  = 1
    qs = np.linspace(12, 34, 56)
    s  = qs.sum()
    es = e_from_q(qs, e)
    assert_almost_equal(es, qs/s)

def test_e_from_q_uniform():
    qs = np.ones(12)
    e  = 345
    es = e_from_q(qs, e)
    assert_almost_equal(es[0], es)

@given(lists(floats(1, 10), min_size=1, max_size=20))
def test_e_from_q_conserves_energy(qs):
    qs = np.asarray(qs)
    e  = 5678
    es = e_from_q(qs, e)
    assert np.isclose(es.sum(), e)

def test_e_from_q_does_not_crash_with_empty_input():
    empty  = np.array([])
    output = e_from_q(empty, 1234)
    assert_almost_equal(output, empty)

def test_e_from_q_does_not_crash_with_zeros():
    zeros  = np.zeros(12)
    output = e_from_q(zeros, 1234)
    assert_almost_equal(output, zeros)

def test_sipms_above_threshold_simple():
    xys = np.arange(6*2).reshape(6, 2)
    qs  = np.arange(6)
    thr = 1.5 # only the last 4 elements should survive
    e   = qs[2:].sum()
    out = sipms_above_threshold(xys, qs, thr, e)
    for i, item in enumerate(out):
        assert len(item)==4, f"{i}th output failed: got {len(item)}, expected 4"

    assert_almost_equal(out[0], xys[2:, 0])
    assert_almost_equal(out[1], xys[2:, 1])
    assert_almost_equal(out[2],  qs[2:])
    assert_almost_equal(out[3],  qs[2:]) # same energy as q, no change

def test_sipms_above_threshold_thr_too_high_produces_NN():
    xys = np.arange(6*2).reshape(6, 2)
    qs  = np.arange(6)
    thr = qs.max() + 1
    e   = 123
    out = sipms_above_threshold(xys, qs, thr, e)
    for i, item in enumerate(out):
        assert len(item)==1, f"{i}th output failed: got {len(item)}, expected 1"

    assert out[0][0] == NN
    assert out[1][0] == NN
    assert out[2][0] == NN
    assert out[3][0] == e  # conserves energy

@given(list_of_hits())
def test_merge_NN_does_not_modify_input(hits):
    hits_org = deepcopy(hits)
    merge_NN_hits(hits)
    assert_dataframes_close(hits_org, hits)


@given(list_of_hits())
def test_merge_hits_energy_conserved(hits):
    hits_merged = merge_NN_hits(hits)
    assert_almost_equal(hits.E .sum(), hits_merged.E .sum())
    assert_almost_equal(hits.Ec.sum(), hits_merged.Ec.sum())


@given(list_of_hits())
def test_merge_nn_hits_does_not_leave_nn_hits(hits):
    hits_merged = merge_NN_hits(hits)
    assert all(hits_merged.Q != NN)


@given(list_of_hits(), floats())
def test_threshold_hits_does_not_modify_input(hits, th):
    hits_org = deepcopy(hits)
    threshold_hits(hits, th)
    assert_dataframes_close(hits_org, hits)


@given(hits=list_of_hits(), th=floats())
def test_threshold_hits_energy_conserved(hits, th):
    hits_thresh = threshold_hits(hits, th)
    assert_almost_equal(hits.E .sum(), hits_thresh.E .sum())
    assert_almost_equal(hits.Ec.sum(), hits_thresh.Ec.sum())


@given(hits=list_of_hits(), th=floats())
def test_threshold_hits_all_larger_than_th(hits, th):
    hits_thresh = threshold_hits(hits, th)
    non_nn = hits_thresh.loc[hits_thresh.Q != NN]
    assert np.all(non_nn.Q >= th)

# ----- CLUSTER TAGGER TESTS ----- #

gen_cluster_df = data_frames( index=range_indexes(min_size=1, max_size=50),
                              columns=[
                                column('event', dtype=int,   elements=integers(min_value=0, max_value=10)),
                                column('X',     dtype=float, elements=floats(min_value=-500, max_value=500)),
                                column('Y',     dtype=float, elements=floats(min_value=-500, max_value=500)),
                                column('Z',     dtype=float, elements=floats(min_value=0,    max_value=1200)),
                                column('E',     dtype=float, elements=floats(min_value=0.1,  max_value=100)),    
                                ])

@given(df=gen_cluster_df)
def test_dummy(df):
    """
    Hypothesis calls this function multiple times.
    'df' will be a different pandas DataFrame in every call.
    """
    # Just for demonstration purposes, we print the shape of the generated DFs
    print(f"Generated dataframe shape: {df.shape}")
    
    # Check some stuff here
    assert 'X' in df.columns
    assert df['event'].dtype == int
    assert not df.empty

@settings(deadline=None)
@given(df=gen_cluster_df)
def test_cluster_tagger_structure_preservation(df):
    """
    Verifies that cluster_tagger:
        - Returns a DataFrame with the exact same length as the input.
        - Adds exactly one column named 'cluster'.
        - Does not modify any of the original columns (X, Y, Z, E, etc.).
        - Preserves the original Index and order of rows.
        - The 'cluster' column contains valid integers (no NaNs).
    """
    # Shuffle the input DataFrame to ensure cluster_tagger does not rely on any specific order
    df_input = df.sample(frac=1.0).copy()
    df_original = df_input.copy()           # Keep a copy of the original for later comparison

    # Run the cluster tagger
    params = dict(eps=10.0, min_samples=1, scale_xy=1.0, scale_z=1.0)     # Dummy values
    df_result = cluster_tagger(df_input, **params)

    # --- Assertations
    assert len(df_result) == len(df_original), "Output DataFrame has different length than input."
    assert 'cluster' in df_result.columns,     "Output DataFrame does not contain 'cluster' column."
    expected_cols = set(df_original.columns) | {'cluster'}
    assert set(df_result.columns) == expected_cols, "Output DataFrame has unexpected columns."
    pd.testing.assert_frame_equal(  
                                    df_result.drop(columns=['cluster']),
                                    df_original,
                                    check_dtype=True,
                                    obj="Dataframe structure check"
                                 )
    assert pd.api.types.is_integer_dtype(df_result['cluster']), "'cluster' column is not of integer type."
    assert not df_result['cluster'].isna().any(), "'cluster' column contains NaN values."

def test_cluster_tagger_row_alignment():
    """
    Verifies that the calculated cluster label is assigned to the correct 
    spatial hit, even if the input DataFrame is shuffled.
    
    Scenario:
    - Event 0:
        - Cluster A: 2 hits at (0,0,0) and (1,1,0)         -> Should be Cluster 0
        - Cluster B: 2 hits at (100,100,0) and (101,101,0) -> Should be Cluster 1
    - We check that hits near 0 get Label 0 and hits near 100 get Label 1 (NO noise here).
    """
    # Setup data
    data = {
                'event': [0, 0, 0, 0],
                'X':     [0., 1., 100., 101.],
                'Y':     [0., 1., 100., 101.],
                'Z':     [0., 0.,   0.,   0.],
                'E':     [10, 10,  10,   10 ]
    }
    df = pd.DataFrame(data)
    df['expected_label'] = [0, 0, 1, 1]

    # Shuffle the input DataFrame
    df_input = df.sample(frac=1.0).copy()

    # Run the cluster tagger
    params = dict(eps=5.0, min_samples=1, scale_xy=1.0, scale_z=1.0)      # Enough to consider both clusters
    df_result = cluster_tagger(df_input, **params)

    # --- Assertations
    hits_group_0 = df_result[df_result['expected_label'] == 0]
    hits_group_1 = df_result[df_result['expected_label'] == 1]
    assert hits_group_0['cluster'].nunique() == 1, "Hits near (0,0,0) were assigned multiple cluster labels."
    assert hits_group_1['cluster'].nunique() == 1, "Hits near (100,100,0) were assigned multiple cluster labels."
    label_0 = hits_group_0['cluster'].iloc[0]
    label_1 = hits_group_1['cluster'].iloc[0]
    assert label_0 != label_1, "Both clusters were assigned the same label."
    assert label_0 != -1 and label_1 != -1, "One of the clusters was labeled as noise (-1)."
    
def test_cluster_tagger_noise_rejection():
    """
    Verifies that isolated hits (outliers) are correctly identified as noise (-1).
    
    Scenario:
    - 3 points very close together (0,0), (1,0), (0,1). They should form a cluster.
    - 1 point very far away (100, 100). It has 0 neighbors. Should be noise.
    """
    # Setup data
    data = {
                'event': [0, 0, 0, 0],
                'X':     [0., 1., 0., 100.],
                'Y':     [0., 0., 1., 100.],
                'Z':     [0., 0., 0.,   0.],
                'E':     [10, 10, 10,  10 ]
    }
    df = pd.DataFrame(data)

    # Shuffle the input DataFrame
    df_input = df.sample(frac=1.0).copy()

    # Run the cluster tagger
    params = dict(eps=5.0, min_samples=3, scale_xy=1.0, scale_z=1.0)      # Enough to consider one cluster
    df_result = cluster_tagger(df_input, **params)

    # --- Assertations
    cluster_labels = df_result['cluster'].unique()
    assert cluster_labels.size == 2, "Expected exactly 2 unique cluster labels (one cluster + one noise)."
    cluster_hits = df_result[df_result['cluster'] != -1]
    assert cluster_hits.shape[0] == 3, "Expected exactly 3 hits to be clustered together."
    noise_hit = df_result[df_result['cluster'] == -1]
    assert noise_hit.shape[0] == 1, "Expected exactly 1 noise hit."
    assert noise_hit['X'].iloc[0] == 100 and noise_hit['Y'].iloc[0] == 100, "The noise hit identified is NOT the distant one."

def test_cluster_tagger_event_distinction():
    """
    Verifies that hits from different events are not clustered together.
    
    Scenario:
    - Event 0: 2 hits at (0,0,0) and (1,1,0) -> Should be Cluster 0
    - Event 1: 2 hits at (100,100,0) and (101,101,0) and 1 hit at (0.5,0.5,0) -> Should be marked as noise (-1)
    - We check that noise hit from Event 1 get a different cluster label than hits from Event 0, even if they are spatially close.
    """
    # Setup data
    data = {
                'event': [0, 0, 1, 1, 1],
                'X':     [0., 1., 100., 101., 0.5],
                'Y':     [0., 1., 100., 101., 0.5],
                'Z':     [0., 0.,   0.,   0.,  0.],
                'E':     [10, 10,   10,   10,  10]
    }
    df = pd.DataFrame(data)

    # Shuffle the input DataFrame
    df_input = df.sample(frac=1.0).copy()

    # Run the cluster tagger
    params = dict(eps=5.0, min_samples=2, scale_xy=1.0, scale_z=1.0)      # Enough to consider both clusters
    df_result = cluster_tagger(df_input, **params)

    # --- Assertations
    event_0_clusters = df_result[df_result['event'] == 0]['cluster'].unique()
    event_1_clusters = df_result[df_result['event'] == 1]['cluster'].unique()
    assert len(event_0_clusters) == 1, "For event 0: expected exactly 1 unique cluster label (one cluster)."
    assert len(event_1_clusters) == 2, "For event 1: expected exactly 2 unique cluster labels (one cluster + one noise)."
    event_0_hits = df_result[df_result['event'] == 0]
    noise_1_hit  = df_result[(df_result['X'] == 0.5)]
    assert noise_1_hit['cluster'].iloc[0] == -1, "The hit at (0.5,0.5,0) in event 1 should be marked as noise (-1)."
    assert event_0_hits['cluster'].iloc[0] != noise_1_hit['cluster'].iloc[0], "Hits from event 0 and the noise hit from event 1 were assigned the same cluster label."
