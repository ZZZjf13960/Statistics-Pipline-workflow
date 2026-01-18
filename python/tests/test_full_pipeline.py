
import sys
import os
import numpy as np
import pandas as pd
import warnings

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from eda import EDA_Diagnosis
from method_selector import MethodSelector
from inference import Inference

def test_full_pipeline_ideal():
    """
    Simulates Case Study A: Ideal Scenario
    """
    print("\n=== Testing Case A: Ideal ===")
    np.random.seed(42)
    n = 50
    group_a = np.random.normal(loc=300, scale=30, size=n)
    group_b = np.random.normal(loc=320, scale=30, size=n)

    df = pd.DataFrame({
        'RT': np.concatenate([group_a, group_b]),
        'Group': ['Control']*n + ['Treatment']*n
    })

    # 1. EDA
    eda = EDA_Diagnosis(df['RT'])
    dist_check = eda.check_distribution()
    print(f"Normality: {dist_check['Is_Normal_Assumption']}")

    # 2. Selector
    sel = MethodSelector(df, group_col='Group')
    var_check = sel.check_homogeneity('RT')
    print(f"Homogeneity: {var_check['Homogeneity']}")

    method, advice = sel.recommend_method(dist_check['Is_Normal_Assumption'], var_check['Homogeneity'])
    print(f"Recommended: {method}")

    # Assertions
    if not dist_check['Is_Normal_Assumption']:
        warnings.warn("Case A: Should be normal but failed test (random chance).")
    if method != 'Independent T-test':
         warnings.warn(f"Case A: Expected Independent T-test, got {method}")

    # 3. Inference
    inf = Inference(df)
    res = inf.run_ttest('RT', 'Group', equal_var=True)
    print("Inference Result:", res)
    assert res['p-value'] < 0.05, "Case A: Expected significant difference."

def test_full_pipeline_messy():
    """
    Simulates Case Study B: Messy/Hierarchical
    """
    print("\n=== Testing Case B: Messy ===")
    np.random.seed(99)
    n_subjects = 20
    n_trials = 10

    data_rows = []
    for subj in range(n_subjects):
        subj_intercept = np.random.normal(0, 5)
        for trial in range(n_trials):
            condition = np.random.choice(['A', 'B'])
            cond_effect = 2 if condition == 'B' else 0
            noise = np.random.lognormal(0, 0.5)
            value = 10 + subj_intercept + cond_effect + noise
            data_rows.append({'SubjectID': f'S{subj}', 'Condition': condition, 'Value': value})

    df = pd.DataFrame(data_rows)

    # 1. EDA
    eda = EDA_Diagnosis(df['Value'])
    dist_check = eda.check_distribution()
    print(f"Normality: {dist_check['Is_Normal_Assumption']}")

    # 2. Selector (With ID Col)
    sel = MethodSelector(df, group_col='Condition', id_col='SubjectID')
    # Force fail normality check if it passed by chance (unlikely for lognormal)
    is_normal = dist_check['Is_Normal_Assumption']

    method, advice = sel.recommend_method(is_normal, homogeneity_check=True)
    print(f"Recommended: {method}")

    # Assertions
    if 'LMM' not in method:
         raise AssertionError(f"Case B: Expected LMM recommendation, got {method}")

    # 3. Inference
    inf = Inference(df)
    # Fit LMM
    print("Running LMM...")
    res = inf.run_lmm('Value ~ Condition', 'SubjectID')
    print("LMM Converged:", res.converged)
    print(res.summary())

    assert res.converged, "LMM failed to converge."

if __name__ == "__main__":
    try:
        test_full_pipeline_ideal()
        test_full_pipeline_messy()
        print("\nALL PIPELINE TESTS PASSED.")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
