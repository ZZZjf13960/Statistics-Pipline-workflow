
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from eda import EDA_Diagnosis

def test_run():
    # 1. Generate Normal Data
    np.random.seed(42)
    normal_data = np.random.normal(loc=100, scale=15, size=200)

    print("\n\n=== Testing Normal Data ===")
    eda_norm = EDA_Diagnosis(normal_data)

    # Check stats
    stats = eda_norm.descriptive_stats()
    print("Stats:", stats)

    # Check outliers (should be few or none)
    outliers = eda_norm.check_outliers()
    print("Outliers (IQR):", len(outliers))

    # Check Distribution
    dist_check = eda_norm.check_distribution()
    print("Distribution Check:", dist_check)

    if not dist_check['Is_Normal_Assumption']:
        print("WARNING: Normal data failed normality check (possible Type I error or random chance).")

    # 2. Generate Skewed Data (Log-Normal)
    skewed_data = np.random.lognormal(mean=0, sigma=1, size=200)

    print("\n\n=== Testing Skewed Data ===")
    eda_skew = EDA_Diagnosis(skewed_data)

    # Check stats
    stats_skew = eda_skew.descriptive_stats()
    print("Stats:", stats_skew)

    # Check outliers (should be many)
    outliers_skew = eda_skew.check_outliers()
    print("Outliers (IQR):", len(outliers_skew))

    # Check Distribution
    dist_check_skew = eda_skew.check_distribution()
    print("Distribution Check:", dist_check_skew)

    if dist_check_skew['Is_Normal_Assumption']:
        print("WARNING: Skewed data passed normality check (Type II error).")
    else:
        print("SUCCESS: Skewed data correctly identified as non-normal.")
        print("Advice given:", dist_check_skew['Advice'])

    # Test Visuals (Will just create the object, usually we don't show plots in CI)
    # But for this environment, we can let it run.
    print("\nGenerating plots (saving to file for verification)...")
    try:
        fig_norm = eda_norm.visualize()
        fig_norm.savefig('data/normal_test_plot.png')
        fig_skew = eda_skew.visualize()
        fig_skew.savefig('data/skew_test_plot.png')
        print("Plots saved successfully.")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    test_run()
