
import scipy.stats as stats
import pandas as pd
import warnings

class MethodSelector:
    def __init__(self, data, group_col=None, id_col=None, paired=False):
        """
        Initialize the selector.
        data: pandas DataFrame (must be used if group_col/id_col are strings) or Series/Array for simple 1D checks.
        group_col: Column name for grouping variable (for t-test/ANOVA).
        id_col: Column name for subject/entity ID (for LMM).
        paired: Boolean, true if data is paired/repeated measures (without explicit ID col, implies simple paired t-test logic).
        """
        self.data = data
        self.group_col = group_col
        self.id_col = id_col
        self.paired = paired

        # If data is DataFrame and group_col is provided, we can do variance checks
        pass

    def check_homogeneity(self, value_col):
        """
        Performs Levene's test for equality of variances.
        """
        if not isinstance(self.data, pd.DataFrame) or self.group_col is None:
            return {"Levene_p": None, "Homogeneity": None}

        groups = self.data[self.group_col].unique()
        if len(groups) < 2:
            return {"Levene_p": None, "Homogeneity": None}

        group_data = [self.data[self.data[self.group_col] == g][value_col].dropna() for g in groups]

        stat, p = stats.levene(*group_data)
        return {"Levene_p": p, "Homogeneity": p > 0.05}

    def recommend_method(self, normality_check, homogeneity_check):
        """
        Logic for recommending a test.
        normality_check: Boolean (True if Normal).
        homogeneity_check: Boolean (True if Variances are equal).
        """
        advice = "No recommendation."
        method = "Unknown"

        is_normal = normality_check
        is_homogeneous = homogeneity_check if homogeneity_check is not None else True # Assume True if not checked/applicable

        # 1. Check for Hierarchical/Nested Data
        if self.id_col is not None:
            advice = "Data has hierarchical structure (ID column detected)."
            if is_normal:
                 advice += " Recommend Linear Mixed Model (LMM)."
                 method = "LMM"
            else:
                 advice += " Data is non-normal. Recommend Generalized Linear Mixed Model (GLMM) or Transformation + LMM."
                 method = "GLMM_or_LMM"
            return method, advice

        # 2. Check for Paired Data (Simple)
        if self.paired:
            if is_normal:
                method = "Paired T-test"
                advice = "Data is Normal and Paired."
            else:
                method = "Wilcoxon Signed-Rank"
                advice = "Data is Non-Normal and Paired."
            return method, advice

        # 3. Independent Groups
        # Count groups
        num_groups = 1
        if isinstance(self.data, pd.DataFrame) and self.group_col:
            num_groups = self.data[self.group_col].nunique()

        if num_groups == 2:
            if is_normal and is_homogeneous:
                method = "Independent T-test"
                advice = "Normality and Homogeneity assumptions met."
            elif is_normal and not is_homogeneous:
                method = "Welch's T-test"
                advice = "Normality met, but Variance is unequal."
            else:
                method = "Mann-Whitney U"
                advice = "Normality assumption violated."

        elif num_groups > 2:
            if is_normal and is_homogeneous:
                method = "One-way ANOVA"
                advice = "Normality and Homogeneity met."
            elif is_normal and not is_homogeneous:
                method = "Welch ANOVA" # Or Kruskal
                advice = "Normality met, Variance unequal."
            else:
                method = "Kruskal-Wallis"
                advice = "Normality violated."
        else:
            # Single group (One-sample)
            if is_normal:
                method = "One-sample T-test"
            else:
                method = "One-sample Wilcoxon"

        return method, advice
