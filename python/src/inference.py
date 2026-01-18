
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm

class Inference:
    def __init__(self, data):
        """
        data: pandas DataFrame or dict of arrays/lists.
        """
        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            self.data = pd.DataFrame(data)

    def run_ttest(self, value_col, group_col, paired=False, equal_var=True):
        """
        Runs T-test (Indep, Welch, or Paired).
        """
        groups = self.data[group_col].unique()
        if len(groups) != 2:
            raise ValueError("T-test requires exactly 2 groups.")

        g1 = self.data[self.data[group_col] == groups[0]][value_col].dropna()
        g2 = self.data[self.data[group_col] == groups[1]][value_col].dropna()

        if paired:
            stat, p = stats.ttest_rel(g1, g2)
            test_name = "Paired T-test"
        else:
            stat, p = stats.ttest_ind(g1, g2, equal_var=equal_var)
            test_name = "Independent T-test" if equal_var else "Welch's T-test"

        return {
            "Test": test_name,
            "Statistic": stat,
            "p-value": p,
            "dof": len(g1) + len(g2) - 2 # Approx
        }

    def run_mann_whitney(self, value_col, group_col):
        """
        Non-parametric independent test.
        """
        groups = self.data[group_col].unique()
        if len(groups) != 2:
            raise ValueError("Mann-Whitney requires 2 groups.")

        g1 = self.data[self.data[group_col] == groups[0]][value_col].dropna()
        g2 = self.data[self.data[group_col] == groups[1]][value_col].dropna()

        stat, p = stats.mannwhitneyu(g1, g2)
        return {"Test": "Mann-Whitney U", "Statistic": stat, "p-value": p}

    def run_lmm(self, formula, group_col):
        """
        Runs Linear Mixed Model.
        formula: string, e.g., 'Response ~ Condition'
        group_col: column name for Random Effects (groups), e.g., 'SubjectID'
        """
        model = mixedlm(formula, self.data, groups=self.data[group_col])
        result = model.fit()
        return result

    def run_anova(self, value_col, group_col):
        """
        One-way ANOVA.
        """
        formula = f'{value_col} ~ C({group_col})'
        model = ols(formula, data=self.data).fit()
        aov_table = sm.stats.anova_lm(model, typ=2)
        return {"Test": "One-way ANOVA", "Table": aov_table}
