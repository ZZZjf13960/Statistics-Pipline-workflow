import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors
from scipy.signal import find_peaks
import warnings

class EDA_Diagnosis:
    def __init__(self, data):
        """
        Initialize with a pandas Series or numpy array.
        """
        if isinstance(data, pd.Series):
            self.data = data
            self.name = data.name if data.name else "Data"
        elif isinstance(data, np.ndarray) or isinstance(data, list):
            self.data = pd.Series(data, name="Data")
            self.name = "Data"
        else:
            raise ValueError("Input must be a pandas Series, numpy array, or list.")

        # Remove NaNs for analysis
        self.clean_data = self.data.dropna()

    def descriptive_stats(self):
        """
        Calculates 4 moments and provides text description.
        """
        mean_val = self.clean_data.mean()
        std_val = self.clean_data.std()
        skew_val = self.clean_data.skew()
        kurt_val = self.clean_data.kurtosis() # Fisher kurtosis (normal=0)

        # Interpretation
        skew_desc = "Symmetric"
        if skew_val > 1: skew_desc = "Highly Positively Skewed"
        elif skew_val > 0.5: skew_desc = "Moderately Positively Skewed"
        elif skew_val < -1: skew_desc = "Highly Negatively Skewed"
        elif skew_val < -0.5: skew_desc = "Moderately Negatively Skewed"

        kurt_desc = "Mesokurtic (Normal-like)"
        if kurt_val > 1: kurt_desc = "Leptokurtic (Heavy tails/Peaked)"
        elif kurt_val < -1: kurt_desc = "Platykurtic (Light tails/Flat)"

        return {
            "Mean": mean_val,
            "Std": std_val,
            "Skewness": skew_val,
            "Kurtosis": kurt_val,
            "Skew_Desc": skew_desc,
            "Kurt_Desc": kurt_desc
        }

    def visualize(self):
        """
        Generates Hist+KDE, QQ-Plot, Boxplot.
        Returns the figure object.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Hist + KDE
        sns.histplot(self.clean_data, kde=True, ax=axes[0])
        axes[0].set_title(f'{self.name} Distribution')

        # QQ Plot
        stats.probplot(self.clean_data, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot')

        # Boxplot & Violin
        # Combining them: Violin with Box inside
        sns.violinplot(x=self.clean_data, ax=axes[2], color='lightgray', inner=None)
        sns.boxplot(x=self.clean_data, ax=axes[2], width=0.2, boxprops={'zorder': 2})
        axes[2].set_title('Box & Violin Plot')

        plt.tight_layout()
        return fig

    def check_outliers(self, method='iqr', threshold=1.5):
        """
        Detect outliers using IQR or Z-score.
        """
        if method == 'iqr':
            Q1 = self.clean_data.quantile(0.25)
            Q3 = self.clean_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = self.clean_data[(self.clean_data < lower_bound) | (self.clean_data > upper_bound)]

        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(self.clean_data))
            outliers = self.clean_data[z_scores > 3] # Standard threshold is 3
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")

        return outliers

    def check_multimodality(self):
        """
        Checks for multimodality by estimating KDE and counting peaks.
        Returns number of peaks and a description.
        """
        # Estimate KDE using gaussian_kde
        kde = stats.gaussian_kde(self.clean_data)
        # Create a range of x values to evaluate KDE
        x_range = np.linspace(self.clean_data.min(), self.clean_data.max(), 1000)
        kde_values = kde(x_range)

        # Find peaks
        peaks, _ = find_peaks(kde_values)
        num_peaks = len(peaks)

        modality = "Unimodal"
        if num_peaks == 2:
            modality = "Bimodal"
        elif num_peaks > 2:
            modality = "Multimodal"

        return num_peaks, modality

    def check_distribution(self):
        """
        Normality tests and multimodality check.
        Returns advice on transformation.
        """
        n = len(self.clean_data)

        # Normality Tests
        # Shapiro-Wilk is good for n < 5000
        if n < 5000:
            shapiro_stat, shapiro_p = stats.shapiro(self.clean_data)
        else:
            shapiro_stat, shapiro_p = stats.shapiro(self.clean_data)
            warnings.warn("Sample size > 5000, Shapiro-Wilk might be too sensitive.")

        # Lilliefors Test (Corrected from KS test with estimated params)
        ks_stat, ks_p = lilliefors(self.clean_data, dist='norm')

        # Jarque-Bera
        jb_stat, jb_p = stats.jarque_bera(self.clean_data)

        # Multimodality Check
        num_peaks, modality = self.check_multimodality()

        # Decision Logic (Using Shapiro p-value < 0.05 as standard reject null)
        is_normal = shapiro_p > 0.05

        advice = "Data appears Normal. Proceed with Parametric Tests."
        if not is_normal:
            advice = "Data is NOT Normal. "

            if modality != "Unimodal":
                 advice += f"Data appears {modality} ({num_peaks} peaks). Consider mixture models or splitting data. "

            if self.clean_data.skew() > 1:
                if self.clean_data.min() > 0:
                    advice += "Positive Skew: Consider Log or Box-Cox Transform."
                else:
                    advice += "Positive Skew: Consider Square Root (if >=0) or Non-parametric tests."
            elif self.clean_data.skew() < -1:
                advice += "Negative Skew: Consider Reflect & Log or Non-parametric tests."
            else:
                advice += "Consider Non-parametric tests."

        return {
            "Shapiro_Wilk_p": shapiro_p,
            "Lilliefors_p": ks_p,
            "Jarque_Bera_p": jb_p,
            "Is_Normal_Assumption": is_normal,
            "Modality": modality,
            "Num_Peaks": num_peaks,
            "Advice": advice
        }
