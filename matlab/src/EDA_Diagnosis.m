classdef EDA_Diagnosis
    % EDA_DIAGNOSIS Module 1: Exploratory Data Analysis & Diagnosis
    % This class implements descriptive statistics, visualization,
    % outlier detection, and distribution checks.

    properties
        Data
        CleanData
        Name
    end

    methods
        function obj = EDA_Diagnosis(data, name)
            % Constructor
            if nargin < 2
                name = 'Data';
            end

            if isvector(data)
                obj.Data = data;
            else
                error('Input must be a vector.');
            end

            obj.Name = name;
            % Remove NaNs
            obj.CleanData = data(~isnan(data));
        end

        function stats_struct = descriptive_stats(obj)
            % Calculates 4 moments and provides text description.
            mean_val = mean(obj.CleanData);
            std_val = std(obj.CleanData);
            skew_val = skewness(obj.CleanData);
            kurt_val = kurtosis(obj.CleanData); % MATLAB returns Pearson kurtosis (normal=3 usually)
            % Adjusting to Excess Kurtosis (Fisher) for consistency with Python/standard reporting?
            % MATLAB 'kurtosis' returns 3 for normal.
            % 'skewness' returns 0 for normal.

            excess_kurt = kurt_val - 3;

            % Interpretation
            skew_desc = 'Symmetric';
            if skew_val > 1
                skew_desc = 'Highly Positively Skewed';
            elseif skew_val > 0.5
                skew_desc = 'Moderately Positively Skewed';
            elseif skew_val < -1
                skew_desc = 'Highly Negatively Skewed';
            elseif skew_val < -0.5
                skew_desc = 'Moderately Negatively Skewed';
            end

            kurt_desc = 'Mesokurtic (Normal-like)';
            if excess_kurt > 1
                kurt_desc = 'Leptokurtic (Heavy tails/Peaked)';
            elseif excess_kurt < -1
                kurt_desc = 'Platykurtic (Light tails/Flat)';
            end

            stats_struct = struct(...
                'Mean', mean_val, ...
                'Std', std_val, ...
                'Skewness', skew_val, ...
                'Kurtosis', kurt_val, ...
                'ExcessKurtosis', excess_kurt, ...
                'Skew_Desc', skew_desc, ...
                'Kurt_Desc', kurt_desc);

            disp('--- Descriptive Statistics ---');
            disp(stats_struct);
        end

        function visualize(obj)
            % Generates Hist+KDE, QQ-Plot, Boxplot.
            figure('Name', [obj.Name ' Diagnosis'], 'Color', 'w');

            % 1. Histogram + KDE
            subplot(1, 3, 1);
            h = histogram(obj.CleanData, 'Normalization', 'pdf');
            hold on;
            [f, xi] = ksdensity(obj.CleanData);
            plot(xi, f, 'LineWidth', 2);
            title([obj.Name ' Distribution']);
            hold off;

            % 2. QQ Plot
            subplot(1, 3, 2);
            qqplot(obj.CleanData);
            title('Q-Q Plot');

            % 3. Boxplot
            subplot(1, 3, 3);
            % Note: Standard MATLAB < R2024a does not have violinplot built-in.
            % We stick to boxplot here. Users with R2024a+ can use violinplot(obj.CleanData).
            boxplot(obj.CleanData);
            title('Boxplot (Violin Plot requires R2024a+)');
        end

        function outliers = check_outliers(obj, method, threshold)
            % Detect outliers using IQR or Z-score.
            if nargin < 2
                method = 'iqr';
            end
            if nargin < 3
                if strcmp(method, 'iqr')
                    threshold = 1.5;
                elseif strcmp(method, 'zscore')
                    threshold = 3;
                end
            end

            outliers = [];
            if strcmp(method, 'iqr')
                Q1 = quantile(obj.CleanData, 0.25);
                Q3 = quantile(obj.CleanData, 0.75);
                IQR_val = Q3 - Q1;
                lower_bound = Q1 - threshold * IQR_val;
                upper_bound = Q3 + threshold * IQR_val;
                outliers = obj.CleanData(obj.CleanData < lower_bound | obj.CleanData > upper_bound);

            elseif strcmp(method, 'zscore')
                z_scores = abs(zscore(obj.CleanData));
                outliers = obj.CleanData(z_scores > threshold);
            end

            if ~isempty(outliers)
                fprintf('Found %d outliers using %s method.\n', length(outliers), method);
            else
                fprintf('No outliers found using %s method.\n', method);
            end
        end

        function [num_peaks, modality] = check_multimodality(obj)
            % Check multimodality using Kernel Density Estimation (KDE)
            [f, xi] = ksdensity(obj.CleanData);
            % Find peaks in the density estimate
            [pks, locs] = findpeaks(f);
            num_peaks = length(pks);

            modality = 'Unimodal';
            if num_peaks == 2
                modality = 'Bimodal';
            elseif num_peaks > 2
                modality = 'Multimodal';
            end
        end

        function check_dist = check_distribution(obj)
            % Normality tests and Decision Advice

            % Shapiro-Wilk (Note: MATLAB standard toolbox doesn't have swtest builtin usually,
            % but 'adtest', 'jbtest', 'lillietest' (KS) are common.
            % We will use Jarque-Bera and Lilliefors (KS approximation)

            [h_jb, p_jb] = jbtest(obj.CleanData); % Jarque-Bera
            [h_lillie, p_lillie] = lillietest(obj.CleanData); % Lilliefors test for normality

            is_normal = (p_lillie > 0.05);

            [num_peaks, modality] = obj.check_multimodality();

            advice = 'Data appears Normal. Proceed with Parametric Tests.';
            if ~is_normal
                advice = 'Data is NOT Normal. ';

                if ~strcmp(modality, 'Unimodal')
                     advice = [advice sprintf('Data appears %s (%d peaks). Consider mixture models or splitting data. ', modality, num_peaks)];
                end

                s = skewness(obj.CleanData);
                if s > 1
                    if min(obj.CleanData) > 0
                        advice = [advice 'Positive Skew: Consider Log or Box-Cox Transform.'];
                    else
                        advice = [advice 'Positive Skew: Consider Non-parametric tests.'];
                    end
                elseif s < -1
                    advice = [advice 'Negative Skew: Consider Reflect & Log or Non-parametric tests.'];
                else
                    advice = [advice 'Consider Non-parametric tests.'];
                end
            end

            check_dist = struct(...
                'Lilliefors_p', p_lillie, ...
                'JarqueBera_p', p_jb, ...
                'Is_Normal_Assumption', is_normal, ...
                'Modality', modality, ...
                'Num_Peaks', num_peaks, ...
                'Advice', advice);

            disp('--- Distribution Check ---');
            disp(check_dist);
        end
    end
end
