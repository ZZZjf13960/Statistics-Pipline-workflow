%% Statistical Analysis Pipeline Demo (MATLAB)
% This script demonstrates the "Diagnosis First, Inference Second" workflow.

% Setup path
addpath('../src');

%% Case Study A: The "Ideal" Scenario (Parametric Path)
% Scenario: Comparison of Reaction Times between two groups (Control vs Treatment).
% Expectation: Data is Normal -> Independent T-test.

% 1. Data Generation
rng(42);
n = 50;
group_a = normrnd(300, 30, [n, 1]);
group_b = normrnd(320, 30, [n, 1]);

% Create Table
groups = [repmat({'Control'}, n, 1); repmat({'Treatment'}, n, 1)];
rt = [group_a; group_b];
data_ideal = table(rt, groups, 'VariableNames', {'RT', 'Group'});

% Display head
disp(head(data_ideal));

%% Step 1: EDA & Diagnosis
eda = EDA_Diagnosis(data_ideal.RT, 'Reaction Time');
eda.descriptive_stats();
dist_check = eda.check_distribution();
eda.visualize();

% Interpretation:
% Skewness should be low.
% Normality assumption should hold (p > 0.05).

%% Step 2: Method Selection
selector = MethodSelector(data_ideal, 'Group');
[~, is_homo] = selector.check_homogeneity('RT', 'Group');
fprintf('Homogeneity Check (p > 0.05): %d\n', is_homo);

[method, advice] = selector.recommend_method(dist_check.Is_Normal_Assumption, is_homo);
fprintf('Recommended Method: %s\n', method);
fprintf('Advice: %s\n', advice);

%% Step 3: Inference
inf = Inference(data_ideal);

if strcmp(method, 'Independent T-test')
    inf.run_ttest('RT', 'Group', false, true);
elseif strcmp(method, 'Welch T-test')
    inf.run_ttest('RT', 'Group', false, false);
else
    disp('Running alternative...');
end

%% Case Study B: The "Messy" Scenario (Real-world)
% Scenario: Multi-subject repeated measures (Hierarchical). Data is skewed.
% Expectation: Detect Hierarchical structure -> Recommend LMM.

% 1. Data Generation
rng(99);
n_subjects = 20;
n_trials = 10;
subject_ids = {};
conditions = {};
values = [];

for i = 1:n_subjects
    subj_intercept = normrnd(0, 5);
    for j = 1:n_trials
        if rand < 0.5
            cond = 'A'; eff = 0;
        else
            cond = 'B'; eff = 2;
        end
        % Log-normal noise (Skewed)
        noise = lognrnd(0, 0.5);
        val = 10 + subj_intercept + eff + noise;

        subject_ids{end+1, 1} = sprintf('S%d', i);
        conditions{end+1, 1} = cond;
        values(end+1, 1) = val;
    end
end

data_messy = table(subject_ids, conditions, values, 'VariableNames', {'SubjectID', 'Condition', 'Value'});
data_messy.Condition = string(data_messy.Condition); % Ensure string/categorical

%% Step 1: EDA & Diagnosis
eda_messy = EDA_Diagnosis(data_messy.Value, 'Messy Data');
eda_messy.descriptive_stats();
dist_check_messy = eda_messy.check_distribution();
eda_messy.visualize();

% Interpretation:
% Skewness > 1, Normality check fails.

%% Step 2: Method Selection
% Provide ID column to indicate Hierarchy
selector_messy = MethodSelector(data_messy, 'Condition', 'SubjectID');

[method_m, advice_m] = selector_messy.recommend_method(dist_check_messy.Is_Normal_Assumption, true);
fprintf('Recommended Method: %s\n', method_m);
fprintf('Advice: %s\n', advice_m);

%% Step 3: Inference (LMM)
if contains(method_m, 'LMM')
    disp('Running Linear Mixed Model...');
    inf_messy = Inference(data_messy);
    % Formula: Value ~ Condition + (1|SubjectID)
    inf_messy.run_lmm('Value ~ Condition + (1|SubjectID)');
end
