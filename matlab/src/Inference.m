classdef Inference
    % INFERENCE Module 3: Statistical Inference execution.

    properties
        Data
    end

    methods
        function obj = Inference(data)
            obj.Data = data;
        end

        function result = run_ttest(obj, value_col, group_col, paired, equal_var)
            if nargin < 5, equal_var = true; end
            if nargin < 4, paired = false; end

            y = obj.Data.(value_col);
            groups = unique(obj.Data.(group_col));

            if length(groups) ~= 2
                error('T-test requires exactly 2 groups.');
            end

            g1 = y(obj.Data.(group_col) == groups(1));
            g2 = y(obj.Data.(group_col) == groups(2));

            if paired
                [h, p, ci, stats] = ttest(g1, g2);
                name = 'Paired T-test';
            else
                vartype = 'equal';
                if ~equal_var, vartype = 'unequal'; end
                [h, p, ci, stats] = ttest2(g1, g2, 'Vartype', vartype);
                name = 'Independent T-test';
            end

            result = struct('Test', name, 'h', h, 'p', p, 'tstat', stats.tstat, 'df', stats.df);
            disp('--- Inference Result ---');
            disp(result);
        end

        function result = run_mann_whitney(obj, value_col, group_col)
            y = obj.Data.(value_col);
            groups = unique(obj.Data.(group_col));

             if length(groups) ~= 2
                error('Mann-Whitney requires 2 groups.');
            end

            g1 = y(obj.Data.(group_col) == groups(1));
            g2 = y(obj.Data.(group_col) == groups(2));

            [p, h, stats] = ranksum(g1, g2);
            result = struct('Test', 'Mann-Whitney U', 'h', h, 'p', p, 'zval', stats.zval);
            disp('--- Inference Result ---');
            disp(result);
        end

        function result = run_lmm(obj, formula)
            % Runs Linear Mixed Model using fitlme.
            % Formula example: 'Response ~ Condition + (1|SubjectID)'
            result = fitlme(obj.Data, formula);
            disp('--- Inference Result (LMM) ---');
            disp(result);
        end
    end
end
