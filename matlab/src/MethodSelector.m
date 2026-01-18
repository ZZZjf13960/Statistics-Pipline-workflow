classdef MethodSelector
    % METHODSELECTOR Module 2: Logic Bridge
    % Recommends statistical tests based on data properties.

    properties
        Data
        GroupCol
        IDCol
        Paired
    end

    methods
        function obj = MethodSelector(data, group_col, id_col, paired)
            if nargin < 4, paired = false; end
            if nargin < 3, id_col = []; end
            if nargin < 2, group_col = []; end

            obj.Data = data;
            obj.GroupCol = group_col;
            obj.IDCol = id_col;
            obj.Paired = paired;
        end

        function [p_levene, is_homo] = check_homogeneity(obj, value_col_name, group_col_name)
            % Performs Levene's test (or similar variance test).
            % MATLAB's 'vartestn' can do Levene, Bartlett, etc.
            if isempty(obj.Data) || isempty(group_col_name)
                 p_levene = NaN; is_homo = false; return;
            end

            y = obj.Data.(value_col_name);
            g = obj.Data.(group_col_name);

            [p_levene, ~] = vartestn(y, g, 'TestType', 'LeveneQuadratic', 'Display', 'off');
            is_homo = p_levene > 0.05;
        end

        function [method, advice] = recommend_method(obj, is_normal, is_homo)
            % Recommend test based on flags.
            method = 'Unknown';
            advice = 'No recommendation.';

            if isempty(is_homo), is_homo = true; end % Default if skipped

            % 1. Hierarchical
            if ~isempty(obj.IDCol)
                advice = 'Data has hierarchical structure (ID column detected).';
                if is_normal
                    advice = [advice ' Recommend Linear Mixed Model (LMM).'];
                    method = 'LMM';
                else
                    advice = [advice ' Data is non-normal. Recommend GLMM or Transform + LMM.'];
                    method = 'GLMM_or_LMM';
                end
                return;
            end

            % 2. Paired
            if obj.Paired
                 if is_normal
                    method = 'Paired T-test';
                    advice = 'Data is Normal and Paired.';
                else
                    method = 'Wilcoxon Signed-Rank';
                    advice = 'Data is Non-Normal and Paired.';
                 end
                 return;
            end

            % 3. Independent (Assuming 2 groups for simplicity in this demo)
            % Check number of groups if possible
             if is_normal && is_homo
                method = 'Independent T-test';
                advice = 'Normality and Homogeneity assumptions met.';
            elseif is_normal && ~is_homo
                method = 'Welch T-test'; % MATLAB ttest2 with 'Vartype','unequal'
                advice = 'Normality met, but Variance is unequal.';
            else
                method = 'Mann-Whitney U';
                advice = 'Normality assumption violated.';
            end
        end
    end
end
