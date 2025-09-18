%% ========================================================================
% optimize_kl_params_for_sine.m
%
% DESCRIPTION:
%   Loads pre-averaged Smoldyn data for the SINE-TO-SQUARE task and sweeps
%   through a range of delay parameters for the 'uniform' and 'multi-random'
%   post-processing methods to find the optimal values for minimizing NRMSE.
%
% =========================================================================

%% 0. Initialize Environment
clear;
close all;
clc;

%% 1. Configuration (USER SETTINGS)
% --- Define the range of delay parameters you want to test ---
uniform_delay_sweep = [1, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30];
max_random_delay_sweep = [10, 20, 30, 40, 50, 60, 80, 100];

%% 2. Load Pre-Averaged Data
fprintf('Loading pre-averaged Sine-to-Square Smoldyn data...\n');
[file, path] = uigetfile('*.mat', 'Select the averaged_sine_smoldyn_data.mat file');
if isequal(file, 0), error('File selection cancelled.'); end
load(fullfile(path, file), 'reservoir_states_smol', 'mg_data', 'config');
fprintf('Data loaded successfully.\n');

%% 3. Sweep Uniform Delay Parameter
fprintf('\n===== 1. SWEEPING UNIFORM DELAY PARAMETER =====\n');
nrmse_results_uniform = zeros(size(uniform_delay_sweep));
for i = 1:length(uniform_delay_sweep)
    current_delay = uniform_delay_sweep(i);
    fprintf('  Testing uniform delay = %d...\n', current_delay);
    temp_config = config;
    temp_config.uniformDelay = current_delay;
    [~, nrmse_results_uniform(i)] = test_readout_kl(...
        train_readout_kl(reservoir_states_smol, mg_data.train_target, temp_config, 'uniform'), ...
        reservoir_states_smol, mg_data, temp_config, 'uniform');
end
[min_nrmse_uni, idx_uni] = min(nrmse_results_uniform);
best_delay_uni = uniform_delay_sweep(idx_uni);

%% 4. Sweep Multi-Random Max Delay Parameter
fprintf('\n===== 2. SWEEPING MULTI-RANDOM MAX DELAY PARAMETER =====\n');
nrmse_results_multirandom = zeros(size(max_random_delay_sweep));
for i = 1:length(max_random_delay_sweep)
    current_max_delay = max_random_delay_sweep(i);
    fprintf('  Testing multi-random max delay = %d...\n', current_max_delay);
    temp_config = config;
    temp_config.maxRandomDelay = current_max_delay;
    [~, nrmse_results_multirandom(i)] = test_readout_kl(...
        train_readout_kl(reservoir_states_smol, mg_data.train_target, temp_config, 'multi-random'), ...
        reservoir_states_smol, mg_data, temp_config, 'multi-random');
end
[min_nrmse_multi, idx_multi] = min(nrmse_results_multirandom);
best_delay_multi = max_random_delay_sweep(idx_multi);

%% 5. Visualize and Report Results
fprintf('\nGenerating optimization plots for Sine-to-Square task...\n');
figure('Name', 'KL Parameter Optimization for Sine-to-Square Smoldyn Data', 'Position', [100, 100, 1200, 500]);

% --- Subplot for Uniform Delay ---
subplot(1, 2, 1);
plot(uniform_delay_sweep, nrmse_results_uniform, '-o', 'LineWidth', 2);
hold on;
plot(best_delay_uni, min_nrmse_uni, 'r*', 'MarkerSize', 12, 'LineWidth', 2);
grid on; xlabel('Uniform Delay Steps'); ylabel('NRMSE (Test Set)');
title('Uniform Delay Performance');
legend('NRMSE', sprintf('Best (NRMSE=%.4f)', min_nrmse_uni), 'Location', 'best');

% --- Subplot for Multi-Random Delay ---
subplot(1, 2, 2);
plot(max_random_delay_sweep, nrmse_results_multirandom, '-o', 'LineWidth', 2, 'Color', [0.8500 0.3250 0.0980]);
hold on;
plot(best_delay_multi, min_nrmse_multi, 'r*', 'MarkerSize', 12, 'LineWidth', 2);
grid on; xlabel('Max Random Delay Steps'); ylabel('NRMSE (Test Set)');
title('Multi-Random Delay Performance');
legend('NRMSE', sprintf('Best (NRMSE=%.4f)', min_nrmse_multi), 'Location', 'best');

% --- Final Summary in Command Window ---
fprintf('\n================== SINE-TO-SQUARE OPTIMIZATION COMPLETE ==================\n');
fprintf('Best Uniform Delay:       %d steps, giving NRMSE = %.4f\n', best_delay_uni, min_nrmse_uni);
fprintf('Best Multi-Random Max Delay: %d steps, giving NRMSE = %.4f\n', best_delay_multi, min_nrmse_multi);
fprintf('========================================================================\n');


%% ========================================================================
%                    --- LOCAL HELPER FUNCTIONS ---
% =========================================================================
function [augmented_states, config] = apply_post_processing_internal(reservoir_states, config, method)
    num_total_points = size(reservoir_states, 2);
    switch method, case 'none', augmented_states = reservoir_states; config.feature_dimension = config.Nres; 
        case 'uniform', delay = config.uniformDelay; if delay < 1, augmented_states = reservoir_states; config.feature_dimension = config.Nres; return; end
            original_part = reservoir_states; delayed_part = NaN(size(reservoir_states));
            delayed_part(:, (1+delay):end) = reservoir_states(:, 1:(end-delay));
            augmented_states = [original_part; delayed_part]; config.feature_dimension = 2 * config.Nres;
        case 'multi-random', N_replicas = config.numReplicas; R_max = config.maxRandomDelay;
            if N_replicas <= 1, augmented_states = reservoir_states; config.feature_dimension = config.Nres; return; end
            all_replicas = [];
            for i_rep = 1:N_replicas, delays = randi([0 R_max], [config.Nres, 1]);
                current_replica = NaN(config.Nres, num_total_points);
                for j_feature = 1:config.Nres, d = delays(j_feature);
                    if d == 0, current_replica(j_feature, :) = reservoir_states(j_feature, :);
                    else, current_replica(j_feature, (1+d):end) = reservoir_states(j_feature, 1:(end-d)); end
                end, all_replicas = [all_replicas; current_replica];
            end, augmented_states = all_replicas; config.feature_dimension = N_replicas * config.Nres;
        otherwise, error('Unknown post-processing method: %s', method);
    end
end

function W_out = train_readout_kl(augmented_states, train_target, config, method_override)
    if nargin > 3 && ~isempty(method_override), [processed_states, temp_config] = apply_post_processing_internal(augmented_states, config, method_override);
    else, processed_states = augmented_states; temp_config = config; end
    train_states_raw = processed_states(:, temp_config.washout1+1 : temp_config.washout1+temp_config.num_train_points);
    valid_cols = ~any(isnan(train_states_raw), 1);
    train_states = train_states_raw(:, valid_cols); y_train = train_target(valid_cols);
    X_train = [train_states; ones(1, size(train_states, 2))];
    W_out = pinv(X_train * X_train' + temp_config.lambda * eye(temp_config.feature_dimension + 1)) * (X_train * y_train);
end

function [nrmse_train, nrmse_test] = test_readout_kl(W_out, augmented_states, mg_data, config, method_override)
    if nargin > 4 && ~isempty(method_override), [processed_states, temp_config] = apply_post_processing_internal(augmented_states, config, method_override);
    else, processed_states = augmented_states; temp_config = config; end
    train_states_raw = processed_states(:, temp_config.washout1+1 : temp_config.washout1+temp_config.num_train_points);
    valid_train_cols = ~any(isnan(train_states_raw), 1); train_states = train_states_raw(:, valid_train_cols);
    y_train_valid = mg_data.train_target(valid_train_cols);
    if isempty(y_train_valid), nrmse_train = NaN;
    else, X_train = [train_states; ones(1, size(train_states, 2))]; y_train_hat = X_train' * W_out;
        nrmse_train = sqrt(mean((y_train_hat - y_train_valid).^2)) / std(y_train_valid);
    end
    test_indices = temp_config.wheretostarttest + 1 : temp_config.wheretostarttest + temp_config.num_test_points;
    test_states_raw = processed_states(:, test_indices);
    valid_test_cols = ~any(isnan(test_states_raw), 1); test_states = test_states_raw(:, valid_test_cols);
    y_test_valid = mg_data.test_target(valid_test_cols);
    if isempty(y_test_valid), nrmse_test = NaN;
    else, X_test = [test_states; ones(1, size(test_states, 2))]; y_test_hat = X_test' * W_out;
        nrmse_test = sqrt(mean((y_test_hat - y_test_valid).^2)) / std(y_test_valid);
    end
end