%% ========================================================================
% mg_compare_all_methods_with_predictions_plot.m
%
% DESCRIPTION:
%   A final analysis script that consolidates all previous steps. It runs
%   the deterministic model, loads and averages existing Smoldyn data,
%   and applies post-processing techniques. It calculates NRMSE for four
%   methods and visualizes the results with both an NRMSE summary plot
%   and a detailed time-series prediction plot.
%
% =========================================================================

%% 0. Initialize Environment
clear;
close all;
clc;

%% 1. Parameter Setup (from Iteration 74)
fprintf('Using hyperparameters from optimized Iteration 74.\n');
config.N = 500;
config.k_on = 3.9338e-18;
config.k_off = 8.6418;
config.T = 1.9288;
config.washout1 = 500;
config.num_train_points = 500;
config.wheretostarttest = 1200;
config.num_test_points  = 500;
config.washout2 = config.wheretostarttest - (config.washout1 + config.num_train_points);
config.num_tot_points = config.washout1 + config.washout2 + config.num_train_points + config.num_test_points;
config.distance = 6.4287e-06;
config.D = 3.9627e-11;
config.N_min = 100;
config.N_max = 16838;
config.predictlength = 6;
config.lambda = 1e-6;
config.Nres_base = 50;
config.memorywindowlength = 5;
config.Nres = config.Nres_base * config.memorywindowlength;
config.dt = 0.001; % Added missing dt
fprintf('Effective reservoir size (Nres) set to %d.\n', config.Nres);

%% 2. Post-Processing Configuration (USER SETTINGS)
% --- Set the parameters for the post-processing methods here ---
config.uniformDelay = 8;
config.numReplicas = 4;
config.maxRandomDelay = 40;

%% 3. Auto-Detect, Load, and Average Data
% --- Load Ground Truth ---
MG_filename = 'MGseries_RK4_tau17_beta0.20_gamma0.1_n10_len5000_dt1.0.mat';
load(MG_filename, 'mackey_glass_series');
fprintf('Loaded ground truth Mackey-Glass data.\n');
series_segment = mackey_glass_series(1 : config.num_tot_points + config.predictlength);
series_normalized = (series_segment - min(series_segment)) / (max(series_segment) - min(series_segment));
mg_data.input_series  = series_normalized(1:end - config.predictlength);
mg_data.target_series = series_normalized((config.predictlength+1):end);
mg_data.train_target = mg_data.target_series(config.washout1+1 : config.washout1 + config.num_train_points);
mg_data.test_target  = mg_data.target_series(config.wheretostarttest + 1 : config.wheretostarttest + config.num_test_points);
mg_data.train_target = mg_data.train_target(:);
mg_data.test_target = mg_data.test_target(:);

% --- Find and Average Smoldyn Runs ---
parent_folder = uigetdir('', 'Select the Main Project Folder to Search For Results');
if parent_folder == 0, error('Analysis cancelled. No folder selected.'); end
search_pattern = fullfile(parent_folder, 'MG_Optimized_Iter74_*');
all_dirs = dir(search_pattern);
all_dirs = all_dirs([all_dirs.isdir]);
[~, idx] = sort([all_dirs.datenum], 'descend');
sorted_dirs = all_dirs(idx);
if numel(sorted_dirs) < 3, error('Could not find at least 3 folders matching the pattern "MG_Optimized_Iter74_..."'); end
data_folders = {fullfile(sorted_dirs(1).folder, sorted_dirs(1).name), ...
                fullfile(sorted_dirs(2).folder, sorted_dirs(2).name), ...
                fullfile(sorted_dirs(3).folder, sorted_dirs(3).name)};
fprintf('Found 3 recent data folders to analyze.\n');
all_runs_data = []; START_TIME = 0; STOP_TIME = START_TIME + length(mg_data.input_series) * config.T;
TIME_STEP_SMOL = 0.01; time_index_smol = START_TIME:TIME_STEP_SMOL:STOP_TIME;
for i = 1:numel(data_folders)
    file_search_pattern = fullfile(data_folders{i}, 'allmolecules_*.txt');
    file_list = dir(file_search_pattern);
    if isempty(file_list), continue; end
    filename = fullfile(file_list(1).folder, file_list(1).name);
    try
        data_table = importdata(filename, ' ', 1); receptor_active = data_table.data(:, end);
        time_original = data_table.data(:, 1);
        receptor_active_interp = interp1(time_original, receptor_active, time_index_smol, 'linear', 'extrap');
        all_runs_data(:, end+1) = receptor_active_interp;
    catch, end
end
Active_rec_avg = mean(all_runs_data, 2) / config.N;
fprintf('Averaging complete across %d successful run(s).\n', size(all_runs_data, 2));

%% 4. Generate Reservoir States for Both Models
% --- Deterministic ---
fprintf('Running deterministic simulation...\n');
N_i = config.N_min + (mg_data.input_series - min(mg_data.input_series)) / (max(mg_data.input_series) - min(mg_data.input_series)) * (config.N_max - config.N_min);
t_total = length(N_i) * config.T; t_values = 0:config.dt:t_total; c_values = zeros(size(t_values));
Tpeak = config.distance^2/(6*config.D); memory_length = round((100 * Tpeak)/config.T)*config.T;
for iSym = 1:length(N_i), t_symbol_start = (iSym-1)*config.T; t_memory_end = t_symbol_start+memory_length;
    idxRange = find(t_values > t_symbol_start & t_values <= t_memory_end); if isempty(idxRange), continue; end
    t_local = t_values(idxRange) - t_symbol_start;
    pulse = (N_i(iSym) ./ ((4*pi*config.D*t_local).^(3/2))) .* exp(-config.distance^2./(4*config.D*t_local));
    c_values(idxRange) = c_values(idxRange) + pulse;
end
n_values = zeros(size(t_values));
for idxT = 2:length(t_values), c_t = c_values(idxT-1); n_t = n_values(idxT-1);
    dn_dt = config.k_on*(config.N - n_t*config.N)*c_t - config.k_off*n_t*config.N;
    n_values(idxT) = n_values(idxT-1) + (dn_dt/config.N)*config.dt;
end
steps_per_symbol_det = config.T/config.dt;
reservoir_states_det = zeros(config.Nres, config.num_tot_points);
for iSym = 1:config.num_tot_points, t_start_symbol = (iSym-1)*config.T; start_idx = round(t_start_symbol/config.dt)+1;
    s_indices = round(start_idx - (config.memorywindowlength-1)*steps_per_symbol_det + (0:(config.Nres-1))*(steps_per_symbol_det/config.Nres)*config.memorywindowlength);
    if all(s_indices > 0), reservoir_states_det(:, iSym) = n_values(s_indices); end
end

% --- Stochastic ---
steps_per_symbol_smol = config.T/TIME_STEP_SMOL;
reservoir_states_smol = zeros(config.Nres, config.num_tot_points);
for iSym = 1:config.num_tot_points, t_start_symbol = (iSym-1)*config.T; start_idx = round(t_start_symbol/TIME_STEP_SMOL)+1;
    s_indices = round(start_idx - (config.memorywindowlength-1)*steps_per_symbol_smol + (0:(config.Nres-1))*(steps_per_symbol_smol/config.Nres)*config.memorywindowlength);
    if all(s_indices > 0 & s_indices <= length(Active_rec_avg)), reservoir_states_smol(:, iSym) = Active_rec_avg(s_indices); end
end

%% 5. Calculate NRMSE and Predictions for All Four Methods
results_nrmse = zeros(4, 1);
all_predictions = zeros(config.num_test_points, 4);
method_labels = {'Deterministic', 'Smoldyn (Baseline)', 'Smoldyn + Uniform', 'Smoldyn + Multi-Random'};

% --- 1. Deterministic ---
fprintf('\n===== 1. CALCULATING DETERMINISTIC PERFORMANCE =====\n');
W_out_det = train_readout_kl(reservoir_states_det, mg_data.train_target, config, 'none');
[~, results_nrmse(1), all_predictions(:,1)] = test_readout_kl(W_out_det, reservoir_states_det, mg_data, config, 'none');
fprintf('  -> NRMSE: %.4f\n', results_nrmse(1));

% --- 2. Smoldyn (Baseline) ---
fprintf('\n===== 2. CALCULATING SMOLDYN BASELINE PERFORMANCE =====\n');
W_out_smol_base = train_readout_kl(reservoir_states_smol, mg_data.train_target, config, 'none');
[~, results_nrmse(2), all_predictions(:,2)] = test_readout_kl(W_out_smol_base, reservoir_states_smol, mg_data, config, 'none');
fprintf('  -> NRMSE: %.4f\n', results_nrmse(2));

% --- 3. Smoldyn + Uniform Delay ---
fprintf('\n===== 3. CALCULATING SMOLDYN + UNIFORM DELAY PERFORMANCE =====\n');
W_out_smol_uni = train_readout_kl(reservoir_states_smol, mg_data.train_target, config, 'uniform');
[~, results_nrmse(3), all_predictions(:,3)] = test_readout_kl(W_out_smol_uni, reservoir_states_smol, mg_data, config, 'uniform');
fprintf('  -> NRMSE: %.4f\n', results_nrmse(3));

% --- 4. Smoldyn + Multi-Random Delay ---
fprintf('\n===== 4. CALCULATING SMOLDYN + MULTI-RANDOM DELAY PERFORMANCE =====\n');
W_out_smol_multi = train_readout_kl(reservoir_states_smol, mg_data.train_target, config, 'multi-random');
[~, results_nrmse(4), all_predictions(:,4)] = test_readout_kl(W_out_smol_multi, reservoir_states_smol, mg_data, config, 'multi-random');
fprintf('  -> NRMSE: %.4f\n', results_nrmse(4));

% --- Add this block to the end of Section 4 in your
%     mg_compare_all_methods_lineplot.m script ---

% fprintf('\nSaving averaged data for optimization...\n');
% save('averaged_smoldyn_data.mat', 'Active_rec_avg', 'reservoir_states_smol', 'mg_data', 'config');
% fprintf('Data saved to averaged_smoldyn_data.mat\n');

% --- You can run the script once to generate this file ---
%% 6. Visualize Final Comparison
fprintf('\nGenerating final comparison plots...\n');

% --- PLOT 1: NRMSE Summary (Line Plot) ---
figure('Name', 'NRMSE Comparison Across All Methods', 'Position', [200, 400, 900, 500]);
plot(1:4, results_nrmse, '-o', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', [0 0.4470 0.7410]);
xticks(1:4);
xticklabels(method_labels);
ylabel('NRMSE (Test Set)');
title('Comparison of NRMSE Across Different Methods');
grid on;
labels = string(round(results_nrmse, 4));
text(1:4, results_nrmse, labels, 'VerticalAlignment','bottom', 'HorizontalAlignment','right', 'FontSize', 10);
ylim([0, max(results_nrmse) * 1.2]);
set(gca, 'FontSize', 11);

% --- PLOT 2: Time Series Predictions ---
figure('Name', 'Time Series Prediction Comparison', 'Position', [250, 100, 1200, 600]);
% To avoid clutter, you can plot a subset of the points, e.g., plot_points = 1:200;
plot_points = 1:length(mg_data.test_target); 
plot(mg_data.test_target(plot_points), 'k', 'LineWidth', 2.5, 'DisplayName', 'Ground Truth');
hold on;
plot(all_predictions(plot_points, 1), 'LineWidth', 1.5, 'DisplayName', sprintf('Deterministic (NRMSE=%.4f)', results_nrmse(1)));
plot(all_predictions(plot_points, 2), 'LineWidth', 1.5, 'DisplayName', sprintf('Smoldyn Baseline (NRMSE=%.4f)', results_nrmse(2)));
plot(all_predictions(plot_points, 3), '--', 'LineWidth', 2, 'DisplayName', sprintf('Smoldyn + Uniform (NRMSE=%.4f)', results_nrmse(3)));
plot(all_predictions(plot_points, 4), ':', 'LineWidth', 2, 'DisplayName', sprintf('Smoldyn + Multi-Random (NRMSE=%.4f)', results_nrmse(4)));
hold off;
title('Comparison of Model Predictions on Mackey-Glass Test Set');
xlabel('Time Step in Test Set');
ylabel('Normalized Value');
legend('show', 'Location', 'northwest');
grid on;
axis tight;
set(gca, 'FontSize', 11);

fprintf('Analysis complete.\n');

%% ========================================================================
%                    --- LOCAL HELPER FUNCTIONS ---
% =========================================================================

function [augmented_states, config] = apply_post_processing(reservoir_states, config)
    % This helper function is now just a wrapper for the real one
    [augmented_states, config] = apply_post_processing_internal(reservoir_states, config, config.postProcessingMethod);
end

function [augmented_states, config] = apply_post_processing_internal(reservoir_states, config, method)
    fprintf('  Applying post-processing method: %s\n', method);
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
    end, fprintf('  -> New feature dimension O = %d\n', config.feature_dimension);
end

function W_out = train_readout_kl(augmented_states, train_target, config, method_override)
    if nargin > 3 && ~isempty(method_override)
        [processed_states, temp_config] = apply_post_processing_internal(augmented_states, config, method_override);
    else, processed_states = augmented_states; temp_config = config; end
    train_states_raw = processed_states(:, temp_config.washout1+1 : temp_config.washout1+temp_config.num_train_points);
    valid_cols = ~any(isnan(train_states_raw), 1);
    train_states = train_states_raw(:, valid_cols); y_train = train_target(valid_cols);
    X_train = [train_states; ones(1, size(train_states, 2))];
    W_out = pinv(X_train * X_train' + temp_config.lambda * eye(temp_config.feature_dimension + 1)) * (X_train * y_train);
end

function [nrmse_train, nrmse_test, y_test_hat] = test_readout_kl(W_out, augmented_states, mg_data, config, method_override)
    if nargin > 4 && ~isempty(method_override)
        [processed_states, temp_config] = apply_post_processing_internal(augmented_states, config, method_override);
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
    y_test_hat = NaN(length(mg_data.test_target),1); % Initialize with NaNs
    if isempty(y_test_valid), nrmse_test = NaN;
    else, X_test = [test_states; ones(1, size(test_states, 2))]; 
        y_test_hat_valid = X_test' * W_out;
        y_test_hat(valid_test_cols) = y_test_hat_valid; % Place valid predictions
        nrmse_test = sqrt(mean((y_test_hat_valid - y_test_valid).^2)) / std(y_test_valid);
    end
end