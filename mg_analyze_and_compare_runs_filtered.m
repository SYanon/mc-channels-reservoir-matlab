%% ========================================================================
%  ANALYSIS SCRIPT: MG FORECASTING - WITH STOCHASTIC FILTERING
% ========================================================================
%
% This script compares four signals for the Mackey-Glass forecasting task:
%   1. Ground Truth
%   2. Deterministic Prediction
%   3. Averaged Stochastic Prediction (Raw)
%   4. Filtered Stochastic Prediction (Smoothed)
%
% It uses the optimal hyperparameters from Iteration 74.
%
% =========================================================================

%% 0. Initialize Environment & Auto-Detect Data Folders
clear; close all; clc;
fprintf('Starting analysis of MG FORECASTING with filtering...\n');
parent_folder = uigetdir('', 'Select the Main Project Folder Containing Smoldyn Runs');
if parent_folder == 0, error('Analysis cancelled.'); end
search_pattern = fullfile(parent_folder, 'MG_Optimized_Iter74_*');
all_dirs = dir(search_pattern);
all_dirs = all_dirs([all_dirs.isdir]);
[~, idx] = sort([all_dirs.datenum], 'descend');
sorted_dirs = all_dirs(idx);
if numel(sorted_dirs) < 3, warning('Fewer than 3 folders found. Using %d.', numel(sorted_dirs)); end
num_folders_to_process = min(3, numel(sorted_dirs));
data_folders = arrayfun(@(x) fullfile(x.folder, x.name), sorted_dirs(1:num_folders_to_process), 'UniformOutput', false);
fprintf('Found %d recent data folder(s) to analyze.\n', num_folders_to_process);

%% 1. Parameter Setup (from Iteration 74)
fprintf('Using hyperparameters from optimized Iteration 74.\n');
N = 500; k_on = 3.9338e-18; k_off = 8.6418; T = 1.9288;
washout1 = 500; num_train_points = 500; wheretostarttest = 1200; num_test_points = 500;
washout2 = wheretostarttest - (washout1 + num_train_points);
num_tot_points = washout1 + washout2 + num_train_points + num_test_points;
distance = 6.4287e-06; D = 3.9627e-11;
N_min = 100; N_max = 16838; predictlength = 6;
lambda = 1e-6; dt = 0.001; memlengthsweep = 100;
Nres_base = 50; memorywindowlength = 5; Nres = Nres_base * memorywindowlength;

% --- GIVES YOU THE POWER TO CHANGE THE SMOOTHING WINDOW ---
movmean_window_stoch = 2000; % <-- TUNE THIS VALUE
fprintf('Stochastic filtering window set to %d.\n', movmean_window_stoch);
% -----------------------------------------------------------

%% 2. Load and Prepare Mackey-Glass Ground Truth Data
MG_filename = 'MGseries_RK4_tau17_beta0.20_gamma0.1_n10_len5000_dt1.0.mat';
load(MG_filename, 'mackey_glass_series');
series_segment = mackey_glass_series(1 : num_tot_points + predictlength);
series_normalized = (series_segment - min(series_segment)) / (max(series_segment) - min(series_segment));
input_series  = series_normalized(1:end - predictlength);
target_series = series_normalized((predictlength+1):end);
y_train = target_series(washout1+1 : washout1 + num_train_points);
y_test  = target_series(wheretostarttest+1 : wheretostarttest + num_test_points);
y_train = y_train(:); y_test = y_test(:);

%% 3. Load, Average, and Filter Stochastic (Smoldyn) Data
fprintf('Loading, averaging, and filtering data from %d Smoldyn run(s)...\n', num_folders_to_process);
all_runs_data = []; TIME_STEP_SMOL = 0.01;
STOP_TIME = length(input_series) * T;
time_index_smol = 0:TIME_STEP_SMOL:STOP_TIME;
for i = 1:num_folders_to_process
    file_list = dir(fullfile(data_folders{i}, 'allmolecules_*.txt'));
    if isempty(file_list), warning('No data file in %s. Skipping.', data_folders{i}); continue; end
    filename = fullfile(file_list(1).folder, file_list(1).name);
    try
        data_table = importdata(filename, ' ', 1);
        receptor_active_interp = interp1(data_table.data(:, 1), data_table.data(:, end), time_index_smol, 'linear', 'extrap');
        all_runs_data(:, end+1) = receptor_active_interp;
    catch ME, warning('Could not process file: %s', filename); end
end
if isempty(all_runs_data), error('Failed to load any valid Smoldyn data.'); end

% --- CREATE BOTH RAW AND FILTERED SIGNALS ---
Active_rec_avg_raw = mean(all_runs_data, 2) / N; % This is the original, noisy signal
Active_rec_avg_filtered = movmean(Active_rec_avg_raw, [movmean_window_stoch 0]); % This is the new, smoothed signal
fprintf('Averaging and filtering complete.\n');
% ----------------------------------------------

%% 4. Run Deterministic Model Simulation
% This section is condensed for brevity, logic remains the same.
fprintf('Running deterministic simulation...\n');
N_i = N_min + (input_series - min(input_series)) / (max(input_series) - min(input_series)) * (N_max - N_min);
t_total = length(N_i) * T; t_values = 0:dt:t_total; c_values = zeros(size(t_values));
Tpeak = distance^2/(6*D); memory_length = round((memlengthsweep * Tpeak)/T)*T;
for iSym = 1:length(N_i)
    t_symbol_start = (iSym - 1) * T; t_memory_end = t_symbol_start + memory_length;
    idxRange = find(t_values > t_symbol_start & t_values <= t_memory_end);
    if isempty(idxRange), continue; end
    t_local = t_values(idxRange) - t_symbol_start;
    pulse = (N_i(iSym) ./ ((4*pi*D*t_local).^(3/2))) .* exp(-distance^2./(4*D*t_local));
    c_values(idxRange) = c_values(idxRange) + pulse;
end
n_values = zeros(size(t_values));
for idxT = 2:length(t_values), dn_dt = k_on*(N - n_values(idxT-1)*N)*c_values(idxT-1) - k_off*n_values(idxT-1)*N; n_values(idxT) = n_values(idxT-1) + (dn_dt/N)*dt; end
fprintf('Deterministic simulation complete.\n');

%% 5. Process Models and Calculate NRMSE

% --- 5A: Deterministic Model ---
fprintf('Training and testing deterministic model...\n');
steps_per_symbol_det = T/dt;
X_train_det = zeros(Nres, num_train_points);
for i=1:num_train_points, t_start_symbol=(washout1+i-1)*T; start_idx=round(t_start_symbol/dt)+1; indices=round(start_idx-(memorywindowlength-1)*steps_per_symbol_det+(0:(Nres-1))*(steps_per_symbol_det/Nres)*memorywindowlength); if all(indices>0), X_train_det(:,i)=n_values(indices); end, end
X_train_det_bias = [X_train_det; ones(1, num_train_points)];
W_out_det = (X_train_det_bias * X_train_det_bias' + lambda * eye(Nres+1)) \ (X_train_det_bias * y_train);
X_test_det = zeros(Nres, num_test_points);
for i=1:num_test_points, t_start_symbol=(wheretostarttest+i-1)*T; start_idx=round(t_start_symbol/dt)+1; indices=round(start_idx-(memorywindowlength-1)*steps_per_symbol_det+(0:(Nres-1))*(steps_per_symbol_det/Nres)*memorywindowlength); if all(indices>0), X_test_det(:,i)=n_values(indices); end, end
X_test_det_bias = [X_test_det; ones(1, num_test_points)];
y_hat_det_test = X_test_det_bias' * W_out_det;
nrmse_det = sqrt(mean((y_hat_det_test - y_test).^2)) / std(y_test);
fprintf('  -> Deterministic NRMSE: %.4f\n', nrmse_det);

% --- Helper function for stochastic training/testing ---
function [y_hat, nrmse] = train_and_test_stochastic(reservoir_signal, y_train, y_test, params)
    steps_per_symbol = params.T / params.TIME_STEP_SMOL;
    X_train = zeros(params.Nres, params.num_train_points);
    for i = 1:params.num_train_points
        t_start = (params.washout1 + i - 1) * params.T;
        start_idx = round(t_start/params.TIME_STEP_SMOL) + 1;
        indices = round(start_idx - (params.memorywindowlength-1)*steps_per_symbol + (0:(params.Nres-1))*(steps_per_symbol/params.Nres)*params.memorywindowlength);
        if all(indices > 0 & indices <= length(reservoir_signal)), X_train(:,i) = reservoir_signal(indices); end
    end
    X_train_bias = [X_train; ones(1, params.num_train_points)];
    W_out = (X_train_bias * X_train_bias' + params.lambda * eye(params.Nres+1)) \ (X_train_bias * y_train);
    
    X_test = zeros(params.Nres, params.num_test_points);
    for i = 1:params.num_test_points
        t_start = (params.wheretostarttest + i - 1) * params.T;
        start_idx = round(t_start/params.TIME_STEP_SMOL) + 1;
        indices = round(start_idx - (params.memorywindowlength-1)*steps_per_symbol + (0:(params.Nres-1))*(steps_per_symbol/params.Nres)*params.memorywindowlength);
        if all(indices > 0 & indices <= length(reservoir_signal)), X_test(:,i) = reservoir_signal(indices); end
    end
    X_test_bias = [X_test; ones(1, params.num_test_points)];
    y_hat = X_test_bias' * W_out;
    nrmse = sqrt(mean((y_hat - y_test).^2)) / std(y_test);
end

params = struct('T', T, 'TIME_STEP_SMOL', TIME_STEP_SMOL, 'Nres', Nres, 'num_train_points', num_train_points, ...
                'num_test_points', num_test_points, 'washout1', washout1, 'wheretostarttest', wheretostarttest, ...
                'memorywindowlength', memorywindowlength, 'lambda', lambda);

% --- 5B: Averaged Stochastic Model (Raw) ---
fprintf('Training and testing RAW averaged stochastic model...\n');
[y_hat_stoch_raw, nrmse_stoch_raw] = train_and_test_stochastic(Active_rec_avg_raw, y_train, y_test, params);
fprintf('  -> Raw Stochastic NRMSE: %.4f\n', nrmse_stoch_raw);

% --- 5C: Filtered Stochastic Model ---
fprintf('Training and testing FILTERED stochastic model...\n');
[y_hat_stoch_filtered, nrmse_stoch_filtered] = train_and_test_stochastic(Active_rec_avg_filtered, y_train, y_test, params);
fprintf('  -> Filtered Stochastic NRMSE: %.4f\n', nrmse_stoch_filtered);

%% 6. Visualize Final Comparison
fprintf('Generating final comparison plot...\n');
figure('Name', 'MG Forecasting: Model Comparison', 'Position', [100, 100, 1200, 600]);
plot(y_test, 'k', 'LineWidth', 2.5, 'DisplayName', 'Ground Truth');
hold on;
plot(y_hat_det_test, 'b', 'LineWidth', 1.5, 'DisplayName', ...
    sprintf('Deterministic (NRMSE = %.4f)', nrmse_det));
plot(y_hat_stoch_raw, 'r-', 'LineWidth', 1.5, 'DisplayName', ...
    sprintf('Averaged Stochastic - Raw (NRMSE = %.4f)', nrmse_stoch_raw));
plot(y_hat_stoch_filtered, 'Color', [0.4940 0.1840 0.5560], 'LineWidth', 2, 'DisplayName', ... % Purple
    sprintf('Averaged Stochastic - Filtered (NRMSE = %.4f)', nrmse_stoch_filtered));
hold off;
title('Model Comparison on Mackey-Glass Test Set (Iter 74)');
xlabel('Time Step in Test Set');
ylabel('Normalized Value');
legend('show', 'Location', 'northwest');
grid on; axis tight;
fprintf('Analysis complete.\n');