%% ========================================================================
%  ANALYSIS SCRIPT: SINE-TO-SQUARE - WITH STOCHASTIC FILTERING
% ========================================================================
%
% This script compares four signals for the Sine-to-Square task:
%   1. Ground Truth
%   2. Deterministic Prediction
%   3. Averaged Stochastic Prediction (Raw)
%   4. Filtered Stochastic Prediction (Smoothed)
%
% It uses the optimal hyperparameters from Iteration 23.
%
% =========================================================================

%% 0. Initialize Environment & Auto-Detect Data Folders
clear; close all; clc;
fprintf('Starting analysis of SINE-TO-SQUARE with filtering...\n');
parent_folder = uigetdir('', 'Select the Main Project Folder Containing Smoldyn Runs');
if parent_folder == 0, error('Analysis cancelled.'); end
search_pattern = fullfile(parent_folder, 'Sine_Smoldyn_Iter23_*');
all_dirs = dir(search_pattern);
all_dirs = all_dirs([all_dirs.isdir]);
[~, idx] = sort([all_dirs.datenum], 'descend');
sorted_dirs = all_dirs(idx);
if numel(sorted_dirs) < 3, warning('Fewer than 3 folders found. Using %d.', numel(sorted_dirs)); end
num_folders_to_process = min(3, numel(sorted_dirs));
data_folders = arrayfun(@(x) fullfile(x.folder, x.name), sorted_dirs(1:num_folders_to_process), 'UniformOutput', false);
fprintf('Found %d recent data folder(s) to analyze.\n', num_folders_to_process);

%% 1. Parameter Setup (from Iteration 23)
fprintf('Using hyperparameters from optimized Iteration 23.\n');
N = 500; k_on = 1.98e-17; k_off = 6.44; T = 1.64;
washout1 = 500; num_train_points = 2000; wheretostarttest = 3000; num_test_points = 1000;
washout2 = wheretostarttest - (washout1 + num_train_points);
num_tot_points = washout1 + washout2 + num_train_points + num_test_points;
distance = 4.53e-06; D = 1.17e-11; N_max = 9564; N_min = 100;
dt = 0.001; memlengthsweep = 100; lambda = 1e-6;
Nres_init = 100; memorywindowlength = 5; Nres = Nres_init * memorywindowlength;

% --- GIVES YOU THE POWER TO CHANGE THE SMOOTHING WINDOW ---
movmean_window_stoch = 6000; % <-- TUNE THIS VALUE
fprintf('Stochastic filtering window set to %d.\n', movmean_window_stoch);
% -----------------------------------------------------------

%% 2. Load and Prepare Sine-to-Square Ground Truth Data
load('SINEseries_len5000_p25.mat', 'input_sine_series', 'target_square_series');
input_series = input_sine_series(1:num_tot_points);
target_series = target_square_series(1:num_tot_points);
y_train = target_series(washout1+1 : washout1 + num_train_points);
y_test  = target_series(wheretostarttest+1 : wheretostarttest + num_test_points);
y_train = y_train(:); y_test = y_test(:);

%% 3. Load, Average, and Filter Stochastic (Smoldyn) Data
fprintf('Loading, averaging, and filtering data from %d Smoldyn run(s)...\n', num_folders_to_process);
all_runs_data = []; TIME_STEP_SMOL = 0.01;
STOP_TIME = length(input_series) * T;
time_index_smol = (0:TIME_STEP_SMOL:STOP_TIME)';
for i = 1:num_folders_to_process
    file_list = dir(fullfile(data_folders{i}, 'allmolecules_*.txt'));
    if isempty(file_list), warning('No data file in %s. Skipping.', data_folders{i}); continue; end
    filename = fullfile(file_list(1).folder, file_list(1).name);
    try
        data_table = importdata(filename, ' ', 1);
        receptor_active_interp = interp1(data_table.data(:, 1), data_table.data(:, 4), time_index_smol, 'linear', 'extrap');
        all_runs_data(:, end+1) = receptor_active_interp;
    catch ME, warning('Could not process file: %s', filename); end
end
if isempty(all_runs_data), error('Failed to load any valid Smoldyn data.'); end

% --- CREATE BOTH RAW AND FILTERED SIGNALS ---
Active_rec_avg_raw = mean(all_runs_data, 2) / N;
Active_rec_avg_filtered = movmean(Active_rec_avg_raw, [movmean_window_stoch 0]);
fprintf('Averaging and filtering complete.\n');
% ----------------------------------------------

%% 4. Run Deterministic Model Simulation
fprintf('Running deterministic simulation...\n');
N_i = N_min + input_series * (N_max - N_min);
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
X_train_det=zeros(Nres,num_train_points); for i=1:num_train_points, t_start=(washout1+i-1)*T; s_idx=round(t_start/dt)+1; idx=round(s_idx-(memorywindowlength-1)*steps_per_symbol_det+(0:(Nres-1))*(steps_per_symbol_det/Nres)*memorywindowlength); if all(idx>0 & idx<=length(n_values)), X_train_det(:,i)=n_values(idx); end, end
X_train_det_bias=[X_train_det; ones(1,num_train_points)]; W_out_det = (X_train_det_bias * X_train_det_bias' + lambda * eye(Nres+1)) \ (X_train_det_bias * y_train);
X_test_det=zeros(Nres,num_test_points); for i=1:num_test_points, t_start=(wheretostarttest+i-1)*T; s_idx=round(t_start/dt)+1; idx=round(s_idx-(memorywindowlength-1)*steps_per_symbol_det+(0:(Nres-1))*(steps_per_symbol_det/Nres)*memorywindowlength); if all(idx>0 & idx<=length(n_values)), X_test_det(:,i)=n_values(idx); end, end
X_test_det_bias=[X_test_det; ones(1,num_test_points)]; y_hat_det_test = X_test_det_bias' * W_out_det;
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
figure('Name', 'Sine-to-Square: Model Comparison', 'Position', [100, 100, 1200, 600]);
plot(y_test, 'k', 'LineWidth', 2.5, 'DisplayName', 'Ground Truth (Square Wave)');
hold on;
plot(y_hat_det_test, 'b', 'LineWidth', 1.5, 'DisplayName', ...
    sprintf('Deterministic (NRMSE = %.4f)', nrmse_det));
plot(y_hat_stoch_raw, 'r-', 'LineWidth', 1.5, 'DisplayName', ...
    sprintf('Averaged Stochastic - Raw (NRMSE = %.4f)', nrmse_stoch_raw));
plot(y_hat_stoch_filtered, 'Color', [0.4940 0.1840 0.5560], 'LineWidth', 2, 'DisplayName', ...
    sprintf('Averaged Stochastic - Filtered (NRMSE = %.4f)', nrmse_stoch_filtered));
hold off;
title('Model Comparison on Sine-to-Square Test Set (Iter 23)');
xlabel('Time Step in Test Set');
ylabel('Value');
legend('show', 'Location', 'northwest');
grid on; axis tight;
fprintf('Analysis complete.\n');