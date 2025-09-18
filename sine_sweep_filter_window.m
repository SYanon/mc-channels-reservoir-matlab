%% ========================================================================
%  SWEEP SCRIPT: FIND OPTIMAL FILTER WINDOW FOR SINE-TO-SQUARE
% ========================================================================
clear; close all; clc;

%% --- 1. SETUP THE SWEEP ---
% Define the range of window sizes you want to test. 
% This task is sensitive; smaller windows are recommended.
window_sizes_to_test = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80,100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000];
nrmse_results = []; % To store the NRMSE for each window size

fprintf('Starting filter window sweep for Sine-to-Square...\n');
fprintf('Testing %d different window sizes.\n', numel(window_sizes_to_test));

%% --- 2. LOAD DATA AND RUN DETERMINISTIC SIM (Done once) ---
% Load Smoldyn Data
parent_folder = uigetdir('', 'Select the Main Project Folder (Sine_Smoldyn_Iter23 runs)');
if parent_folder == 0, error('Analysis cancelled.'); end
search_pattern = fullfile(parent_folder, 'Sine_Smoldyn_Iter23_*');
all_dirs = dir(search_pattern); all_dirs = all_dirs([all_dirs.isdir]);
[~, idx] = sort([all_dirs.datenum], 'descend'); sorted_dirs = all_dirs(idx);
num_folders_to_process = min(3, numel(sorted_dirs));
data_folders = arrayfun(@(x) fullfile(x.folder, x.name), sorted_dirs(1:num_folders_to_process), 'UniformOutput', false);

% Load Parameters and Ground Truth
params.N = 500; params.k_on = 1.98e-17; params.k_off = 6.44; params.T = 1.64;
params.washout1 = 500; params.num_train_points = 2000; params.wheretostarttest = 3000; params.num_test_points = 1000;
params.num_tot_points = params.washout1 + (params.wheretostarttest - (params.washout1+params.num_train_points)) + params.num_train_points + params.num_test_points;
params.lambda = 1e-6;
params.Nres_init = 100; params.memorywindowlength = 5; params.Nres = params.Nres_init * params.memorywindowlength;
params.TIME_STEP_SMOL = 0.01;

load('SINEseries_len5000_p25.mat', 'input_sine_series', 'target_square_series');
input_series = input_sine_series(1:params.num_tot_points);
target_series = target_square_series(1:params.num_tot_points);
y_train = target_series(params.washout1+1 : params.washout1 + params.num_train_points);
y_test  = target_series(params.wheretostarttest+1 : params.wheretostarttest + params.num_test_points);
y_train = y_train(:); y_test = y_test(:);

% Pre-load and average all Smoldyn runs
all_runs_data = []; 
STOP_TIME = length(input_series) * params.T; time_index_smol = (0:params.TIME_STEP_SMOL:STOP_TIME)';
for i = 1:num_folders_to_process
    file_list = dir(fullfile(data_folders{i}, 'allmolecules_*.txt'));
    if isempty(file_list), continue; end
    filename = fullfile(file_list(1).folder, file_list(1).name);
    data_table = importdata(filename, ' ', 1);
    receptor_active_interp = interp1(data_table.data(:, 1), data_table.data(:, 4), time_index_smol, 'linear', 'extrap');
    all_runs_data(:, end+1) = receptor_active_interp;
end
Active_rec_avg_raw = mean(all_runs_data, 2) / params.N;

%% --- 3. LOOP THROUGH WINDOW SIZES ---
for i = 1:length(window_sizes_to_test)
    movmean_window = window_sizes_to_test(i);
    
    % Apply the filter for the current iteration
    if movmean_window == 1
        Active_rec_avg_filtered = Active_rec_avg_raw; % No filtering
    else
        Active_rec_avg_filtered = movmean(Active_rec_avg_raw, [movmean_window 0]);
    end
    
    % Train and Test the filtered model
    [~, nrmse] = train_and_test_stochastic(Active_rec_avg_filtered, y_train, y_test, params);
    nrmse_results(i) = nrmse;
    
    fprintf('  Window Size: %d -> Filtered NRMSE: %.4f\n', movmean_window, nrmse);
end

%% --- 4. ANALYZE AND PLOT RESULTS ---
[min_nrmse, min_idx] = min(nrmse_results);
optimal_window = window_sizes_to_test(min_idx);

fprintf('\n--- SWEEP COMPLETE ---\n');
fprintf('Optimal Filter Window: %d\n', optimal_window);
fprintf('Lowest NRMSE Achieved: %.4f\n\n', min_nrmse);

% Generate plot
figure('Name', 'NRMSE vs. Filter Window (Sine-to-Square)');
plot(window_sizes_to_test, nrmse_results, '-o', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot(optimal_window, min_nrmse, 'r*', 'MarkerSize', 12, 'LineWidth', 2, ...
    'DisplayName', sprintf('Optimal (%d)', optimal_window));
title('Filter Window Optimization for Sine-to-Square');
xlabel('Moving Average Window Size');
ylabel('NRMSE of Filtered Stochastic Model');
grid on;
legend('show', 'Location', 'best');
xticks(window_sizes_to_test);

%% --- Helper Function ---
function [y_hat, nrmse] = train_and_test_stochastic(reservoir_signal, y_train, y_test, params)
    steps_per_symbol = params.T / params.TIME_STEP_SMOL;
    X_train = zeros(params.Nres, params.num_train_points);
    for i = 1:params.num_train_points, t_start = (params.washout1 + i - 1) * params.T; start_idx = round(t_start/params.TIME_STEP_SMOL) + 1; indices = round(start_idx - (params.memorywindowlength-1)*steps_per_symbol + (0:(params.Nres-1))*(steps_per_symbol/params.Nres)*params.memorywindowlength); if all(indices > 0 & indices <= length(reservoir_signal)), X_train(:,i) = reservoir_signal(indices); end, end
    X_train_bias = [X_train; ones(1, params.num_train_points)];
    W_out = (X_train_bias * X_train_bias' + params.lambda * eye(params.Nres+1)) \ (X_train_bias * y_train);
    X_test = zeros(params.Nres, params.num_test_points);
    for i = 1:params.num_test_points, t_start = (params.wheretostarttest + i - 1) * params.T; start_idx = round(t_start/params.TIME_STEP_SMOL) + 1; indices = round(start_idx - (params.memorywindowlength-1)*steps_per_symbol + (0:(params.Nres-1))*(steps_per_symbol/params.Nres)*params.memorywindowlength); if all(indices > 0 & indices <= length(reservoir_signal)), X_test(:,i) = reservoir_signal(indices); end, end
    X_test_bias = [X_test; ones(1, params.num_test_points)];
    y_hat = X_test_bias' * W_out;
    nrmse = sqrt(mean((y_hat - y_test).^2)) / std(y_test);
end