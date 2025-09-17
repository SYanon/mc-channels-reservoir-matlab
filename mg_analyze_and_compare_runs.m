%% ========================================================================
%  ANALYSIS SCRIPT: COMPARE DETERMINISTIC VS. AVERAGED STOCHASTIC RUNS
%  -- SAMPLING LOGIC FIX VERSION 4 --
% ========================================================================
% FILENAME:
%   mg_analyze_and_compare_runs_auto_v4.m
% =========================================================================

%% 0. Initialize Environment & Auto-Detect Data Folders
% ... (This section is unchanged) ...
clear; close all; clc;
fprintf('Starting analysis of deterministic vs. averaged stochastic models...\n');
parent_folder = uigetdir('', 'Select the Main Project Folder (e.g., mc-channels-reservoir-matlab)');
if parent_folder == 0, error('Analysis cancelled. No folder selected.'); end
fprintf('Searching for results folders in: %s\n', parent_folder);
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

%% 1. Parameter Setup (from Iteration 74)
% ... (This section is unchanged but now correctly applied) ...
fprintf('Using hyperparameters from optimized Iteration 74.\n');
N = 500; k_on = 3.9338e-18; k_off = 8.6418; T = 1.9288;
washout1 = 500; num_train_points = 500; wheretostarttest = 1200; num_test_points = 500;
washout2 = wheretostarttest - (washout1 + num_train_points);
num_tot_points = washout1 + washout2 + num_train_points + num_test_points;
distance = 6.4287e-06; D = 3.9627e-11;
N_min = 100; N_max = 16838; predictlength = 6; 
lambda = 1e-6; dt = 0.001; memlengthsweep = 100;
% Correctly set the base Nres and memorywindowlength from the optimization run
Nres_base = 50; 
memorywindowlength = 5; % The crucial value from Iteration 74
Nres = Nres_base * memorywindowlength; % Effective Nres is 250
fprintf('Effective reservoir size (Nres) set to %d.\n', Nres);

%% 2. Load and Prepare Mackey-Glass Ground Truth Data
% ... (This section is unchanged) ...
MG_filename = 'MGseries_RK4_tau17_beta0.20_gamma0.1_n10_len5000_dt1.0.mat';
load(MG_filename, 'mackey_glass_series');
fprintf('Loaded ground truth Mackey-Glass data.\n');
series_segment = mackey_glass_series(1 : num_tot_points + predictlength);
series_normalized = (series_segment - min(series_segment)) / (max(series_segment) - min(series_segment));
input_series  = series_normalized(1:end - predictlength);
target_series = series_normalized((predictlength+1):end);
y_train = target_series(washout1+1 : washout1 + num_train_points);
y_test  = target_series(washout1 + num_train_points + washout2 + 1 : end);
y_train = y_train(:);
y_test = y_test(:);

%% 3. Load and Average Stochastic (Smoldyn) Data
% ... (This section is unchanged) ...
fprintf('Loading and averaging data from %d Smoldyn runs...\n', numel(data_folders));
all_runs_data = []; START_TIME = 0; STOP_TIME = START_TIME + length(input_series) * T;
TIME_STEP_SMOL = 0.01; time_index_smol = START_TIME:TIME_STEP_SMOL:STOP_TIME;
for i = 1:numel(data_folders)
    file_search_pattern = fullfile(data_folders{i}, 'allmolecules_*.txt');
    file_list = dir(file_search_pattern);
    if isempty(file_list), warning('No ''allmolecules_*.txt'' file found in folder: %s. Skipping.', data_folders{i}); continue; end
    if numel(file_list) > 1, warning('Multiple ''allmolecules'' files found in %s. Using the first one: %s', data_folders{i}, file_list(1).name); end
    filename = fullfile(file_list(1).folder, file_list(1).name);
    try
        data_table = importdata(filename, ' ', 1); receptor_active = data_table.data(:, end);
        time_original = data_table.data(:, 1);
        receptor_active_interp = interp1(time_original, receptor_active, time_index_smol, 'linear', 'extrap');
        all_runs_data(:, end+1) = receptor_active_interp;
        fprintf('  - Successfully loaded data from: %s\n', filename);
    catch ME, warning('Could not load or process file: %s\n', filename); disp(ME.message); end
end
if size(all_runs_data, 2) < numel(data_folders), warning('Failed to load data from one or more folders.'); end
if isempty(all_runs_data), error('Could not load any data. Please check folder contents.'); end
Active_rec_avg = mean(all_runs_data, 2) / N;
fprintf('Averaging complete across %d successful run(s).\n', size(all_runs_data, 2));

%% 4. Run Deterministic Model Simulation
% ... (This section is unchanged) ...
fprintf('Running deterministic model simulation...\n');
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
for idxT = 2:length(t_values)
    c_t = c_values(idxT - 1); n_t = n_values(idxT - 1);
    dn_dt = k_on*(N - n_t*N)*c_t - k_off*n_t*N;
    n_values(idxT) = n_values(idxT-1) + (dn_dt/N)*dt;
end
fprintf('Deterministic simulation complete.\n');

%% 5. Process Models and Calculate NRMSE

% --- 5A: Deterministic Model ---
fprintf('Training and testing deterministic model...\n');
steps_per_symbol_det = T/dt;
X_train_det = zeros(Nres, num_train_points);
for i = 1:num_train_points
    t_start_symbol = (washout1 + i - 1) * T;
    start_idx = round(t_start_symbol/dt) + 1;
    % =================== MODIFICATION START ===================
    % USE THE CORRECT SAMPLING FORMULA FROM THE OPTIMIZATION SCRIPT
    sample_indices_float = start_idx - (memorywindowlength-1)*steps_per_symbol_det + (0:(Nres-1))*(steps_per_symbol_det/Nres)*memorywindowlength;
    indices = round(sample_indices_float);
    % ==================== MODIFICATION END ====================
    if all(indices > 0) % Check if indices are valid
        X_train_det(:,i) = n_values(indices);
    end
end
X_train_det = [X_train_det; ones(1, num_train_points)];
W_out_det = (X_train_det * X_train_det' + lambda * eye(Nres+1)) \ (X_train_det * y_train);

X_test_det = zeros(Nres, num_test_points);
for i = 1:num_test_points
    t_start_symbol = (wheretostarttest + i - 1) * T;
    start_idx = round(t_start_symbol/dt) + 1;
    % =================== MODIFICATION START ===================
    sample_indices_float = start_idx - (memorywindowlength-1)*steps_per_symbol_det + (0:(Nres-1))*(steps_per_symbol_det/Nres)*memorywindowlength;
    indices = round(sample_indices_float);
    % ==================== MODIFICATION END ====================
    if all(indices > 0)
         X_test_det(:,i) = n_values(indices);
    end
end
X_test_det = [X_test_det; ones(1, num_test_points)];
y_hat_det_test = X_test_det' * W_out_det;
nrmse_det = sqrt(mean((y_hat_det_test - y_test).^2)) / std(y_test);
fprintf('  -> Deterministic Model NRMSE (Test): %.4f\n', nrmse_det);

% --- 5B: Averaged Stochastic Model ---
fprintf('Training and testing averaged stochastic model...\n');
steps_per_symbol_stoch = T/TIME_STEP_SMOL;
X_train_stoch = zeros(Nres, num_train_points);
for i = 1:num_train_points
    t_start_symbol = (washout1 + i - 1) * T;
    start_idx = round(t_start_symbol/TIME_STEP_SMOL) + 1;
    % =================== MODIFICATION START ===================
    sample_indices_float = start_idx - (memorywindowlength-1)*steps_per_symbol_stoch + (0:(Nres-1))*(steps_per_symbol_stoch/Nres)*memorywindowlength;
    indices = round(sample_indices_float);
    % ==================== MODIFICATION END ====================
    if all(indices > 0 & indices <= length(Active_rec_avg))
        X_train_stoch(:,i) = Active_rec_avg(indices);
    end
end
X_train_stoch = [X_train_stoch; ones(1, num_train_points)];
W_out_stoch = (X_train_stoch * X_train_stoch' + lambda * eye(Nres+1)) \ (X_train_stoch * y_train);

X_test_stoch = zeros(Nres, num_test_points);
for i = 1:num_test_points
    t_start_symbol = (wheretostarttest + i - 1) * T;
    start_idx = round(t_start_symbol/TIME_STEP_SMOL) + 1;
    % =================== MODIFICATION START ===================
    sample_indices_float = start_idx - (memorywindowlength-1)*steps_per_symbol_stoch + (0:(Nres-1))*(steps_per_symbol_stoch/Nres)*memorywindowlength;
    indices = round(sample_indices_float);
    % ==================== MODIFICATION END ====================
    if all(indices > 0 & indices <= length(Active_rec_avg))
         X_test_stoch(:,i) = Active_rec_avg(indices);
    end
end
X_test_stoch = [X_test_stoch; ones(1, num_test_points)];
y_hat_stoch_test = X_test_stoch' * W_out_stoch;
nrmse_stoch = sqrt(mean((y_hat_stoch_test - y_test).^2)) / std(y_test);
fprintf('  -> Averaged Stochastic Model NRMSE (Test): %.4f\n', nrmse_stoch);

%% 6. Visualize Final Comparison
% ... (This section is unchanged) ...
fprintf('Generating final comparison plot...\n');
figure('Name', 'Model Comparison on Test Data', 'Position', [100, 100, 1200, 600]);
plot(y_test, 'k', 'LineWidth', 2.5, 'DisplayName', 'Ground Truth');
hold on;
plot(y_hat_det_test, 'b', 'LineWidth', 1.5, 'DisplayName', ...
    sprintf('Deterministic Prediction (NRMSE = %.4f)', nrmse_det));
plot(y_hat_stoch_test, 'r-', 'LineWidth', 1.5, 'DisplayName', ...
    sprintf('Averaged Stochastic Prediction (NRMSE = %.4f)', nrmse_stoch));
hold off;
title('Comparison of Model Predictions on Mackey-Glass Test Set');
xlabel('Time Step in Test Set');
ylabel('Normalized Value');
legend('show', 'Location', 'northwest');
grid on;
axis tight;
fprintf('Analysis complete.\n');