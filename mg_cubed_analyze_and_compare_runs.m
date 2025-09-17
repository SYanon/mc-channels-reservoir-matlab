%% ========================================================================
%  ANALYSIS SCRIPT: MG CUBED - DETERMINISTIC VS. AVERAGED STOCHASTIC
% ========================================================================
%
% This script analyzes the results of the Mackey-Glass Cubed task using the
% optimal hyperparameters found in Iteration 91 of the Bayesian search.
%
% It will:
%   1. Automatically find the 3 most recent Smoldyn run folders.
%   2. Load and average the results from the stochastic simulations.
%   3. Run a deterministic simulation using the same parameters.
%   4. Train and test both models to get their NRMSE scores.
%   5. Generate a single plot comparing the ground truth, the deterministic
%      prediction, and the averaged stochastic prediction.
%
% =========================================================================

%% 0. Initialize Environment & Auto-Detect Data Folders
clear; close all; clc;
fprintf('Starting analysis of MG CUBED: Deterministic vs. Averaged Stochastic...\n');

% Prompt user to select the parent directory containing the result folders
parent_folder = uigetdir('', 'Select the Main Project Folder Containing the Smoldyn Runs');
if parent_folder == 0, error('Analysis cancelled. No folder selected.'); end

fprintf('Searching for MG Cubed results folders in: %s\n', parent_folder);
% Update the search pattern to match your MG Cubed output folders
search_pattern = fullfile(parent_folder, 'MG_Cubed_Smoldyn_Iter91_*');
all_dirs = dir(search_pattern);
all_dirs = all_dirs([all_dirs.isdir]); % Filter for directories only

% Sort directories by date to get the most recent ones
[~, idx] = sort([all_dirs.datenum], 'descend');
sorted_dirs = all_dirs(idx);

if numel(sorted_dirs) < 3
    warning('Could not find at least 3 folders matching the pattern "%s". Using %d folder(s).', 'MG_Cubed_Smoldyn_Iter91_*', numel(sorted_dirs));
    if isempty(sorted_dirs), error('No data folders found.'); end
end
num_folders_to_process = min(3, numel(sorted_dirs));
data_folders = arrayfun(@(x) fullfile(x.folder, x.name), sorted_dirs(1:num_folders_to_process), 'UniformOutput', false);

fprintf('Found %d recent data folder(s) to analyze.\n', num_folders_to_process);

%% 1. Parameter Setup (from Iteration 91 for MG Cubed Task)
fprintf('Using hyperparameters from optimized Iteration 91 for MG Cubed task.\n');
N = 500; k_on = 8.0858e-18; k_off = 3.5861; T = 1.9251;
washout1 = 500; num_train_points = 500; wheretostarttest = 1200; num_test_points = 500;
washout2 = wheretostarttest - (washout1 + num_train_points);
num_tot_points = washout1 + washout2 + num_train_points + num_test_points;
distance = 8.0929e-06; D = 2.9381e-11;
N_min = 100; N_max = 11345; predictlength = 10; % predictlength is 10 for this task
lambda = 1e-6; dt = 0.001; memlengthsweep = 100;

% Reservoir state creation parameters from the optimization run
Nres_base = 50; 
memorywindowlength = 5; % The crucial value from Iteration 91
Nres = Nres_base * memorywindowlength; % Effective Nres is 250
fprintf('Effective reservoir size (Nres) set to %d.\n', Nres);

%% 2. Load and Prepare Mackey-Glass CUBED Ground Truth Data
MG_CUBED_FILENAME = 'MGCubed_series_k10.mat';
try
    loaded_data = load(MG_CUBED_FILENAME, 'input_series', 'target_series');
    fprintf('Loaded ground truth MG CUBED data from "%s".\n', MG_CUBED_FILENAME);
catch ME
    fprintf('ERROR: Could not load ground truth data. Make sure "%s" is in your path.\n', MG_CUBED_FILENAME);
    rethrow(ME);
end

input_series = loaded_data.input_series(1:num_tot_points);
target_series = loaded_data.target_series(1:num_tot_points);

y_train = target_series(washout1+1 : washout1 + num_train_points);
y_test  = target_series(washout1 + num_train_points + washout2 + 1 : end);
y_train = y_train(:);
y_test = y_test(:);

%% 3. Load and Average Stochastic (Smoldyn) Data
fprintf('Loading and averaging data from %d Smoldyn run(s)...\n', num_folders_to_process);
all_runs_data = [];
% Smoldyn's default time step is 0.01, taken from the Smoldyn run script
TIME_STEP_SMOL = 0.01; 
% Create a common time vector for interpolation
STOP_TIME = length(input_series) * T;
time_index_smol = 0:TIME_STEP_SMOL:STOP_TIME;

for i = 1:num_folders_to_process
    % --- MODIFICATION START ---
    % Search for the file using a wildcard to match the actual output name
    file_search_pattern = fullfile(data_folders{i}, 'allmolecules_*.txt');
    file_list = dir(file_search_pattern);
    
    if isempty(file_list)
        warning('Cannot find "allmolecules_*.txt" file in folder: %s. Skipping.', data_folders{i});
        continue;
    end
    
    % Use the first file found that matches the pattern
    filename = fullfile(file_list(1).folder, file_list(1).name);
    % --- MODIFICATION END ---

    try
        data_table = importdata(filename, ' ', 1);
        time_original = data_table.data(:, 1);
        % The last column is ReceptorActive in the allmolecules file format
        receptor_active = data_table.data(:, end); 
        % Interpolate to common time base to ensure runs are comparable
        receptor_active_interp = interp1(time_original, receptor_active, time_index_smol, 'linear', 'extrap');
        all_runs_data(:, end+1) = receptor_active_interp;
        fprintf('  - Successfully loaded and processed data from: %s\n', filename);
    catch ME
        warning('Could not load or process file: %s\n', filename); disp(ME.message);
    end
end

if isempty(all_runs_data), error('Failed to load any valid Smoldyn data. Check folders and file names.'); end
Active_rec_avg = mean(all_runs_data, 2) / N; % Average runs and normalize by N
fprintf('Averaging complete across %d successful run(s).\n', size(all_runs_data, 2));

%% 4. Run Deterministic Model Simulation
fprintf('Running deterministic model simulation for comparison...\n');
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
    sample_indices_float = start_idx - (memorywindowlength-1)*steps_per_symbol_det + (0:(Nres-1))*(steps_per_symbol_det/Nres)*memorywindowlength;
    indices = round(sample_indices_float);
    if all(indices > 0 & indices <= length(n_values))
        X_train_det(:,i) = n_values(indices);
    end
end
X_train_det_bias = [X_train_det; ones(1, num_train_points)];
W_out_det = (X_train_det_bias * X_train_det_bias' + lambda * eye(Nres+1)) \ (X_train_det_bias * y_train);

X_test_det = zeros(Nres, num_test_points);
for i = 1:num_test_points
    t_start_symbol = (wheretostarttest + i - 1) * T;
    start_idx = round(t_start_symbol/dt) + 1;
    sample_indices_float = start_idx - (memorywindowlength-1)*steps_per_symbol_det + (0:(Nres-1))*(steps_per_symbol_det/Nres)*memorywindowlength;
    indices = round(sample_indices_float);
    if all(indices > 0 & indices <= length(n_values))
         X_test_det(:,i) = n_values(indices);
    end
end
X_test_det_bias = [X_test_det; ones(1, num_test_points)];
y_hat_det_test = X_test_det_bias' * W_out_det;
nrmse_det = sqrt(mean((y_hat_det_test - y_test).^2)) / std(y_test);
fprintf('  -> Deterministic Model NRMSE (Test): %.4f\n', nrmse_det);

% --- 5B: Averaged Stochastic Model ---
fprintf('Training and testing averaged stochastic model...\n');
steps_per_symbol_stoch = T/TIME_STEP_SMOL;
X_train_stoch = zeros(Nres, num_train_points);
for i = 1:num_train_points
    t_start_symbol = (washout1 + i - 1) * T;
    start_idx = round(t_start_symbol/TIME_STEP_SMOL) + 1;
    sample_indices_float = start_idx - (memorywindowlength-1)*steps_per_symbol_stoch + (0:(Nres-1))*(steps_per_symbol_stoch/Nres)*memorywindowlength;
    indices = round(sample_indices_float);
    if all(indices > 0 & indices <= length(Active_rec_avg))
        X_train_stoch(:,i) = Active_rec_avg(indices);
    end
end
X_train_stoch_bias = [X_train_stoch; ones(1, num_train_points)];
W_out_stoch = (X_train_stoch_bias * X_train_stoch_bias' + lambda * eye(Nres+1)) \ (X_train_stoch_bias * y_train);

X_test_stoch = zeros(Nres, num_test_points);
for i = 1:num_test_points
    t_start_symbol = (wheretostarttest + i - 1) * T;
    start_idx = round(t_start_symbol/TIME_STEP_SMOL) + 1;
    sample_indices_float = start_idx - (memorywindowlength-1)*steps_per_symbol_stoch + (0:(Nres-1))*(steps_per_symbol_stoch/Nres)*memorywindowlength;
    indices = round(sample_indices_float);
    if all(indices > 0 & indices <= length(Active_rec_avg))
         X_test_stoch(:,i) = Active_rec_avg(indices);
    end
end
X_test_stoch_bias = [X_test_stoch; ones(1, num_test_points)];
y_hat_stoch_test = X_test_stoch_bias' * W_out_stoch;
nrmse_stoch = sqrt(mean((y_hat_stoch_test - y_test).^2)) / std(y_test);
fprintf('  -> Averaged Stochastic Model NRMSE (Test): %.4f\n', nrmse_stoch);

%% 6. Visualize Final Comparison
fprintf('Generating final comparison plot...\n');
figure('Name', 'MG Cubed: Model Comparison on Test Data', 'Position', [100, 100, 1200, 600]);
plot(y_test, 'k', 'LineWidth', 2.5, 'DisplayName', 'Ground Truth');
hold on;
plot(y_hat_det_test, 'b', 'LineWidth', 1.5, 'DisplayName', ...
    sprintf('Deterministic Prediction (NRMSE = %.4f)', nrmse_det));
plot(y_hat_stoch_test, 'r-', 'LineWidth', 1.5, 'DisplayName', ...
    sprintf('Averaged Stochastic Prediction (NRMSE = %.4f)', nrmse_stoch));
hold off;
title('Comparison of Model Predictions on Mackey-Glass Cubed Test Set (Iter 91)');
xlabel('Time Step in Test Set');
ylabel('Normalized Value');
legend('show', 'Location', 'northwest');
grid on;
axis tight;
fprintf('Analysis complete.\n');