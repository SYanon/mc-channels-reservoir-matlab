%% ========================================================================
%  ANALYSIS SCRIPT: SINE-TO-SQUARE - DETERMINISTIC VS. AVERAGED STOCHASTIC
% ========================================================================
%
% VERSION 2.2 (FINAL CHECK) - Aligned matrix inversion method to pinv()
%
% This script analyzes the results of the sine-to-square transformation task 
% using the optimal hyperparameters found in Iteration 23 of the Bayesian search.
%
% =========================================================================

%% 0. Initialize Environment & Auto-Detect Data Folders
clear; close all; clc;

fprintf('--- EXECUTING SCRIPT VERSION 2.2 (FINAL CHECK) ---\n\n');

fprintf('Starting analysis of SINE-TO-SQUARE: Deterministic vs. Averaged Stochastic...\n');

% Prompt user to select the parent directory containing the result folders
parent_folder = uigetdir('', 'Select the Main Project Folder Containing the Smoldyn Runs');
if parent_folder == 0, error('Analysis cancelled. No folder selected.'); end

fprintf('Searching for Sine-to-Square results folders in: %s\n', parent_folder);
% Update the search pattern to match your Sine-to-Square output folders
search_pattern = fullfile(parent_folder, 'Sine_Smoldyn_Iter23_*');
all_dirs = dir(search_pattern);
all_dirs = all_dirs([all_dirs.isdir]); % Filter for directories only

% Sort directories by date to get the most recent ones
[~, idx] = sort([all_dirs.datenum], 'descend');
sorted_dirs = all_dirs(idx);

if numel(sorted_dirs) < 3
    warning('Could not find at least 3 folders matching the pattern "%s". Using %d folder(s).', 'Sine_Smoldyn_Iter23_*', numel(sorted_dirs));
    if isempty(sorted_dirs), error('No data folders found.'); end
end
num_folders_to_process = min(3, numel(sorted_dirs));
data_folders = arrayfun(@(x) fullfile(x.folder, x.name), sorted_dirs(1:num_folders_to_process), 'UniformOutput', false);

fprintf('Found %d recent data folder(s) to analyze.\n', num_folders_to_process);

%% 1. Parameter Setup (from Iteration 23 for Sine-to-Square Task)
fprintf('Using hyperparameters from optimized Iteration 23 for Sine-to-Square task.\n');
% The parameters below are from Iteration 23, identified as the best
% realistic configuration for the Sine-to-Square task.
N = 500;                % Total number of receptors (fixed default)
k_on = 1.98e-17;        % Binding rate constant (s^-1 M^-1) -- OPTIMIZED
k_off = 6.44;           % Unbinding rate constant (s^-1) -- OPTIMIZED
T = 1.64;               % Symbol duration (seconds) -- OPTIMIZED
washout1         = 500;
num_train_points = 2000;
wheretostarttest = 3000;
num_test_points  = 1000;
washout2 = wheretostarttest - (washout1 + num_train_points);
num_tot_points = washout1 + washout2 + num_train_points + num_test_points;
distance = 4.53e-06;    % Distance (m) -- OPTIMIZED
D = 1.17e-11;         % Diffusion coefficient (m^2/s) -- OPTIMIZED
N_min = 100;            % Minimum value for input normalization (fixed default)
N_max = 9564;           % Maximum value for input normalization -- OPTIMIZED
dt = 0.001;
memlengthsweep = 100;
lambda = 1e-6;          % Ridge regression parameter (a common default)

% Reservoir state creation parameters from the optimization run
Nres_init = 100;         % The base number of nodes from the optimization config
memorywindowlength = 5;  % The crucial value from Iteration 23
Nres = Nres_init * memorywindowlength; % Effective Nres is 500
fprintf('Effective reservoir size (Nres) set to %d.\n', Nres);

%% 2. Load and Prepare Sine-to-Square Ground Truth Data
SINE_FILENAME = 'SINEseries_len5000_p25.mat';
try
    loaded_data = load(SINE_FILENAME, 'input_sine_series', 'target_square_series');
    fprintf('Loaded ground truth Sine-to-Square data from "%s".\n', SINE_FILENAME);
catch ME
    fprintf('ERROR: Could not load ground truth data. Make sure "%s" is in your path.\n', SINE_FILENAME);
    rethrow(ME);
end

input_series = loaded_data.input_sine_series(1:num_tot_points);
target_series = loaded_data.target_square_series(1:num_tot_points);

% Split into training and testing sets for the target (square wave)
y_train = target_series(washout1+1 : washout1 + num_train_points);
y_test  = target_series(wheretostarttest + 1 : wheretostarttest + num_test_points);
y_train = y_train(:); % Ensure column vectors
y_test = y_test(:);

%% 3. Load and Average Stochastic (Smoldyn) Data
fprintf('Loading and averaging data from %d Smoldyn run(s)...\n', num_folders_to_process);
all_runs_data = [];
% Smoldyn's default time step is 0.01, taken from the Smoldyn run script
TIME_STEP_SMOL = 0.01; 
% Create a common time vector for interpolation
STOP_TIME = length(input_series) * T;
time_index_smol = (0:TIME_STEP_SMOL:STOP_TIME)'; % Ensure column vector

for i = 1:num_folders_to_process
    file_search_pattern = fullfile(data_folders{i}, 'allmolecules_*.txt');
    file_list = dir(file_search_pattern);
    if isempty(file_list), warning('Cannot find "allmolecules_*.txt" file in folder: %s. Skipping.', data_folders{i}); continue; end
    filename = fullfile(file_list(1).folder, file_list(1).name);
    try
        data_table = importdata(filename, ' ', 1);
        time_original = data_table.data(:, 1);
        receptor_active = data_table.data(:, 4); 
        receptor_active_interp = interp1(time_original, receptor_active, time_index_smol, 'linear', 'extrap');
        all_runs_data(:, end+1) = receptor_active_interp;
        fprintf('  - Successfully loaded and processed data from: %s\n', file_list(1).name);
    catch ME
        warning('Could not load or process file: %s\n', filename); disp(ME.message);
    end
end

if isempty(all_runs_data), error('Failed to load any valid Smoldyn data. Check folders and file names.'); end
Active_rec_avg = mean(all_runs_data, 2) / N; % Average runs and normalize by N
fprintf('Averaging complete across %d successful run(s).\n', size(all_runs_data, 2));

%% 4. Run Deterministic Model Simulation
fprintf('Running deterministic model simulation for comparison...\n');
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
n_values = zeros(size(t_values)); % This will hold occupation fraction
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

% Filter out all-zero columns from the training states
valid_cols_train = any(X_train_det, 1);
X_train_det_filtered = X_train_det(:, valid_cols_train);
y_train_filtered = y_train(valid_cols_train);
X_train_det_bias = [X_train_det_filtered; ones(1, size(X_train_det_filtered, 2))];

% --- MODIFIED LINE ---
% Changed from '\' to 'pinv' to perfectly match the optimization script's method
W_out_det = pinv(X_train_det_bias * X_train_det_bias' + lambda * eye(Nres+1)) * (X_train_det_bias * y_train_filtered);
% --- END MODIFICATION ---

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

% Filter the test states using the same logic for a consistent NRMSE calculation.
valid_cols_test = any(X_test_det, 1);
X_test_det_filtered = X_test_det(:, valid_cols_test);
y_test_filtered = y_test(valid_cols_test);
X_test_det_bias = [X_test_det_filtered; ones(1, size(X_test_det_filtered, 2))];
y_hat_det_test_filtered = X_test_det_bias' * W_out_det;
y_hat_det_plot = NaN(num_test_points, 1);
y_hat_det_plot(valid_cols_test) = y_hat_det_test_filtered;
nrmse_det = sqrt(mean((y_hat_det_test_filtered - y_test_filtered).^2)) / std(y_test_filtered);

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
figure('Name', 'Sine-to-Square: Model Comparison on Test Data', 'Position', [100, 100, 1200, 600]);
plot(y_test, 'k', 'LineWidth', 2.5, 'DisplayName', 'Ground Truth (Square Wave)');
hold on;
plot(y_hat_det_plot, 'b', 'LineWidth', 1.5, 'DisplayName', ...
    sprintf('Deterministic Prediction (NRMSE = %.4f)', nrmse_det));
plot(y_hat_stoch_test, 'r-', 'LineWidth', 1.5, 'DisplayName', ...
    sprintf('Averaged Stochastic Prediction (NRMSE = %.4f)', nrmse_stoch));
hold off;
title('Comparison of Model Predictions on Sine-to-Square Test Set (Iter 23)');
xlabel('Time Step in Test Set');
ylabel('Normalized Value');
legend('show', 'Location', 'northwest');
grid on;
axis tight;
fprintf('Analysis complete.\n');