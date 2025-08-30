% =========================================================================
% analyze_single_smoldyn_run_mg.m
%
% DESCRIPTION:
%   Analyzes the output from a SINGLE high-fidelity Smoldyn run for the
%   Mackey-Glass task. It loads the Smoldyn data, trains and tests a
%   readout, calculates the final NRMSE, and generates a plot comparing
%   the predicted vs. the true time series, similar to Figure 3 in the paper.
%
% =========================================================================

%% 0. Initialize Environment
close all;
clear;
fprintf('Starting single Smoldyn run analysis for Mackey-Glass...\n');

%% 1. CONFIGURATION: EDIT THIS SECTION
% =========================================================================
% Copy and paste the champion hyperparameters for the run you want to analyze.
% The script is pre-filled with the 'Multi-Random' champion values.

% --- A. Describe the run ---
run_descriptor = 'Optimized for No post Processing'; 

% --- B. Set the path to your Smoldyn output folder ---
smoldyn_data_dir = 'MG_Optimized_None'; % <-- IMPORTANT: CHANGE THIS to your exact folder name

% --- C. Set the hyperparameters for this run ---
run_params.k_on = 1.8248e-17;
run_params.k_off = 7.6434;
run_params.T = 1.5163;
run_params.distance = 5.0777e-06;
run_params.memorywindowlength = 5;
run_params.N_max = 16135;
run_params.D = 1.7926e-10;

% --- D. Set other fixed parameters ---
run_params.N = 500;
run_params.N_min = 100;
run_params.lambda = 1e-6; % Use the standard lambda for MG
run_params.Nres_init = 50;
run_params.predictlength = 6;
run_params.washout1 = 500;
run_params.num_train_points = 500;
run_params.wheretostarttest = 1200;
run_params.num_test_points = 500;
run_params.washout2 = run_params.wheretostarttest - (run_params.washout1 + run_params.num_train_points);
run_params.num_tot_points = run_params.washout1 + run_params.washout2 + run_params.num_train_points + run_params.num_test_points;
run_params.smol_dt = 0.01;
run_params.Nres = run_params.Nres_init * run_params.memorywindowlength;
run_params.MG_FILENAME = 'MGseries_RK4_tau17_beta0.20_gamma0.1_n10_len5000_dt1.0.mat';

% =========================================================================

%% 2. Load and Prepare Data
% Load the original Mackey-Glass series
try
    loaded_data = load(run_params.MG_FILENAME, 'mackey_glass_series');
    full_series_raw = loaded_data.mackey_glass_series;
    series_segment = full_series_raw(1 : run_params.num_tot_points + run_params.predictlength);
    full_series_norm = (series_segment - min(series_segment)) / (max(series_segment) - min(series_segment));
catch
    error('Failed to load MG data from "%s".', run_params.MG_FILENAME);
end

% Create the input/target splits
mg_data.input_series  = full_series_norm(1:end - run_params.predictlength);
target_series = full_series_norm(run_params.predictlength + 1 : end);
train_end_idx = run_params.washout1 + run_params.num_train_points;
mg_data.train_target = target_series(run_params.washout1 + 1 : train_end_idx);
test_start_idx = run_params.wheretostarttest + 1;
mg_data.test_target = target_series(test_start_idx : test_start_idx + run_params.num_test_points - 1);
mg_data.train_target = mg_data.train_target(:);
mg_data.test_target = mg_data.test_target(:);

%% 3. Load and Sample Stochastic Reservoir States from Smoldyn File
fprintf('Loading Smoldyn data from: %s\n', smoldyn_data_dir);
filename = 'allmolecules_varNo_2_varValue_1_iter_1.txt'; % Default Smoldyn output name
filepath = fullfile(smoldyn_data_dir, filename);
if ~exist(filepath, 'file'), error('Smoldyn output file not found at: %s', filepath); end

data_temp = importdata(filepath, ' ', 1);
smol_data = data_temp.data;
time_vec = smol_data(:, 1);
active_receptors = smol_data(:, end) / run_params.N; % Normalize by N

% Sample the stochastic states
steps_per_symbol = run_params.T / run_params.smol_dt;
num_symbols = run_params.num_tot_points;
reservoir_states = zeros(run_params.Nres, num_symbols);

for iSym = 1:num_symbols
    t_symbol_start = (iSym - 1) * run_params.T;
    start_idx = find(time_vec >= t_symbol_start, 1);
    if isempty(start_idx), continue; end
    sample_indices_float = start_idx - (run_params.memorywindowlength-1)*steps_per_symbol + (0:(run_params.Nres-1))*(steps_per_symbol/run_params.Nres)*run_params.memorywindowlength;
    sample_indices = round(sample_indices_float);
    if all(sample_indices > 0 & sample_indices <= length(active_receptors))
        reservoir_states(:, iSym) = active_receptors(sample_indices);
    end
end
fprintf('Smoldyn data loaded and reservoir states sampled.\n');

%% 4. Train and Test the Readout
% Train on the training portion of the stochastic data
train_states = reservoir_states(:, run_params.washout1+1 : run_params.washout1+run_params.num_train_points);
X_train = [train_states; ones(1, run_params.num_train_points)];
y_train = mg_data.train_target(:);
W_out = pinv(X_train * X_train' + run_params.lambda * eye(size(X_train, 1))) * (X_train * y_train);
fprintf('Readout trained.\n');

% Test on the testing portion
test_indices = run_params.wheretostarttest + 1 : run_params.wheretostarttest + run_params.num_test_points;
test_states = reservoir_states(:, test_indices);
X_test = [test_states; ones(1, run_params.num_test_points)];
y_test_hat = X_test' * W_out;
nrmse_test = sqrt(mean((y_test_hat - mg_data.test_target).^2)) / std(mg_data.test_target);
fprintf('Testing complete. Final NRMSE: %.4f\n', nrmse_test);

%% 5. Generate the Final Plot
fprintf('Generating plot...\n');
figure('Name', ['Stochastic Analysis: ' run_descriptor]);

plot(mg_data.test_target, 'b', 'LineWidth', 2, 'DisplayName', 'Original Target');
hold on;
plot(y_test_hat, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Stochastic Prediction');
hold off;
grid on;
legend('Location', 'best');
xlabel('Sample Index in Test Set');
ylabel('Normalized Value');

% Create a detailed title with all the important info
title_str = sprintf('%s\nFinal NRMSE: %.4f | Prediction Length: %d steps', ...
                    run_descriptor, nrmse_test, run_params.predictlength);
subtitle_str = sprintf('k_{on}=%.2e, k_{off}=%.2f, T=%.2f, dist=%.2fµm, mem=%d, N_{max}=%d, D=%.2e', ...
                       run_params.k_on, run_params.k_off, run_params.T, run_params.distance*1e6, ...
                       run_params.memorywindowlength, run_params.N_max, run_params.D);
title(title_str, 'FontWeight', 'bold');
subtitle(subtitle_str, 'FontSize', 8);

fprintf('All done.\n');