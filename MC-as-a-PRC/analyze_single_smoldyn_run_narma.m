% =========================================================================
% analyze_single_smoldyn_run_narma.m
%
% DESCRIPTION:
%   Analyzes the output from a SINGLE high-fidelity Smoldyn run for the
%   NARMA10 task. It generates the target data, loads the Smoldyn data,
%   trains and tests a readout, and generates a plot comparing the
%   predicted vs. the true time series.
%
% =========================================================================

%% 0. Initialize Environment
close all;
clear;
fprintf('Starting single Smoldyn run analysis for NARMA10...\n');

%% 1. CONFIGURATION: EDIT THIS SECTION
% =========================================================================
% Copy and paste the champion hyperparameters for the run you want to analyze.
% The script is pre-filled with the 'Baseline (none)' champion values.

% --- A. Describe the run ---
run_descriptor = 'Optimized NARMA Uniform Delay:8 Post Proc'; 

% --- B. Set the path to your Smoldyn output folder ---
smoldyn_data_dir = 'NARMA_Optimized_Uniform'; % <-- IMPORTANT: CHANGE THIS to your exact folder name

% --- C. Set the hyperparameters for this run ---
run_params.k_on = 1.9961e-17;
run_params.k_off = 9.998;
run_params.T = 1.9959;
run_params.distance = 1.0006e-06;
run_params.memorywindowlength = 5;
run_params.N_max = 19985;
run_params.D = 1.9997e-10;

% --- D. Set other fixed parameters ---
run_params.N = 500;
run_params.N_min = 100;
run_params.lambda = 1e-10; % Use the standard lambda for NARMA
run_params.Nres_init = 100;
run_params.washout1 = 300;
run_params.num_train_points = 2000;
run_params.wheretostarttest = 2500;
run_params.num_test_points  = 2000;
run_params.washout2 = run_params.wheretostarttest - (run_params.washout1 + run_params.num_train_points);
run_params.num_tot_points = run_params.washout1 + run_params.washout2 + run_params.num_train_points + run_params.num_test_points;
run_params.smol_dt = 0.01;
run_params.Nres = run_params.Nres_init * run_params.memorywindowlength;

% =========================================================================

%% 2. Generate and Prepare Data
% Generate the NARMA10 time series to get the target values
narma_len = run_params.num_tot_points + 10 + 1;
u = 0.5 * rand(narma_len, 1);
q = zeros(narma_len, 1);
for n = 10:(run_params.num_tot_points - 1)
    q(n+1) = 0.3*q(n) + 0.05*q(n)*sum(q(n-9:n)) + 1.5*u(n-9)*u(n) + 0.1;
end

% Create the training and testing target splits
train_indices = (run_params.washout1+2) : (run_params.washout1+run_params.num_train_points+1);
test_indices = (run_params.washout1+run_params.num_train_points+run_params.washout2+2) : (run_params.num_tot_points+1);
narma_data.train_target = q(train_indices);
narma_data.test_target = q(test_indices);
narma_data.train_target = narma_data.train_target(:);
narma_data.test_target = narma_data.test_target(:);
fprintf('NARMA10 target data generated.\n');

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
valid_cols_train = any(train_states, 1);
X_train = [train_states(:, valid_cols_train); ones(1, sum(valid_cols_train))];
y_train = narma_data.train_target(valid_cols_train);
W_out = pinv(X_train * X_train' + run_params.lambda * eye(size(X_train, 1))) * (X_train * y_train);
fprintf('Readout trained.\n');

% Test on the testing portion
test_indices = run_params.wheretostarttest + 1 : run_params.wheretostarttest + run_params.num_test_points;
test_states = reservoir_states(:, test_indices);
valid_cols_test = any(test_states, 1);
X_test = [test_states(:, valid_cols_test); ones(1, sum(valid_cols_test))];
y_test_hat = X_test' * W_out;
y_test_true = narma_data.test_target(valid_cols_test);
nrmse_test = sqrt(mean((y_test_hat - y_test_true).^2)) / std(y_test_true);
fprintf('Testing complete. Final NRMSE: %.4f\n', nrmse_test);

%% 5. Generate the Final Plot
fprintf('Generating plot...\n');
figure('Name', ['Stochastic Analysis: ' run_descriptor]);

plot(y_test_true, 'b', 'LineWidth', 2, 'DisplayName', 'Original Target');
hold on;
plot(y_test_hat, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Stochastic Prediction');
hold off;
grid on;
legend('Location', 'best');
xlabel('Sample Index in Test Set');
ylabel('Value');

% Create a detailed title with all the important info
title_str = sprintf('NARMA10 Stochastic Prediction: %s\nFinal NRMSE: %.4f', ...
                    run_descriptor, nrmse_test);
subtitle_str = sprintf('k_{on}=%.2e, k_{off}=%.2f, T=%.2f, dist=%.2fµm, mem=%d, N_{max}=%d, D=%.2e', ...
                       run_params.k_on, run_params.k_off, run_params.T, run_params.distance*1e6, ...
                       run_params.memorywindowlength, run_params.N_max, run_params.D);
title(title_str, 'FontWeight', 'bold');
subtitle(subtitle_str, 'FontSize', 8);

fprintf('All done.\n');