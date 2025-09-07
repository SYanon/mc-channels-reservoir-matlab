% =========================================================================
% optimize_lorenz_hyperparams_bayesopt.m
%
% DESCRIPTION:
%   Uses Bayesian Optimization to find optimal hyperparameters for the
%   Lorenz '63 PRC prediction task.
% =========================================================================

%% 0. Initialize Environment
close all;
clear;
rng('default'); % for reproducibility
fprintf('Starting Bayesian Optimization for Lorenz ''63 Hyperparameters...\n');

%% 1. Load Data and Default Config (Done ONCE)
config = get_default_config();
lorenz_data = load_lorenz_data(config.LORENZ_FILENAME, config.num_tot_points, config.predictlength);
fprintf('Lorenz ''63 data loaded.\n');

%% 2. Define Hyperparameters for Optimization(update for SMoldyn workings)
vars = [
    optimizableVariable('k_on', [5e-20, 2e-17], 'Transform', 'log');
    optimizableVariable('k_off', [0.1, 10]);
    optimizableVariable('T', [0.2, 1.5]); % Sub-symbol duration
    optimizableVariable('distance', [2e-6, 50e-6], 'Transform', 'log');
    optimizableVariable('N_max', [500, 20000], 'Transform', 'log');
    optimizableVariable('D', [2e-11, 2e-10], 'Transform', 'log');
];

%% 3. Create Objective Function Handle
objFun = @(params) bayesopt_objective_function(params, lorenz_data, config);

%% 4. Run Bayesian Optimization
fprintf('Running Bayesian Optimization for 100 trials...\n');
results = bayesopt(objFun, vars, ...
    'MaxObjectiveEvaluations', 100, ...
    'IsObjectiveDeterministic', true, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'Verbose', 1);

fprintf('\n===== Optimization Complete =====\n');

%% 5. Display Best Results and Run a Final Test
bestParams = results.XAtMinObjective;
minNRMSE = results.MinObjective;

fprintf('\n--- Optimal Hyperparameters Found ---\n');
disp(bestParams);
fprintf('Minimum Validation NRMSE: %.5f\n', minNRMSE);

% Run one final simulation with the best parameters to plot the result
fprintf('\nRunning final simulation with optimal parameters to generate plot...\n');
best_config = config;
param_names = bestParams.Properties.VariableNames;
for i = 1:length(param_names)
    best_config.(param_names{i}) = bestParams.(param_names{i});
end

[~, reservoir_states] = run_channel_simulation(lorenz_data.input_series, best_config);
W_out = train_readout(reservoir_states, lorenz_data.train_target, best_config);
[~, nrmse_test, y_test_hat] = test_readout(W_out, reservoir_states, lorenz_data, best_config);

% Plot the final, best-case result
figure('Name', 'Optimal Lorenz Attractor Prediction');
plot3(lorenz_data.test_target(:,1), lorenz_data.test_target(:,2), lorenz_data.test_target(:,3), 'b-', 'LineWidth', 1.5, 'DisplayName', 'True Attractor');
hold on;
plot3(y_test_hat(:,1), y_test_hat(:,2), y_test_hat(:,3), 'g-', 'LineWidth', 1.5, 'DisplayName', 'Predicted Attractor');
title(sprintf('Optimal Result | Test NRMSE = %.4f', nrmse_test));
xlabel('x'); ylabel('y'); zlabel('z');
legend; grid on; axis tight; view(35, 25);
fprintf('Final plot generated. Best test NRMSE: %.4f\n', nrmse_test);


%==========================================================================
%                       HELPER FUNCTIONS
%==========================================================================

function nrmse_test = bayesopt_objective_function(params, lorenz_data, config)
    run_config = config;
    param_names = params.Properties.VariableNames;
    for i = 1:length(param_names)
        run_config.(param_names{i}) = params.(param_names{i});
    end
    
    try
        [~, reservoir_states] = run_channel_simulation(lorenz_data.input_series, run_config);
        W_out = train_readout(reservoir_states, lorenz_data.train_target, run_config);
        [~, nrmse_test] = test_readout(W_out, reservoir_states, lorenz_data, run_config);
        
        % Handle potential NaNs or Infs from unstable simulations
        if ~isfinite(nrmse_test)
            nrmse_test = 100; % Assign a large penalty
        end
    catch ME
        fprintf('Error during simulation: %s. Assigning high penalty.\n', ME.message);
        nrmse_test = 100; % Assign a large penalty if simulation fails
    end
end

function config = get_default_config()
    % Default parameters are copied from the sweep script for consistency
    config.N     = 500;
    config.N_min = 100;
    config.lambda = 1e-7;
    config.Nres   = 150;
    config.predictlength = 1;
    config.washout1         = 500;
    config.num_train_points = 2000;
    config.wheretostarttest = 3000;
    config.num_test_points  = 1500;
    config.washout2 = config.wheretostarttest - (config.washout1 + config.num_train_points);
    config.num_tot_points = config.washout1 + config.washout2 + config.num_train_points + config.num_test_points;
    config.dt = 0.001;
    config.memlengthsweep = 100;
    config.LORENZ_FILENAME = 'lorenz_series_len8000.mat';
end

function lorenz_data = load_lorenz_data(filename, required_len, predict_len)
    loaded = load(filename, 'lorenz_series');
    full_series = loaded.lorenz_series;
    if size(full_series, 1) < required_len + predict_len, error('Loaded Lorenz series is too short.'); end
    series_segment = full_series(1 : required_len + predict_len, :);
    lorenz_data.input_series  = series_segment(1:end - predict_len, :);
    lorenz_data.target_series = series_segment(predict_len + 1 : end, :);
    config = get_default_config();
    tr_end = config.washout1 + config.num_train_points;
    te_start = config.wheretostarttest + 1;
    te_end = te_start + config.num_test_points - 1;
    lorenz_data.train_target = lorenz_data.target_series(config.washout1+1 : tr_end, :);
    lorenz_data.test_target  = lorenz_data.target_series(te_start : te_end, :);
end

function [sim, reservoir_states] = run_channel_simulation(input_matrix, config)
    [num_points, num_dims] = size(input_matrix);
    serialized_input = reshape(input_matrix', 1, []);
    sub_symbol_T = config.T;
    t_total = num_points * num_dims * sub_symbol_T;
    N_i = config.N_min + (serialized_input - 0)/(1 - 0) * (config.N_max - config.N_min);
    sim.time = 0:config.dt:t_total;
    sim.concentration = zeros(size(sim.time));
    Tpeak = config.distance^2/(6*config.D);
    memory_length = round((config.memlengthsweep * Tpeak)/sub_symbol_T)*sub_symbol_T;
    for iSym = 1:length(N_i)
        t_symbol_start = (iSym - 1) * sub_symbol_T;
        t_memory_end   = t_symbol_start + memory_length;
        idx_start = round(t_symbol_start / config.dt) + 2;
        idx_end = round(t_memory_end / config.dt) + 1;
        idxRange = idx_start:min(idx_end, length(sim.time));
        if isempty(idxRange), continue; end
        t_local = sim.time(idxRange) - t_symbol_start;
        % Add a small epsilon to prevent division by zero
        epsilon = 1e-6; 
        pulse = (N_i(iSym) ./ ((4*pi*config.D*t_local + epsilon).^(3/2))) .* exp(-config.distance^2./(4*config.D*t_local + epsilon));
        sim.concentration(idxRange) = sim.concentration(idxRange) + pulse;
    end
    sim.occupation = zeros(size(sim.time));
    for idxT = 2:length(sim.time)
        c_t = sim.concentration(idxT - 1);
        n_t = sim.occupation(idxT - 1);
        dn_dt = config.k_on*(config.N - n_t*config.N)*c_t - config.k_off*n_t*config.N;
        sim.occupation(idxT) = sim.occupation(idxT-1) + (dn_dt/config.N)*config.dt;
    end
    reservoir_states = zeros(config.Nres, num_points);
    steps_per_full_symbol = (num_dims * sub_symbol_T) / config.dt;
    for iPoint = 1:num_points
        t_symbol_start = (iPoint - 1) * num_dims * sub_symbol_T;
        start_idx = round(t_symbol_start/config.dt) + 1;
        sample_indices_float = start_idx + (0:(config.Nres-1)) * (steps_per_full_symbol / config.Nres);
        sample_indices = round(sample_indices_float);
        if all(sample_indices > 0 & sample_indices <= length(sim.occupation))
            reservoir_states(:, iPoint) = sim.occupation(sample_indices);
        else
            reservoir_states(:, iPoint) = 0; % Handle edge cases
        end
    end
end

function W_out = train_readout(reservoir_states, train_target, config)
    train_indices = config.washout1+1 : config.washout1+config.num_train_points;
    train_states = reservoir_states(:, train_indices);
    X_train = [train_states; ones(1, size(train_states, 2))];
    Y_train = train_target;
    W_out = pinv(X_train * X_train' + config.lambda * eye(size(X_train, 1))) * (X_train * Y_train);
end

function [nrmse_train, nrmse_test, y_test_hat] = test_readout(W_out, reservoir_states, lorenz_data, config)
    % MODIFIED: Added y_test_hat to the function output list
    
    % --- Test on Training Data ---
    train_indices = config.washout1+1 : config.washout1+config.num_train_points;
    train_states = reservoir_states(:, train_indices);
    X_train = [train_states; ones(1, size(train_states, 2))];
    y_train_hat = X_train' * W_out;
    train_error = y_train_hat - lorenz_data.train_target;
    nrmse_train = sqrt(mean(train_error(:).^2)) / std(lorenz_data.train_target(:));
    
    % --- Test on Testing Data ---
    test_indices = config.wheretostarttest+1 : config.wheretostarttest+config.num_test_points;
    test_states = reservoir_states(:, test_indices);
    X_test = [test_states; ones(1, size(test_states, 2))];
    y_test_hat = X_test' * W_out; % This is the variable we need to output
    test_error = y_test_hat - lorenz_data.test_target;
    nrmse_test = sqrt(mean(test_error(:).^2)) / std(lorenz_data.test_target(:));
end