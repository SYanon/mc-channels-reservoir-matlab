% =========================================================================
% optimize_wh_hyperparams_bayesopt.m
%
% DESCRIPTION:
%   Uses Bayesian Optimization to efficiently find the optimal combination
%   of key hyperparameters for the Wiener-Hammerstein PRC transformation task.
%   This is a complete, runnable, and updated script.
%
% REQUIRES:
%   Statistics and Machine Learning Toolbox
% =========================================================================

%% 0. Initialize Environment
close all;
clear;
rng('default'); % for reproducibility
fprintf('Starting Bayesian Optimization for Wiener-Hammerstein Hyperparameters...\n');

%% 1. Load Data and Default Config (Done ONCE)
config = get_default_config();
try
    task_data = load_wh_data(config);
    fprintf('Wiener-Hammerstein task data loaded successfully.\n');
catch ME
    fprintf('Error loading data file. Please run createWHseries.m first.\n');
    rethrow(ME);
end

%% 2. Define Hyperparameters for Optimization
% UPDATED: We are now optimizing 5 key parameters to give the model
% more flexibility to find a good working regime for this complex task.
vars = [
    optimizableVariable('k_on', [1.66*1e-18, 1.66*1e-15], 'Transform', 'log');
    optimizableVariable('k_off', [1e-3, 1000]);
    optimizableVariable('T', [0.5, 2]);
    optimizableVariable('distance', [1e-7, 20*1e-6], 'Transform', 'log');
    optimizableVariable('memorywindowlength', [1, 5], 'Type', 'integer');
    optimizableVariable('N_max', [500, 50000]);
    optimizableVariable('D', [1e-11, 5*1e-10], 'Transform', 'log');
];

%% 3. Define Objective Function
% The function that bayesopt will try to minimize.
% It takes a table of hyperparameters and returns the NRMSE.
objFun = @(params) bayesopt_objective_function(params, task_data, config);

%% 4. Run Bayesian Optimization
fprintf('Running Bayesian Optimization for 100 trials...\n');
results = bayesopt(objFun, vars, ...
    'MaxObjectiveEvaluations', 100, ...      % Number of simulations to run
    'IsObjectiveDeterministic', true, ...   % True, since our model is deterministic
    'AcquisitionFunctionName', 'expected-improvement-plus', ... % A good modern default
    'Verbose', 1);                          % Show progress in the command window

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
best_config.k_on = bestParams.k_on;
best_config.k_off = bestParams.k_off;
best_config.D = bestParams.D;
best_config.N_max = round(bestParams.N_max);
best_config.lambda = bestParams.lambda;

reservoir_states = run_reservoir(task_data.input, best_config);
W_out = train_readout(reservoir_states, task_data.target, best_config);
[~, nrmse_test, y_test_hat] = test_readout(W_out, reservoir_states, task_data.target, best_config);

% Plot the final, best-case result
plot_final_result(y_test_hat, task_data, best_config, nrmse_test);
fprintf('Final plot generated. Best test NRMSE: %.4f\n', nrmse_test);


% =========================================================================
%                       HELPER FUNCTIONS
% =========================================================================

function nrmse_test = bayesopt_objective_function(params, task_data, config)
    % This function is called by bayesopt for each trial.
    % It takes the proposed parameters, runs the simulation, and returns the NRMSE.
    
    % 1. Create a config for this specific run
    run_config = config;
    run_config.k_on = params.k_on;
    run_config.k_off = params.k_off;
    run_config.T = params.T;
    run_config.distance = params.distance;
    run_config.memorywindowlength = params.memorywindowlength;
    run_config.N_max = params.N_max;
    run_config.D = params.D;
    
    % Update dependent parameters
    run_config.Nres = run_config.Nres_init * run_config.memorywindowlength;
    
    % 2. Run the full simulation and learning pipeline
    [~, reservoir_states] = run_channel_simulation(task_data.input_series, run_config);
    W_out = train_readout(reservoir_states, task_data.train_target, run_config);
    [~, nrmse_test] = test_readout(W_out, reservoir_states, task_data, run_config);
end


function config = get_default_config()
    
    % Provides fixed (non-optimized) parameters.
    config.N     = 500;
    config.N_min = 100;
    config.lambda = 1e-6;
    config.Nres_init = 100;
    
    % Data Lengths (from sine numerical analysis)
    config.washout1         = 500;
    config.num_train_points = 2000;
    config.wheretostarttest = 3000;
    config.num_test_points  = 1000;
    config.washout2 = config.wheretostarttest - (config.washout1 + config.num_train_points);
    config.num_tot_points = config.washout1 + config.washout2 + config.num_train_points + config.num_test_points;
    
    % Simulation Control
    config.dt = 0.001;
    config.memlengthsweep = 100;
    
    % Central configuration for the simulation
    config.WH_FILENAME = 'WHseries_len5000.mat'; 

end

function task_data = load_wh_data(config)
    % Loads the Wiener-Hammerstein data from the .mat file.
    data = load(config.WH_FILENAME); % needs input_WH_series, target_WH_series
    total_points_needed = config.num_tot_points;

    if length(data.input_WH_series) < total_points_needed || ...
       length(data.target_WH_series) < total_points_needed
        error('Not enough data points in the file for the specified train/test split.');
    end

    full_input  = data.input_WH_series(1:total_points_needed);
    full_target = data.target_WH_series(1:total_points_needed);

    task_data.input_series = full_input;
    task_data.full_target  = full_target;

    % Prepare train/test targets consistent with config indices
    tr_idx_start = config.washout1 + 1;
    tr_idx_end   = config.washout1 + config.num_train_points;

    te_idx_start = config.wheretostarttest + 1;
    te_idx_end   = config.wheretostarttest + config.num_test_points;

    task_data.train_target = full_target(tr_idx_start:tr_idx_end);
    task_data.test_target  = full_target(te_idx_start:te_idx_end);
end

function [sim, reservoir_states] = run_channel_simulation(input_series, config)
    inp_min = 0; inp_max = 1; % Input is already normalized
    N_i = config.N_min + (input_series - inp_min)/(inp_max - inp_min) * (config.N_max - config.N_min);
    t_total = length(N_i) * config.T;
    sim.time = 0:config.dt:t_total;
    sim.concentration = zeros(size(sim.time));
    Tpeak = config.distance^2/(6*config.D);
    memory_length = round((config.memlengthsweep * Tpeak)/config.T)*config.T;
    for iSym = 1:length(N_i)
        t_symbol_start = (iSym - 1) * config.T;
        t_memory_end   = t_symbol_start + memory_length;
        idxRange = find(sim.time > t_symbol_start & sim.time <= t_memory_end);
        if isempty(idxRange), continue; end
        t_local = sim.time(idxRange) - t_symbol_start;
        pulse = (N_i(iSym) ./ ((4*pi*config.D*t_local).^(3/2))) .* exp(-config.distance^2./(4*config.D*t_local));
        sim.concentration(idxRange) = sim.concentration(idxRange) + pulse;
    end
    sim.occupation = zeros(size(sim.time));
    for idxT = 2:length(sim.time)
        c_t = sim.concentration(idxT - 1);
        n_t = sim.occupation(idxT - 1);
        dn_dt = config.k_on*(config.N - n_t*config.N)*c_t - config.k_off*n_t*config.N;
        sim.occupation(idxT) = sim.occupation(idxT-1) + (dn_dt/config.N)*config.dt;
    end
    steps_per_symbol = config.T/config.dt;
    num_symbols = config.num_tot_points;
    reservoir_states = zeros(config.Nres, num_symbols);
    for iSym = 1:num_symbols
        t_symbol_start = (iSym - 1) * config.T;
        start_idx = round(t_symbol_start/config.dt) + 1;
        sample_indices_float = start_idx - (config.memorywindowlength-1)*steps_per_symbol + (0:(config.Nres-1))*(steps_per_symbol/config.Nres)*config.memorywindowlength;
        sample_indices = round(sample_indices_float);
        valid_indices = sample_indices > 0 & sample_indices <= length(sim.occupation);
        if all(valid_indices)
            reservoir_states(:, iSym) = sim.occupation(sample_indices);
        else
            reservoir_states(:, iSym) = 0; % Handle edge case
        end
    end
end

function W_out = train_readout(reservoir_states, train_target, config)
    train_indices = config.washout1+1 : config.washout1+config.num_train_points;
    X_train_raw = reservoir_states(:, train_indices);
    valid_cols = any(X_train_raw, 1);
    X_train_raw = X_train_raw(:, valid_cols);
    y_train = train_target(valid_cols);
    X_train = [X_train_raw; ones(1, size(X_train_raw, 2))];
    W_out = pinv(X_train * X_train' + config.lambda * eye(size(X_train, 1))) * (X_train * y_train(:));
end

function [nrmse_train, nrmse_test] = test_readout(W_out, reservoir_states, task_data, config)
    train_indices = config.washout1+1 : config.washout1+config.num_train_points;
    X_train_raw = reservoir_states(:, train_indices);
    valid_cols_train = any(X_train_raw, 1);
    X_train = [X_train_raw(:, valid_cols_train); ones(1, sum(valid_cols_train))];
    y_train_hat = X_train' * W_out;
    y_train_true = task_data.train_target(valid_cols_train);
    nrmse_train = sqrt(mean((y_train_hat - y_train_true(:)).^2)) / std(y_train_true);
    test_indices = config.wheretostarttest+1 : config.wheretostarttest + config.num_test_points;
    X_test_raw = reservoir_states(:, test_indices);
    valid_cols_test = any(X_test_raw, 1);
    X_test = [X_test_raw(:, valid_cols_test); ones(1, sum(valid_cols_test))];
    y_test_hat = X_test' * W_out;
    y_test_true = task_data.test_target(valid_cols_test);
    nrmse_test = sqrt(mean((y_test_hat - y_test_true(:)).^2)) / std(y_test_true);
end

function plot_final_result(y_test_hat, task_data, config, nrmse_test)
    % Plots the final predicted vs. actual output for the best run.
    figure('Name', 'Optimal Bayesian Result');
    clf;
    
    % FIX: Correctly identify the target data for plotting
    train_end_idx = config.washout_points + config.train_points;
    test_start_idx = train_end_idx + 1;
    test_end_idx = train_end_idx + config.test_points;
    y_test_true = task_data.target(test_start_idx:test_end_idx);
    
    plot(y_test_true, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Target Signal');
    hold on;
    plot(y_test_hat, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Reservoir Output');
    title(sprintf('Optimal Result | NRMSE=%.4f', nrmse_test));
    xlabel('Time Step');
    ylabel('Normalized Amplitude');
    legend('show', 'Location', 'southeast');
    grid on;
    ylim([-0.1 1.1]);
    
    % Add text box with parameters
    param_str = {
        sprintf('k_{on} = %.2e', config.k_on),
        sprintf('k_{off} = %.2f', config.k_off),
        sprintf('D = %.2e', config.D),
        sprintf('N_{max} = %d', config.N_max),
        sprintf('\\lambda = %.2e', config.lambda)
    };
    annotation('textbox', [0.15 0.15 0.3 0.3], 'String', param_str, 'FitBoxToText', 'on', 'BackgroundColor', 'white');
end