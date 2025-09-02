% =========================================================================
% optimize_sine_hyperparams_bayesopt.m
%
% DESCRIPTION:
%   Uses Bayesian Optimization to efficiently find the optimal combination
%   of 7 key hyperparameters for the sine-to-square PRC transformation task.
%
% REQUIRES:
%   Statistics and Machine Learning Toolbox
%
% =========================================================================

%% 0. Initialize Environment
close all;
clear;
rng('default'); % for reproducibility of the optimization process
fprintf('Starting Bayesian Optimization for Sine-to-Square Hyperparameters...\n');

%% 1. Load Data and Default Config (Done ONCE)
% MODIFIED: Load the pre-generated sine/square data instead of generating NARMA data.
config = get_default_config();
try
    task_data = load_sine_data(config.SINE_FILENAME, config.num_tot_points);
    fprintf('Sine-to-square task data loaded successfully.\n');
catch ME
    fprintf('Error loading data file. Please run createSINEseries first.\n');
    rethrow(ME);
end

%% 2. Define Hyperparameters for Optimization
% These are the 7 variables to be optimized. The ranges are taken from your
% narma optimization script for consistency.
vars = [
    optimizableVariable('k_on', [5e-20, 2e-17], 'Transform', 'log');
    optimizableVariable('k_off', [0.1, 10]);
    optimizableVariable('T', [0.5, 2]);
    optimizableVariable('distance', [1e-6, 50e-6], 'Transform', 'log');
    optimizableVariable('memorywindowlength', [1, 5], 'Type', 'integer');
    optimizableVariable('N_max', [200, 20000]);
    optimizableVariable('D', [0.5e-11, 2e-10], 'Transform', 'log');
];

%% 3. Create Objective Function Handle
% This is the function that bayesopt will try to minimize. We use an
% anonymous function to pass in the static `task_data` and `config` structs.
objFun = @(params) bayesopt_objective_function(params, task_data, config);

%% 4. Run Bayesian Optimization
fprintf('Running Bayesian Optimization for 100 trials...\n');
results = bayesopt(objFun, vars, ...
    'MaxObjectiveEvaluations', 100, ...      % Number of simulations to run
    'IsObjectiveDeterministic', true, ...   % True, since our model is deterministic
    'AcquisitionFunctionName', 'expected-improvement-plus', ... % A good modern default
    'Verbose', 1);                          % Show progress in the command window

%% 5. Display Final Results
fprintf('\n===== Optimization Complete =====\n');
best_params = results.XAtMinObjective;
min_nrmse = results.MinObjective;

fprintf('Best NRMSE (Test) Found: %.4f\n', min_nrmse);
fprintf('With the following hyperparameters:\n');
disp(best_params);

%==========================================================================
%                    --- LOCAL HELPER FUNCTIONS ---
%==========================================================================

function nrmse_test = bayesopt_objective_function(params_table, task_data, default_config)
    % This function is called by bayesopt for each trial.
    % It takes the proposed parameters, runs the simulation, and returns the NRMSE.
    
    % 1. Create a config for this specific run
    run_config = default_config;
    run_config.k_on = params_table.k_on;
    run_config.k_off = params_table.k_off;
    run_config.T = params_table.T;
    run_config.distance = params_table.distance;
    run_config.memorywindowlength = params_table.memorywindowlength;
    run_config.N_max = params_table.N_max;
    run_config.D = params_table.D;
    
    % Update dependent parameters
    run_config.Nres = run_config.Nres_init * run_config.memorywindowlength;
    
    % 2. Run the full simulation and learning pipeline
    [~, reservoir_states] = run_channel_simulation(task_data.input_series, run_config);
    W_out = train_readout(reservoir_states, task_data.train_target, run_config);
    [~, nrmse_test] = test_readout(W_out, reservoir_states, task_data, run_config);
end

% --- NOTE: The helper functions below are copied from your ---
% --- validated plot_sine_numerical_analysis.m script. ---

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
    
    % Data Source
    config.SINE_FILENAME = 'SINEseries_len5000_p25.mat';
end

function task_data = load_sine_data(filename, required_len)
    loaded_data = load(filename, 'input_sine_series', 'target_square_series');
    input_series  = loaded_data.input_sine_series;
    target_series = loaded_data.target_square_series;
    task_data.input_series = input_series(1:required_len);
    task_data.target_series = target_series(1:required_len);
    config = get_default_config();
    task_data.train_target = task_data.target_series(config.washout1+1 : config.washout1 + config.num_train_points);
    task_data.test_target  = task_data.target_series(config.wheretostarttest + 1 : config.wheretostarttest + config.num_test_points);
end

function [sim, reservoir_states] = run_channel_simulation(u_series, config)
    N_i = config.N_min + (u_series - 0) / (1 - 0) * (config.N_max - config.N_min);
    t_total = length(N_i) * config.T;
    sim.time = 0:config.dt:t_total;
    sim.concentration = zeros(size(sim.time));
    Tpeak = config.distance^2 / (6 * config.D);
    memory_length = round((config.memlengthsweep * Tpeak) / config.T) * config.T;
    for iSym = 1:length(N_i)
        t_start = (iSym - 1) * config.T;
        t_end   = t_start + memory_length;
        idx_start = floor(t_start / config.dt) + 2;
        idx_end   = ceil(t_end / config.dt) + 1;
        idxRange  = idx_start:min(idx_end, length(sim.time));
        if isempty(idxRange), continue; end
        t_local = sim.time(idxRange) - t_start;
        pulse = (N_i(iSym) ./ ((4*pi*config.D*t_local).^(3/2))) .* exp(-config.distance^2./(4*config.D*t_local));
        sim.concentration(idxRange) = sim.concentration(idxRange) + pulse;
    end
    sim.occupation = zeros(size(sim.time));
    for idxT = 2:length(sim.time)
        c_t = sim.concentration(idxT - 1);
        n_t = sim.occupation(idxT - 1);
        dn_dt = config.k_on*(config.N - n_t*config.N)*c_t - config.k_off*n_t*config.N;
        sim.occupation(idxT) = n_t + (dn_dt/config.N)*config.dt;
    end
    steps_per_symbol = config.T / config.dt;
    num_symbols = config.num_tot_points;
    reservoir_states = zeros(config.Nres, num_symbols);
    for iSym = 1:num_symbols
        t_symbol_start = (iSym - 1) * config.T;
        start_idx = find(sim.time >= t_symbol_start, 1);
        if isempty(start_idx), continue; end
        sample_indices_float = start_idx - (config.memorywindowlength-1)*steps_per_symbol + (0:(config.Nres-1))*(steps_per_symbol/config.Nres)*config.memorywindowlength;
        sample_indices = round(sample_indices_float);
        if all(sample_indices > 0 & sample_indices <= length(sim.occupation))
            reservoir_states(:, iSym) = sim.occupation(sample_indices);
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