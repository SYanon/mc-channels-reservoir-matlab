% =========================================================================
% optimize_mg_hyperparams_bayesopt.m
%
% DESCRIPTION:
%   Uses Bayesian Optimization to efficiently find the optimal combination
%   of 7 key hyperparameters for the Mackey-Glass PRC task.
%
% REQUIRES:
%   Statistics and Machine Learning Toolbox
%
% =========================================================================

%% 0. Initialize Environment
close all;
clear;
rng('default'); % for reproducibility of the optimization process
fprintf('Starting Bayesian Optimization for MG Hyperparameters...\n');

%% 1. Load Data and Default Config (Done ONCE)
% We load these once to pass into the objective function, which is efficient.
config = get_default_config();
mg_data = load_mg_data(config.MG_FILENAME, config.num_tot_points, config.predictlength);
fprintf('Mackey-Glass data loaded.\n');

%% 2. Define Hyperparameters for Optimization
% Define the 7 variables to be optimized, their ranges, and their types.
% Using a 'log' transform is crucial for parameters that span orders of magnitude.
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
% The objective function is what bayesopt tries to minimize. Here, it's a
% wrapper around our simulation that returns the NRMSE. We use an anonymous
% function to pass in the static `mg_data` and `config` structs.
objFun = @(params) objectiveFunction(params, mg_data, config);

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

function nrmse_test = objectiveFunction(params_table, mg_data, default_config)
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
    run_config.Nres = run_config.Nres * run_config.memorywindowlength;
    
    % 2. Run the full simulation and learning pipeline
    [~, reservoir_states] = run_channel_simulation(mg_data.input_series, run_config);
    W_out = train_readout(reservoir_states, mg_data.train_target, run_config);
    [~, nrmse_test] = test_readout(W_out, reservoir_states, mg_data, run_config);
end


% --- NOTE: The helper functions below are copied directly from ---
% --- your validated mg_numerical_analysis.m script. ---

function config = get_default_config()
    config.N     = 500;
    config.N_min = 100;
    config.lambda = 1e-6;
    config.Nres   = 50;
    config.predictlength = 6;
    config.washout1         = 500;
    config.num_train_points = 500;
    config.wheretostarttest = 1200;
    config.num_test_points  = 500;
    config.washout2 = config.wheretostarttest - (config.washout1 + config.num_train_points);
    config.num_tot_points = config.washout1 + config.washout2 + config.num_train_points + config.num_test_points;
    config.dt = 0.001;
    config.memlengthsweep = 100;
    config.MG_FILENAME = 'MGseries_RK4_tau17_beta0.20_gamma0.1_n10_len5000_dt1.0.mat';
end

function mg_data = load_mg_data(filename, required_len, predict_len)
    loaded_data = load(filename, 'mackey_glass_series');
    full_series_raw = loaded_data.mackey_glass_series;
    series_segment = full_series_raw(1 : required_len + predict_len);
    series_normalized = (series_segment - min(series_segment)) / (max(series_segment) - min(series_segment));
    mg_data.input_series  = series_normalized(1:end - predict_len);
    mg_data.target_series = series_normalized((predict_len+1):end);
    config = get_default_config();
    mg_data.train_target = mg_data.target_series(config.washout1+1 : config.washout1 + config.num_train_points);
    mg_data.test_target  = mg_data.target_series(config.washout1 + config.num_train_points + config.washout2 + 1 : end);
end

function [sim, reservoir_states] = run_channel_simulation(input_series, config)
    inp_min = min(input_series);
    inp_max = max(input_series);
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
        if all(sample_indices > 0 & sample_indices <= length(sim.occupation))
            reservoir_states(:, iSym) = sim.occupation(sample_indices);
        end
    end
end

function W_out = train_readout(reservoir_states, train_target, config)
    train_states = reservoir_states(:, config.washout1+1 : config.washout1+config.num_train_points);
    X_train = [train_states; ones(1, config.num_train_points)];
    y_train = train_target(:);
    W_out = pinv(X_train * X_train' + config.lambda * eye(size(X_train, 1))) * (X_train * y_train);
end

function [nrmse_train, nrmse_test] = test_readout(W_out, reservoir_states, mg_data, config)
    train_states = reservoir_states(:, config.washout1+1 : config.washout1+config.num_train_points);
    X_train = [train_states; ones(1, config.num_train_points)];
    y_train_hat = X_train' * W_out;
    nrmse_train = sqrt(mean((y_train_hat - mg_data.train_target(:)).^2)) / std(mg_data.train_target);
    test_indices = config.washout1+config.num_train_points+config.washout2+1 : config.num_tot_points;
    test_states = reservoir_states(:, test_indices);
    X_test = [test_states; ones(1, config.num_test_points)];
    y_test_hat = X_test' * W_out;
    nrmse_test = sqrt(mean((y_test_hat - mg_data.test_target(:)).^2)) / std(mg_data.test_target);
end