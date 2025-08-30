% =========================================================================
% optimize_kl_narma_hyperparams.m
%
% DESCRIPTION:
%   Uses Bayesian Optimization to find the optimal physical hyperparameters
%   for the NARMA10 PRC task WHEN a specific post-processing
%   technique (e.g., uniform, multi-random) is active.
%
% REQUIRES:
%   Statistics and Machine Learning Toolbox
%
% =========================================================================

%% 0. Initialize Environment
close all;
clear;
rng('default'); % for reproducibility of the optimization process
fprintf('Starting Bayesian Optimization with Post-Processing for NARMA10...\n');

%% 1. Load Data and Default Config (Done ONCE)
config = get_default_config();
narma_data = generate_narma10_data(config);
fprintf('NARMA10 data generated.\n');
fprintf('Post-processing method set to: ''%s''\n', config.postProcessingMethod);

%% 2. Define Hyperparameters for Optimization
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
objFun = @(params) bayesopt_objective_function(params, narma_data, config);

%% 4. Run Bayesian Optimization
fprintf('Running Bayesian Optimization for 100 trials...\n');
results = bayesopt(objFun, vars, ...
    'MaxObjectiveEvaluations', 100, ...
    'IsObjectiveDeterministic', true, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'Verbose', 1);

%% 5. Display Final Results
fprintf('\n===== Optimization Complete (Method: ''%s'') =====\n', config.postProcessingMethod);
best_params = results.XAtMinObjective;
min_nrmse = results.MinObjective;

fprintf('Best NRMSE (Test) Found: %.4f\n', min_nrmse);
fprintf('With the following hyperparameters:\n');
disp(best_params);

%==========================================================================
%                    --- LOCAL HELPER FUNCTIONS ---
%==========================================================================

function nrmse_test = bayesopt_objective_function(params_table, narma_data, default_config)
    % This is the core function called by bayesopt for each trial.
    
    % 1. Create a config for this specific run
    run_config = default_config;
    run_config.k_on = params_table.k_on;
    run_config.k_off = params_table.k_off;
    run_config.T = params_table.T;
    run_config.distance = params_table.distance;
    run_config.memorywindowlength = params_table.memorywindowlength;
    run_config.N_max = params_table.N_max;
    run_config.D = params_table.D;
    
    run_config.Nres = run_config.Nres_init * run_config.memorywindowlength;
    
    % 2. Run the core simulation to get the original states
    [~, reservoir_states] = run_channel_simulation(narma_data.u, run_config);
    
    % 3. Apply the selected post-processing method
    [augmented_states, run_config] = apply_post_processing(reservoir_states, run_config);
    
    % 4. Train and test on the (potentially augmented) states
    W_out = train_readout(augmented_states, narma_data.train_target, run_config);
    [~, nrmse_test] = test_readout(W_out, augmented_states, narma_data, run_config);
end

% --- Helper functions below are combined from our previous scripts ---

function config = get_default_config()
    % --- Post-Processing Controls ---
    config.postProcessingMethod = 'uniform'; % Options: 'none', 'uniform', 'multi-random'
    config.uniformDelay = 8;
    config.numReplicas = 4;
    config.maxRandomDelay = 40;
    
    % --- Fixed (non-optimized) parameters ---
    config.N     = 500;
    config.N_min = 100;
    config.lambda = 1e-10;
    config.Nres_init = 100; % Base number of nodes
    config.washout1         = 300;
    config.num_train_points = 2000;
    config.wheretostarttest = 2500;
    config.num_test_points  = 2000;
    config.washout2 = config.wheretostarttest - (config.washout1 + config.num_train_points);
    config.num_tot_points = config.washout1 + config.washout2 + config.num_train_points + config.num_test_points;
    config.dt = 0.001;
    config.memlengthsweep = 100;
    config.offset = 0;
end

function narma_data = generate_narma10_data(config)
    narma_len = config.num_tot_points + 10 + 1;
    u = 0.5 * rand(narma_len, 1);
    q = zeros(narma_len, 1);
    for n = 10:(config.num_tot_points - 1)
        q(n+1) = 0.3*q(n) + 0.05*q(n)*sum(q(n-9:n)) + 1.5*u(n-9)*u(n) + 0.1;
    end
    narma_data.u = u;
    narma_data.q = q;
    train_indices = (config.washout1+2) : (config.washout1+config.num_train_points+1);
    test_indices = (config.washout1+config.num_train_points+config.washout2+2) : (config.num_tot_points+1);
    narma_data.train_target = q(train_indices);
    narma_data.test_target = q(test_indices);
end

function [sim, reservoir_states] = run_channel_simulation(u_series, config)
    N_i = config.N_min + (u_series - min(u_series)) / (max(u_series) - min(u_series)) * (config.N_max - config.N_min);
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
        t_start = (iSym - 1) * config.T + config.offset;
        start_idx = find(sim.time >= t_start, 1);
        if isempty(start_idx), continue; end
        sample_indices_float = start_idx - (config.memorywindowlength-1)*steps_per_symbol + (0:(config.Nres-1))*(steps_per_symbol/config.Nres)*config.memorywindowlength;
        sample_indices = round(sample_indices_float);
        if all(sample_indices > 0 & sample_indices <= length(sim.occupation))
            reservoir_states(:, iSym) = sim.occupation(sample_indices);
        end
    end
end

function [augmented_states, config] = apply_post_processing(reservoir_states, config)
    switch config.postProcessingMethod
        case 'none'
            augmented_states = reservoir_states;
            config.feature_dimension = config.Nres; 
        case 'uniform'
            delay = config.uniformDelay;
            original_part = reservoir_states;
            delayed_part = NaN(size(reservoir_states));
            delayed_part(:, (1+delay):end) = reservoir_states(:, 1:(end-delay));
            augmented_states = [original_part; delayed_part];
            config.feature_dimension = 2 * config.Nres;
        case 'multi-random'
            N = config.numReplicas; R = config.maxRandomDelay;
            all_replicas = [];
            for i_rep = 1:N
                delays = randi([0 R], [config.Nres, 1]);
                current_replica = NaN(config.Nres, size(reservoir_states, 2));
                for j_feature = 1:config.Nres
                    d = delays(j_feature);
                    if d == 0, current_replica(j_feature, :) = reservoir_states(j_feature, :);
                    else, current_replica(j_feature, (1+d):end) = reservoir_states(j_feature, 1:(end-d)); end
                end
                all_replicas = [all_replicas; current_replica];
            end
            augmented_states = all_replicas;
            config.feature_dimension = N * config.Nres;
    end
end

function W_out = train_readout(augmented_states, train_target, config)
    train_states_raw = augmented_states(:, config.washout1+1 : config.washout1+config.num_train_points);
    valid_cols = ~any(isnan(train_states_raw), 1);
    train_states = train_states_raw(:, valid_cols);
    y_train = train_target(valid_cols);
    X_train = [train_states; ones(1, size(train_states, 2))];
    W_out = pinv(X_train * X_train' + config.lambda * eye(config.feature_dimension + 1)) * (X_train * y_train(:));
end

function [nrmse_train, nrmse_test] = test_readout(W_out, augmented_states, narma_data, config)
    train_states_raw = augmented_states(:, config.washout1+1 : config.washout1+config.num_train_points);
    valid_train_cols = ~any(isnan(train_states_raw), 1);
    X_train = [train_states_raw(:, valid_train_cols); ones(1, sum(valid_train_cols))];
    y_train_hat = X_train' * W_out;
    y_train_true = narma_data.train_target(valid_train_cols);
    nrmse_train = sqrt(mean((y_train_hat - y_train_true(:)).^2)) / std(y_train_true);
    test_indices = config.washout1+config.num_train_points+config.washout2+1 : config.num_tot_points;
    test_states_raw = augmented_states(:, test_indices);
    valid_test_cols = ~any(isnan(test_states_raw), 1);
    X_test = [test_states_raw(:, valid_test_cols); ones(1, sum(valid_test_cols))];
    y_test_hat = X_test' * W_out;
    y_test_true = narma_data.test_target(valid_test_cols);
    nrmse_test = sqrt(mean((y_test_hat - y_test_true(:)).^2)) / std(y_test_true);
end