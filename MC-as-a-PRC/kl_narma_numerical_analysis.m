% =========================================================================
% kl_narma_numerical_sweep.m
%
% DESCRIPTION:
%   A modular script for the NARMA10 benchmark, now including state
%   matrix post-processing techniques inspired by Jaurigue, Lüdge, et al.,
%   to test for performance improvements.
%
% =========================================================================

%% 0. Initialize Environment & Define Configuration
close all;
clear;
rng(0); % for reproducibility
fprintf('Starting MODIFIED NARMA10 parameter sweep with post-processing...\n');

% --- Central Configuration Struct ---
config = get_default_config();

%% 1. Generate Data (Done ONCE)
fprintf('Generating NARMA10 data...\n');
narma_data = generate_narma10_data(config);
fprintf('Data generation complete.\n');

%% 2. Prepare for Sweep
nP1 = length(config.param1_values);
nP2 = length(config.param2_values);
results = initialize_results_struct(nP1, nP2);

p1_name = config.param1_name;
p1_vals = config.param1_values;
p2_name = config.param2_name;
p2_vals = config.param2_values;

%% 3. Main Parameter Sweep Loop
fprintf('Starting sweep over "%s" and "%s"...\n', p1_name, p2_name);
total_runs = nP1 * nP2;
for i2 = 1:nP2
    for i1 = 1:nP1
        tic;
        
        % --- Update parameters for the current run ---
        run_config = config;
        run_config.(p2_name) = p2_vals(i2);
        run_config.(p1_name) = p1_vals(i1);
        
        fprintf('Run (%d/%d): %s=%g, %s=%g\n', ...
                (i2-1)*nP1 + i1, total_runs, p1_name, run_config.(p1_name), p2_name, run_config.(p2_name));

        %% 4. Run Core Simulation
        [sim, reservoir_states] = run_channel_simulation(narma_data.u, run_config);
        
        %% 5. Apply Post-Processing to State Matrix (NEW STEP)
        [augmented_states, run_config] = apply_post_processing(reservoir_states, run_config);
        
        %% 6. Train and Test Readout
        W_out = train_readout(augmented_states, narma_data.train_target, run_config);
        [nrmse_train, nrmse_test] = test_readout(W_out, augmented_states, narma_data, run_config);

        %% 7. Store Results
        results.nrmse_train(i1, i2) = nrmse_train;
        results.nrmse_test(i1, i2)  = nrmse_test;
        
        fprintf('  -> NRMSE(Test)=%.3g. (took %.2fs)\n', nrmse_test, toc);
    end
end

%% 8. Visualize Results
fprintf('Sweep complete. Generating plots...\n');
plot_sweep_results(results, config);
fprintf('All done.\n');

%==========================================================================
%                    --- LOCAL HELPER FUNCTIONS ---
%==========================================================================

function config = get_default_config()
    % Central location for all default parameters.
    
    % --- Sweep Parameters ---
    config.param1_name   = 'k_on';
    config.param1_values = [5e-20, 1e-19, 2e-19, 5e-19, 1e-18, 2e-18, 5e-18, 1e-17, 2e-17];
    config.param2_name   = 'k_off';
    config.param2_values = [0.1, 0.2, 0.5, 1, 2, 5, 10];
    
    % =====================================================================
    % --- NEW SECTION: Post-Processing Controls ---
    % =====================================================================
    config.postProcessingMethod = 'none'; % Options: 'none', 'uniform', 'multi-random'
    config.uniformDelay = 8;        % Integer step delay for 'uniform' method
    config.numReplicas = 4;         % Number of replicas for 'multi-random'
    config.maxRandomDelay = 40;     % Max random shift for 'multi-random'
    % =====================================================================

    % --- System & Reservoir Parameters ---
    config.N     = 500;
    config.k_on  = 1e-18;
    config.k_off = 1;
    config.N_min = 100;
    config.N_max = 3000;
    config.lambda = 1e-10;
    config.Nres = 100;
    config.memorywindowlength = 1;
    config.distance = 10e-6;
    config.D        = 1e-11;
    config.T  = 1;
    
    % --- Data Lengths ---
    config.washout1         = 300;
    config.num_train_points = 2000;
    config.wheretostarttest = 2500;
    config.num_test_points  = 2000;
    config.washout2 = config.wheretostarttest - (config.washout1 + config.num_train_points);
    config.num_tot_points = config.washout1 + config.washout2 + config.num_train_points + config.num_test_points;
    
    % --- Simulation Control ---
    config.dt = 0.001;
    config.memlengthsweep = 100;
end

%--------------------------------------------------------------------------
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

%--------------------------------------------------------------------------
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

%--------------------------------------------------------------------------
function [augmented_states, config] = apply_post_processing(reservoir_states, config)
    % This function is copied from the kl_mg_numerical_analysis.m script.
    % It is generic and works on any state matrix.
    switch config.postProcessingMethod
        case 'none'
            augmented_states = reservoir_states;
            config.feature_dimension = config.Nres; 
        case 'uniform'
            delay = config.uniformDelay;
            if delay < 1, augmented_states = reservoir_states; config.feature_dimension = config.Nres; return; end
            original_part = reservoir_states;
            delayed_part = NaN(size(reservoir_states));
            delayed_part(:, (1+delay):end) = reservoir_states(:, 1:(end-delay));
            augmented_states = [original_part; delayed_part];
            config.feature_dimension = 2 * config.Nres;
        case 'multi-random'
            N = config.numReplicas;
            R = config.maxRandomDelay;
            if N <= 1, augmented_states = reservoir_states; config.feature_dimension = config.Nres; return; end
            all_replicas = [];
            for i_rep = 1:N
                delays = randi([0 R], [config.Nres, 1]);
                current_replica = NaN(config.Nres, size(reservoir_states, 2));
                for j_feature = 1:config.Nres
                    d = delays(j_feature);
                    if d == 0
                        current_replica(j_feature, :) = reservoir_states(j_feature, :);
                    else
                        current_replica(j_feature, (1+d):end) = reservoir_states(j_feature, 1:(end-d));
                    end
                end
                all_replicas = [all_replicas; current_replica];
            end
            augmented_states = all_replicas;
            config.feature_dimension = N * config.Nres;
        otherwise
            error('Unknown post-processing method: %s', config.postProcessingMethod);
    end
    fprintf('  Applying method: %s -> New feature dimension O = %d\n', config.postProcessingMethod, config.feature_dimension);
end

%--------------------------------------------------------------------------
function W_out = train_readout(augmented_states, train_target, config)
    % Modified to handle augmented states which may contain NaNs.
    train_states_raw = augmented_states(:, config.washout1+1 : config.washout1+config.num_train_points);
    valid_cols = ~any(isnan(train_states_raw), 1);
    train_states = train_states_raw(:, valid_cols);
    y_train = train_target(valid_cols);
    X_train = [train_states; ones(1, size(train_states, 2))];
    W_out = pinv(X_train * X_train' + config.lambda * eye(config.feature_dimension + 1)) * (X_train * y_train(:));
end

%--------------------------------------------------------------------------
function [nrmse_train, nrmse_test] = test_readout(W_out, augmented_states, narma_data, config)
    % Modified to handle augmented states which may contain NaNs.
    % --- Test on Training Data ---
    train_states_raw = augmented_states(:, config.washout1+1 : config.washout1+config.num_train_points);
    valid_train_cols = ~any(isnan(train_states_raw), 1);
    X_train = [train_states_raw(:, valid_train_cols); ones(1, sum(valid_train_cols))];
    y_train_hat = X_train' * W_out;
    y_train_true = narma_data.train_target(valid_train_cols);
    nrmse_train = sqrt(mean((y_train_hat - y_train_true(:)).^2)) / std(y_train_true);
    
    % --- Test on Testing Data ---
    test_indices = config.wheretostarttest+1 : config.wheretostarttest + config.num_test_points;
    test_states_raw = augmented_states(:, test_indices);
    valid_test_cols = ~any(isnan(test_states_raw), 1);
    X_test = [test_states_raw(:, valid_test_cols); ones(1, sum(valid_test_cols))];
    y_test_hat = X_test' * W_out;
    y_test_true = narma_data.test_target(valid_test_cols);
    nrmse_test = sqrt(mean((y_test_hat - y_test_true(:)).^2)) / std(y_test_true);
end

%--------------------------------------------------------------------------
function results = initialize_results_struct(nP1, nP2)
    results.nrmse_train = NaN(nP1, nP2);
    results.nrmse_test  = NaN(nP1, nP2);
end

%--------------------------------------------------------------------------
function plot_sweep_results(results, config)
    [X, Y] = meshgrid(config.param1_values, config.param2_values);
    plot_heatmap(X, Y, results.nrmse_test', 'NRMSE (Test)', config);
    plot_heatmap(X, Y, results.nrmse_train', 'NRMSE (Train)', config);
end

%--------------------------------------------------------------------------
function plot_heatmap(X, Y, Z, title_str, config)
    figure('Name', title_str);
    contourf(X, Y, Z, 20, 'LineColor', 'none');
    shading interp;
    cb = colorbar;
    colormap(flipud(parula));
    ylabel(cb, 'NRMSE');
    if ~isempty(config.param1_values) && min(config.param1_values)>0 && max(config.param1_values)/min(config.param1_values) > 50, set(gca, 'XScale', 'log'); end
    if ~isempty(config.param2_values) && min(config.param2_values)>0 && max(config.param2_values)/min(config.param2_values) > 50, set(gca, 'YScale', 'log'); end
    xlabel(strrep(config.param1_name, '_', ' '));
    ylabel(strrep(config.param2_name, '_', ' '));
    title(sprintf('%s (Method: %s)', title_str, config.postProcessingMethod));
end