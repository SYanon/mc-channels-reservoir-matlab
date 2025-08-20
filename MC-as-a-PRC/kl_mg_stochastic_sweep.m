% =========================================================================
% kl_mg_stochastic_sweep.m
%
% DESCRIPTION:
%   Integrates Lüdge-Jaurigue (KL) post-processing methods into the
%   modular stochastic sweep script. It applies these techniques to both
%   the numerical and the pre-computed Smoldyn data.
% =========================================================================

%% 0. Initialize Environment & Define Configuration
close all;
clear;
fprintf('Starting KL MODULAR Stochastic Mackey-Glass parameter sweep...\n');

% --- Central Configuration Struct ---
config = get_default_config();

%% 1. Load and Prepare Data (Done ONCE)
mg_series_data = load_mg_series(config.MG_FILENAME, config.num_tot_points, max(config.param2_values));

%% 2. Prepare for Sweep
nP1 = length(config.param1_values);
nP2 = length(config.param2_values);
results = initialize_results_struct(nP1, nP2);

p1_name = config.param1_name;
p1_vals = config.param1_values;
p2_name = config.param2_name;
p2_vals = config.param2_values;

%% 3. Main Parameter Sweep Loop
fprintf('Starting sweep over %s and %s...\n', p1_name, p2_name);
for i2 = 1:nP2
    for i1 = 1:nP1
        tic;
        
        % --- Update parameters for the current run ---
        run_config = config;
        run_config.(p2_name) = p2_vals(i2);
        run_config.(p1_name) = p1_vals(i1);
        
        current_mg_data = get_data_splits(mg_series_data, run_config);
        
        % =================== NUMERICAL (DETERMINISTIC) MODEL ===================
        [~, reservoir_states_num] = run_channel_simulation(current_mg_data.input_series, run_config);
        [augmented_states_num, run_config] = apply_post_processing(reservoir_states_num, run_config); % KL Post-processing
        W_out_num = train_readout(augmented_states_num, current_mg_data.train_target, run_config);
        [nrmse_train_num, nrmse_test_num] = test_readout(W_out_num, augmented_states_num, current_mg_data, run_config);
        
        % =================== STOCHASTIC (SMOLDYN) MODEL ===================
        [reservoir_states_smol_raw, success] = load_smoldyn_data(run_config);
        
        if success
            filtered_states_smol = apply_signal_filter(reservoir_states_smol_raw, run_config);
            [augmented_states_smol, run_config] = apply_post_processing(filtered_states_smol, run_config); % KL Post-processing
            W_out_smol = train_readout(augmented_states_smol, current_mg_data.train_target, run_config);
            [nrmse_train_smol, nrmse_test_smol] = test_readout(W_out_smol, augmented_states_smol, current_mg_data, run_config);
        else
            fprintf('    -> WARNING: Smoldyn data not found. Skipping.\n');
            nrmse_train_smol = NaN;
            nrmse_test_smol  = NaN;
        end

        %% 7. Store Results
        results.nrmse_train_num(i1, i2) = nrmse_train_num;
        results.nrmse_test_num(i1, i2)  = nrmse_test_num;
        results.nrmse_train_smol(i1, i2) = nrmse_train_smol;
        results.nrmse_test_smol(i1, i2)  = nrmse_test_smol;

        fprintf('Run (%d/%d): %s=%g, %s=%g -> NRMSE(Num)=%.3g, NRMSE(Smol)=%.3g. (took %.2fs)\n', ...
                (i2-1)*nP1 + i1, nP1*nP2, p1_name, run_config.(p1_name), p2_name, run_config.(p2_name), ...
                nrmse_test_num, nrmse_test_smol, toc);
    end
end

%% 9. Visualize Results
fprintf('Sweep complete. Generating plots...\n');
plot_sweep_results(results, config);
fprintf('All done.\n');

%==========================================================================
%                    --- LOCAL HELPER FUNCTIONS ---
%==========================================================================

function config = get_default_config()
    % Central location for all default parameters.
    config.param1_name   = 'movmean_window';
    config.param1_values = [1 100 200 500 1000 2000];
    config.param2_name   = 'predictlength';
    config.param2_values = [1 5 10 20 50 100];
    
    % =====================================================================
    % --- NEW SECTION: Post-Processing Controls ---
    % =====================================================================
    config.postProcessingMethod = 'multi-random'; % Options: 'none', 'uniform', 'multi-random'
    config.uniformDelay = 8; 
    config.numReplicas = 4;
    config.maxRandomDelay = 40;
    % =====================================================================
    
    config.smoldyn_data_dir = 'MG_5000_T1_koff1'; 
    config.movmean_on = false;
    config.N     = 500;
    config.k_on  = 1e-18;
    config.k_off = 1;
    config.N_min = 100;
    config.N_max = 3000;
    config.lambda = 1e-6;
    config.Nres   = 100;
    config.memorywindowlength = 1;
    config.distance = 10e-6;
    config.D        = 1e-11;
    config.T  = 1;
    config.washout1         = 250;
    config.num_train_points = 2000;
    config.wheretostarttest = 2500;
    config.num_test_points  = 2000;
    config.washout2 = config.wheretostarttest - (config.washout1 + config.num_train_points);
    config.num_tot_points = config.washout1 + config.washout2 + config.num_train_points + config.num_test_points;
    config.dt = 0.001;
    config.smol_dt = 0.01;
    config.memlengthsweep = 100;
    config.MG_FILENAME = 'MGseries_RK4_tau17_beta0.20_gamma0.1_n10_len5000_dt1.0.mat';
end
%--------------------------------------------------------------------------
function series_data = load_mg_series(filename, required_len, max_predict_len)
    try
        loaded_data = load(filename, 'mackey_glass_series');
        full_series_raw = loaded_data.mackey_glass_series;
    catch
        error('Failed to load MG data from "%s".', filename);
    end
    total_len = required_len + max_predict_len;
    if length(full_series_raw) < total_len
        error('Loaded series is too short. Required length: %d.', total_len);
    end
    series_segment = full_series_raw(1 : total_len);
    series_data = (series_segment - min(series_segment)) / (max(series_segment) - min(series_segment));
end
%--------------------------------------------------------------------------
function mg_data = get_data_splits(full_series, run_config)
    predict_len = run_config.predictlength;
    mg_data.input_series  = full_series(1 : end - predict_len);
    target_series = full_series(predict_len + 1 : end);
    train_end_idx = run_config.washout1 + run_config.num_train_points;
    mg_data.train_target = target_series(run_config.washout1 + 1 : train_end_idx);
    test_start_idx = run_config.wheretostarttest + 1;
    mg_data.test_target = target_series(test_start_idx : test_start_idx + run_config.num_test_points - 1);
    mg_data.train_target = mg_data.train_target(:);
    mg_data.test_target = mg_data.test_target(:);
end
%--------------------------------------------------------------------------
function [reservoir_states, success] = load_smoldyn_data(config)
    filename = 'allmolecules_varNo_2_varValue_1_iter_1.txt';
    filepath = fullfile(config.smoldyn_data_dir, filename);
    if ~exist(filepath, 'file'), reservoir_states = []; success = false; return; end
    try
        data_temp = importdata(filepath, ' ', 1);
        smol_data = data_temp.data;
        time_vec = smol_data(:, 1);
        active_receptors = smol_data(:, end) / config.N;
        steps_per_symbol = config.T / config.smol_dt;
        num_symbols = length(time_vec) / steps_per_symbol; % Adjust based on actual data length
        reservoir_states = zeros(config.Nres, floor(num_symbols));
        for iSym = 1:floor(num_symbols)
            t_symbol_start = (iSym - 1) * config.T;
            start_idx = find(time_vec >= t_symbol_start, 1);
            if isempty(start_idx), continue; end
            sample_indices_float = start_idx - (config.memorywindowlength-1)*steps_per_symbol + (0:(config.Nres-1))*(steps_per_symbol/config.Nres)*config.memorywindowlength;
            sample_indices = round(sample_indices_float);
            valid_indices = sample_indices > 0 & sample_indices <= length(active_receptors);
            if all(valid_indices)
                reservoir_states(:, iSym) = active_receptors(sample_indices);
            end
        end
        success = true;
    catch ME
        warning('Failed to read or process Smoldyn file "%s": %s', filepath, ME.message);
        reservoir_states = [];
        success = false;
    end
end
%--------------------------------------------------------------------------
function filtered_states = apply_signal_filter(reservoir_states, config)
    if config.movmean_on && config.movmean_window > 1
        k = config.movmean_window;
        filtered_states = movmean(reservoir_states, [k-1 0], 2);
    else
        filtered_states = reservoir_states;
    end
end
%--------------------------------------------------------------------------
function [augmented_states, config] = apply_post_processing(reservoir_states, config)
    num_total_points = size(reservoir_states, 2);
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
                current_replica = NaN(config.Nres, num_total_points);
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
end
%--------------------------------------------------------------------------
function [sim, reservoir_states] = run_channel_simulation(input_series, config)
    N_i = config.N_min + (input_series - min(input_series))/(max(input_series) - min(input_series)) * (config.N_max - config.N_min);
    t_total = length(input_series) * config.T;
    sim.time = 0:config.dt:t_total;
    sim.concentration = zeros(size(sim.time));
    Tpeak = config.distance^2/(6*config.D);
    memory_length = round((config.memlengthsweep * Tpeak)/config.T)*config.T;
    for iSym = 1:length(N_i)
        t_symbol_start = (iSym - 1) * config.T;
        t_memory_end   = t_symbol_start + memory_length;
        start_idx = round(t_symbol_start / config.dt) + 1;
        end_idx = round(t_memory_end / config.dt) + 1;
        end_idx = min(end_idx, length(sim.time));
        idxRange = start_idx:end_idx;
        if isempty(idxRange) || idxRange(1) > length(sim.time), continue; end
        t_local = sim.time(idxRange) - t_symbol_start;
        t_local(t_local <= 0) = 1e-9;
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
    num_symbols = length(input_series);
    reservoir_states = zeros(config.Nres, num_symbols);
    for iSym = 1:num_symbols
        t_symbol_start = (iSym - 1) * config.T;
        start_idx = round(t_symbol_start/config.dt) + 1;
        sample_indices_float = start_idx - (config.memorywindowlength-1)*steps_per_symbol + (0:(config.Nres-1))*(steps_per_symbol/config.Nres)*config.memorywindowlength;
        sample_indices = round(sample_indices_float);
        valid_indices = sample_indices > 0 & sample_indices <= length(sim.occupation);
        if all(valid_indices)
            reservoir_states(:, iSym) = sim.occupation(sample_indices);
        end
    end
end
%--------------------------------------------------------------------------
function W_out = train_readout(augmented_states, train_target, config)
    train_states_raw = augmented_states(:, config.washout1+1 : config.washout1+config.num_train_points);
    valid_cols = ~any(isnan(train_states_raw), 1);
    train_states = train_states_raw(:, valid_cols);
    y_train = train_target(valid_cols);
    X_train = [train_states; ones(1, size(train_states, 2))];
    W_out = pinv(X_train * X_train' + config.lambda * eye(config.feature_dimension + 1)) * (X_train * y_train);
end
%--------------------------------------------------------------------------
function [nrmse_train, nrmse_test] = test_readout(W_out, augmented_states, mg_data, config)
    train_states_raw = augmented_states(:, config.washout1+1 : config.washout1+config.num_train_points);
    valid_train_cols = ~any(isnan(train_states_raw), 1);
    train_states = train_states_raw(:, valid_train_cols);
    y_train_valid = mg_data.train_target(valid_train_cols);
    if isempty(y_train_valid)
        nrmse_train = NaN;
    else
        X_train = [train_states; ones(1, size(train_states, 2))];
        y_train_hat = X_train' * W_out;
        nrmse_train = sqrt(mean((y_train_hat - y_train_valid).^2)) / std(y_train_valid);
    end
    test_indices = config.wheretostarttest + 1 : config.wheretostarttest + config.num_test_points;
    test_states_raw = augmented_states(:, test_indices);
    valid_test_cols = ~any(isnan(test_states_raw), 1);
    test_states = test_states_raw(:, valid_test_cols);
    y_test_valid = mg_data.test_target(valid_test_cols);
    if isempty(y_test_valid)
        nrmse_test = NaN;
    else
        X_test = [test_states; ones(1, size(test_states, 2))];
        y_test_hat = X_test' * W_out;
        nrmse_test = sqrt(mean((y_test_hat - y_test_valid).^2)) / std(y_test_valid);
    end
end
%--------------------------------------------------------------------------
function results = initialize_results_struct(nP1, nP2)
    results.nrmse_train_num  = NaN(nP1, nP2);
    results.nrmse_test_num   = NaN(nP1, nP2);
    results.nrmse_train_smol = NaN(nP1, nP2);
    results.nrmse_test_smol  = NaN(nP1, nP2);
end
%--------------------------------------------------------------------------
function plot_sweep_results(results, config)
    [X, Y] = meshgrid(config.param1_values, config.param2_values);
    plot_heatmap(X, Y, results.nrmse_test_num', 'NRMSE (Test) - Numerical Model', config);
    plot_heatmap(X, Y, results.nrmse_test_smol', 'NRMSE (Test) - Stochastic Model', config);
end
%--------------------------------------------------------------------------
function plot_heatmap(X, Y, Z, title_str, config)
    figure('Name', title_str);
    contourf(X, Y, Z, 20, 'LineColor', 'none');
    shading interp;
    cb = colorbar;
    colormap(flipud(parula));
    ylabel(cb, 'NRMSE');
    if max(config.param1_values)/min(config.param1_values) > 50, set(gca, 'XScale', 'log'); end
    if max(config.param2_values)/min(config.param2_values) > 50, set(gca, 'YScale', 'log'); end
    xlabel(strrep(config.param1_name, '_', ' '));
    ylabel(strrep(config.param2_name, '_', ' '));
    title(title_str);
end