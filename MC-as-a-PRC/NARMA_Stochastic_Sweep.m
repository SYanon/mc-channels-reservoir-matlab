% =========================================================================
% narma_stochastic_sweep.m
%
% DESCRIPTION:
%   A modular script to perform a two-parameter sweep comparing the
%   deterministic and stochastic (Smoldyn) models for the NARMA10 benchmark.
%   Sweeps over k_on vs. k_off.
% =========================================================================

%% 0. Initialize Environment & Define Configuration
close all;
clear;
rng(0); % Set random seed for reproducibility
fprintf('Starting MODULAR Stochastic NARMA10 parameter sweep...\n');

% --- Central Configuration Struct ---
config = get_default_config();

%% 1. Generate Data (Done ONCE)
% Generate the NARMA10 series and prepare all necessary data splits.
fprintf('Generating NARMA10 data series...\n');
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
        
        current_run = (i2-1)*nP1 + i1;
        
        % =================== NUMERICAL (DETERMINISTIC) MODEL ===================
        [sim, reservoir_states_num] = run_channel_simulation(narma_data.u, run_config);
        W_out_num = train_readout(reservoir_states_num, narma_data.train_target, run_config);
        [~, nrmse_test_num, y_test_hat_num] = test_readout(W_out_num, reservoir_states_num, narma_data, run_config);
        
        % =================== STOCHASTIC (SMOLDYN) MODEL ===================
        [reservoir_states_smol_raw, success] = load_smoldyn_data(run_config);
        
        if success
            reservoir_states_smol_filtered = apply_signal_filter(reservoir_states_smol_raw, run_config);
            W_out_smol = train_readout(reservoir_states_smol_filtered, narma_data.train_target, run_config);
            [~, nrmse_test_smol, y_test_hat_smol] = test_readout(W_out_smol, reservoir_states_smol_filtered, narma_data, run_config);
        else
            fprintf('    -> WARNING: Smoldyn data not found for Run (%d/%d). Skipping.\n', current_run, total_runs);
            nrmse_test_smol  = NaN;
        end

        %% 7. Store Results
        results.nrmse_test_num(i1, i2)  = nrmse_test_num;
        results.nrmse_test_smol(i1, i2)  = nrmse_test_smol;

        fprintf('Run (%d/%d): %s=%g, %s=%g -> NRMSE(Num)=%.3g, NRMSE(Smol)=%.3g. (took %.2fs)\n', ...
                current_run, total_runs, p1_name, run_config.(p1_name), p2_name, run_config.(p2_name), ...
                nrmse_test_num, nrmse_test_smol, toc);
                
        %% Optional: Plotting inside the loop
        if run_config.plot_in_loop
            run_info.current_run = current_run;
            run_info.total_runs = total_runs;
            run_info.p1_name = p1_name; run_info.p1_val = run_config.(p1_name);
            run_info.p2_name = p2_name; run_info.p2_val = run_config.(p2_name);
            run_info.smol_dt = run_config.smol_dt;
            
            num_results.y_test_hat = y_test_hat_num;
            smol_results.states_raw = reservoir_states_smol_raw;
            if success, smol_results.y_test_hat = y_test_hat_smol; else, smol_results.y_test_hat = []; end

            plot_interim_results(sim, narma_data, num_results, smol_results, run_info);
        end
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
    
    % --- MODIFIED: Swept Parameters ---
    config.param1_name   = 'k_on';
    config.param1_values = [5e-20, 1e-19, 2e-19, 5e-19, 1e-18, 2e-18, 5e-18, 1e-17, 2e-17];
    config.param2_name   = 'k_off';
    config.param2_values = [0.1, 0.2, 0.5, 1, 2, 5, 10];
    
    % --- Stochastic Model Settings ---
    config.smoldyn_data_dir = 'NARMA_5000_T1_koff1'; 
    config.movmean_on = true;
    config.movmean_window = 2000; % Fixed config option
    config.lambda_smol = 1.0; 

    % --- Reservoir & Readout ---
    config.Nres_init = 100; % Fixed config option
    config.memorywindowlength = 1;
    config.Nres = config.Nres_init * config.memorywindowlength;
    config.lambda = 1e-10; 

    % --- Fixed System Parameters ---
    config.N     = 500;
    config.k_on  = 1e-18; % Default value, will be overwritten by sweep
    config.k_off = 1;     % Default value, will be overwritten by sweep
    config.N_min = 100;
    config.N_max = 3000;
    config.distance = 10e-6;
    config.D        = 1e-11;
    config.T  = 1;
    
    % --- Data Lengths ---
    config.washout1         = 250;
    config.num_train_points = 2000;
    config.wheretostarttest = 2500;
    config.num_test_points  = 2000;
    config.washout2 = config.wheretostarttest - (config.washout1 + config.num_train_points);
    config.num_tot_points = config.washout1 + config.washout2 + config.num_train_points + config.num_test_points;
    
    % --- Simulation Control ---
    config.dt = 0.001;
    config.smol_dt = 0.01;
    config.memlengthsweep = 100;
    config.plot_in_loop = false;
end

%--------------------------------------------------------------------------
function narma_data = generate_narma10_data(config)
    % Generates the NARMA10 time series and prepares all necessary data splits.
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
    % This is the deterministic (numerical) model simulation.
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
function [reservoir_states, success] = load_smoldyn_data(config)
    % Loads, processes, and samples the pre-generated Smoldyn data.
    filename = 'allmolecules_varNo_2_varValue_1_iter_1.txt';
    filepath = fullfile(config.smoldyn_data_dir, filename);
    if ~exist(filepath, 'file'), reservoir_states = []; success = false; return; end
    
    try
        data_temp = importdata(filepath, ' ', 1);
        smol_data = data_temp.data;
        time_vec = smol_data(:, 1);
        active_receptors = smol_data(:, end) / config.N; % Normalize
        
        steps_per_symbol = config.T / config.smol_dt;
        num_symbols = config.num_tot_points;
        reservoir_states = zeros(config.Nres, num_symbols);
        
        for iSym = 1:num_symbols
            t_symbol_start = (iSym - 1) * config.T;
            start_idx = find(time_vec >= t_symbol_start, 1);
            if isempty(start_idx), continue; end
            
            sample_indices_float = start_idx - (config.memorywindowlength-1)*steps_per_symbol + (0:(config.Nres-1))*(steps_per_symbol/config.Nres)*config.memorywindowlength;
            sample_indices = round(sample_indices_float);
            
            if all(sample_indices > 0 & sample_indices <= length(active_receptors))
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
    % Applies a moving average filter to the reservoir states if enabled.
    if config.movmean_on && config.movmean_window > 1
        k = config.movmean_window;
        filtered_states = movmean(reservoir_states, [k-1 0], 2);
    else
        filtered_states = reservoir_states; % No filtering
    end
end

%--------------------------------------------------------------------------
function W_out = train_readout(reservoir_states, train_target, config)
    % Generic training function for both numerical and stochastic models.
    train_states = reservoir_states(:, config.washout1+1 : config.washout1+config.num_train_points);
    
    valid_cols = any(train_states, 1);
    train_states = train_states(:, valid_cols);
    y_train = train_target(valid_cols);
    
    X_train = [train_states; ones(1, size(train_states, 2))];
    
    current_lambda = config.lambda; 
    if isfield(config, 'lambda_smol'), current_lambda = config.lambda_smol; end
    
    W_out = pinv(X_train * X_train' + current_lambda * eye(size(X_train, 1))) * (X_train * y_train(:));
end

%--------------------------------------------------------------------------
function [nrmse_train, nrmse_test, y_test_hat] = test_readout(W_out, reservoir_states, narma_data, config)
    % Generic testing function for both models.
    train_indices = config.washout1+1 : config.washout1+config.num_train_points;
    train_states = reservoir_states(:, train_indices);
    valid_cols_train = any(train_states, 1);
    X_train = [train_states(:, valid_cols_train); ones(1, sum(valid_cols_train))];
    y_train_hat = X_train' * W_out;
    nrmse_train = sqrt(mean((y_train_hat - narma_data.train_target(valid_cols_train)).^2)) / std(narma_data.train_target(valid_cols_train));
    
    test_indices = config.wheretostarttest+1 : config.wheretostarttest + config.num_test_points;
    test_states = reservoir_states(:, test_indices);
    valid_cols_test = any(test_states, 1);
    X_test = [test_states(:, valid_cols_test); ones(1, sum(valid_cols_test))];
    y_test_hat = X_test' * W_out;
    y_test_true = narma_data.test_target(valid_cols_test);
    nrmse_test = sqrt(mean((y_test_hat - y_test_true).^2)) / std(y_test_true);
end

%--------------------------------------------------------------------------
function results = initialize_results_struct(nP1, nP2)
    % Creates a struct to hold all results, pre-allocating with NaNs.
    results.nrmse_test_num   = NaN(nP1, nP2);
    results.nrmse_test_smol  = NaN(nP1, nP2);
end

%--------------------------------------------------------------------------
function plot_sweep_results(results, config)
    % Generates all 2D heatmap plots from the final results.
    [X, Y] = meshgrid(config.param1_values, config.param2_values);
    plot_heatmap(X, Y, results.nrmse_test_num', 'NRMSE (Test) - Numerical Model', config);
    plot_heatmap(X, Y, results.nrmse_test_smol', 'NRMSE (Test) - Stochastic (Smoldyn) Model', config);
end

%--------------------------------------------------------------------------
function plot_heatmap(X, Y, Z, title_str, config)
    % Generic function to create a heatmap.
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
    title(title_str);
end

%--------------------------------------------------------------------------
function plot_interim_results(sim, narma_data, num_results, smol_results, run_info)
    % Plots key results from a single run for debugging.
    fig_title = sprintf('Interim Results (Run %d/%d): %s=%g, %s=%g', ...
        run_info.current_run, run_info.total_runs, ...
        run_info.p1_name, run_info.p1_val, ...
        run_info.p2_name, run_info.p2_val);
    
    figure('Name', fig_title, 'NumberTitle', 'off');
    
    % Plot 1: Compare Numerical and Stochastic Reservoir States
    subplot(2,1,1);
    plot(sim.time, sim.occupation, 'b-', 'DisplayName', 'Numerical n(t)');
    hold on;
    if ~isempty(smol_results.states_raw)
        smol_time = 0:run_info.smol_dt:(size(smol_results.states_raw, 2)-1)*run_info.smol_dt;
        plot(smol_time, smol_results.states_raw(1,:), 'r-', 'LineWidth', 0.5, 'DisplayName', 'Stochastic n(t) (Node 1)');
    end
    grid on; title('Reservoir Dynamics'); xlabel('Time (s)'); legend;

    % Plot 2: Compare Final Predictions
    subplot(2,1,2);
    test_target_valid = narma_data.test_target(1:length(num_results.y_test_hat));
    plot(test_target_valid, 'k-', 'LineWidth', 2, 'DisplayName', 'True Target');
    hold on;
    plot(num_results.y_test_hat, 'b--', 'LineWidth', 1.5, 'DisplayName', 'Numerical Prediction');
    if ~isempty(smol_results.y_test_hat)
        plot(smol_results.y_test_hat, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Stochastic Prediction');
    end
    grid on; title('Test Set Predictions'); xlabel('Sample Index'); ylabel('Value'); legend;
end