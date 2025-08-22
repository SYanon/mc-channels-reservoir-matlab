% =========================================================================
% narma_numerical_analysis
%
% DESCRIPTION:
%   A modular and refactored script to perform a two-parameter sweep for
%   the NARMA10 reservoir computing benchmark using the deterministic model.
%
% =========================================================================

%% 0. Initialize Environment & Get Configuration
close all;
clear;
rng(0); % Set random seed for reproducibility
fprintf('Starting modular NARMA10 parameter sweep...\n');

% --- Central Configuration Struct ---
% All parameters are defined in one place for easy management.
config = get_default_config();

%% 1. Generate Data (Done ONCE)
% NARMA10 series is generated programmatically outside the main loop.
fprintf('Generating NARMA10 data...\n');
narma_data = generate_narma10_data(config);
fprintf('Data generation complete.\n');

%% 2. Prepare for Sweep
% Pre-allocate result matrices for storing sweep outcomes.
nP1 = length(config.param1_values);
nP2 = length(config.param2_values);
results = initialize_results_struct(nP1, nP2);

% Get handles to the parameters we are sweeping for clarity.
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
        run_config = config; % Start with default config
        run_config.(p2_name) = p2_vals(i2);
        run_config.(p1_name) = p1_vals(i1);
        
        current_run = (i2-1)*nP1 + i1;
        fprintf('Run (%d/%d): %s=%g, %s=%g\n', ...
                current_run, total_runs, p1_name, run_config.(p1_name), p2_name, run_config.(p2_name));

        %% 4. Run Core Simulation & Reservoir Sampling
        % This function encapsulates the physics and state sampling.
        [sim, reservoir_states] = run_channel_simulation(narma_data.u, run_config);
        
        %% 5. Train and Test Readout
        % These functions perform the machine learning steps.
        W_out = train_readout(reservoir_states, narma_data.train_target, run_config);
        [nrmse_train, nrmse_test, y_test_hat] = test_readout(W_out, reservoir_states, narma_data, run_config);

        %% 6. Calculate Timescales
        % This function calculates all the correlation-based metrics.
        timescales = calculate_timescales(sim, reservoir_states, run_config);

        %% 7. Store Results
        results.nrmse_train(i1, i2) = nrmse_train;
        results.nrmse_test(i1, i2)  = nrmse_test;
        results.mean_occupation(i1, i2) = timescales.mean_occupation;
        results.tau_char_theory(i1, i2) = timescales.tau_char_theory;
        results.tau_n_values(i1, i2)    = timescales.tau_n_values;
        results.tau_c_values(i1, i2)    = timescales.tau_c_values;
        results.tau_rc_samples(i1, i2)  = timescales.tau_rc_samples;
        
        fprintf('  -> NRMSE(Test)=%.3g. (took %.2fs)\n', nrmse_test, toc);

        %% Optional: Plotting inside the loop
        if run_config.plot_in_loop
            plot_interim_results(sim, y_test_hat, narma_data.test_target, current_run, total_runs);
        end
    end
end

%% 8. Visualize Final Results
fprintf('Sweep complete. Generating summary plots...\n');
plot_sweep_results(results, config);
plot_narma_autocorrelation(narma_data.q);
fprintf('All done.\n');


%==========================================================================
%                    --- LOCAL HELPER FUNCTIONS ---
%==========================================================================

function config = get_default_config()
    % Central location for all default parameters. Returns a config struct.
    
    % --- Sweep Parameters ---
    config.param1_name   = 'k_on';
    config.param1_values = [5e-20, 1e-19, 2e-19, 5e-19, 1e-18, 2e-18, 5e-18, 1e-17, 2e-17];
    config.param2_name   = 'k_off';
    config.param2_values = [0.1, 0.2, 0.5, 1, 2, 5, 10];
    
    % --- Fixed System Parameters ---
    config.N     = 500;      % Total number of receptors
    config.k_on  = 1e-18;    % (1/(M*s)) Binding rate constant
    config.k_off = 1;        % (1/s) Unbinding rate constant
    config.N_min = 100;      % Min # of molecules for normalization
    config.N_max = 3000;     % Max # of molecules for normalization
    config.distance = 10e-6; % (m) Transmitter-receiver distance
    config.D        = 1e-11; % (m^2/s) Diffusion coefficient

    % --- Reservoir & Readout ---
    config.lambda = 1e-10;   % Regularization for ridge regression
    config.Nres_init = 100;  % Base number of reservoir nodes
    config.memorywindowlength = 1;
    config.Nres = config.Nres_init * config.memorywindowlength;
    
    % --- Time & Data Lengths ---
    config.T  = 1;           % (s) Symbol duration
    config.washout1         = 300;
    config.num_train_points = 2000;
    config.wheretostarttest = 2500;
    config.num_test_points  = 2000;
    
    % Derived data lengths
    config.washout2 = config.wheretostarttest - (config.washout1 + config.num_train_points);
    config.num_tot_points = config.washout1 + config.washout2 + config.num_train_points + config.num_test_points;

    % --- Simulation Control ---
    config.dt = 0.001;       % (s) Numerical solution time step
    config.memlengthsweep = 100; % Factor for calculating channel memory
    config.offset = 0;       % Time offset for sampling
    config.plot_in_loop = false; % Flag to show plots during the sweep
end

%--------------------------------------------------------------------------

function narma_data = generate_narma10_data(config)
    % Generates the NARMA10 time series and prepares data splits.
    narma_len = config.num_tot_points + 10 + 1; % +10 warmup, +1 for next-step prediction
    u = 0.5 * rand(narma_len, 1);
    q = zeros(narma_len, 1);
    
    for n = 10:(config.num_tot_points - 1)
        q(n+1) = 0.3*q(n) + 0.05*q(n)*sum(q(n-9:n)) + 1.5*u(n-9)*u(n) + 0.1;
    end
    
    narma_data.u = u; % Driving input
    narma_data.q = q; % Target output
    
    % Create training and testing splits for the target series
    train_indices = (config.washout1+2) : (config.washout1+config.num_train_points+1);
    test_indices = (config.washout1+config.num_train_points+config.washout2+2) : (config.num_tot_points+1);
    
    narma_data.train_target = q(train_indices);
    narma_data.test_target = q(test_indices);
end

%--------------------------------------------------------------------------

function [sim, reservoir_states] = run_channel_simulation(u_series, config)
    % Simulates the MC channel response to an input series `u_series`.
    
    % Map input signal to number of molecules
    N_i = config.N_min + (u_series - min(u_series)) / (max(u_series) - min(u_series)) * (config.N_max - config.N_min);

    % Setup time vectors
    t_total = length(N_i) * config.T;
    sim.time = 0:config.dt:t_total;
    
    % Calculate Concentration c(t)
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

    % Solve Receptor Binding ODE n(t)
    sim.occupation = zeros(size(sim.time));
    for idxT = 2:length(sim.time)
        c_t = sim.concentration(idxT - 1);
        n_t = sim.occupation(idxT - 1);
        dn_dt = config.k_on*(config.N - n_t*config.N)*c_t - config.k_off*n_t*config.N;
        sim.occupation(idxT) = n_t + (dn_dt/config.N)*config.dt;
    end
    
    % Sample Reservoir States
    steps_per_symbol = config.T / config.dt;
    num_symbols = config.num_tot_points;
    reservoir_states = zeros(config.Nres, num_symbols);
    
    for iSym = 1:num_symbols
        t_symbol_start = (iSym - 1) * config.T + config.offset;
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

function W_out = train_readout(reservoir_states, train_target, config)
    % Trains the linear readout layer using ridge regression.
    train_indices = config.washout1+1 : config.washout1+config.num_train_points;
    X_train_raw = reservoir_states(:, train_indices);
    
    % Remove columns that were not properly sampled (all zeros)
    valid_cols = any(X_train_raw, 1);
    X_train_raw = X_train_raw(:, valid_cols);
    y_train = train_target(valid_cols);

    X_train = [X_train_raw; ones(1, size(X_train_raw, 2))];
    
    W_out = pinv(X_train * X_train' + config.lambda * eye(size(X_train, 1))) * (X_train * y_train(:));
end

%--------------------------------------------------------------------------

function [nrmse_train, nrmse_test, y_test_hat] = test_readout(W_out, reservoir_states, narma_data, config)
    % Uses trained weights to predict and calculates NRMSE.
    
    % --- Test on Training Data ---
    train_indices = config.washout1+1 : config.washout1+config.num_train_points;
    X_train_raw = reservoir_states(:, train_indices);
    valid_cols_train = any(X_train_raw, 1);
    X_train = [X_train_raw(:, valid_cols_train); ones(1, sum(valid_cols_train))];
    y_train_hat = X_train' * W_out;
    y_train_true = narma_data.train_target(valid_cols_train);
    nrmse_train = sqrt(mean((y_train_hat - y_train_true(:)).^2)) / std(y_train_true);

    % --- Test on Testing Data ---
    test_indices = config.washout1+config.num_train_points+config.washout2+1 : config.num_tot_points;
    X_test_raw = reservoir_states(:, test_indices);
    valid_cols_test = any(X_test_raw, 1);
    X_test = [X_test_raw(:, valid_cols_test); ones(1, sum(valid_cols_test))];
    y_test_hat = X_test' * W_out;
    y_test_true = narma_data.test_target(valid_cols_test);
    nrmse_test = sqrt(mean((y_test_hat - y_test_true(:)).^2)) / std(y_test_true);
end

%--------------------------------------------------------------------------

function timescales = calculate_timescales(sim, reservoir_states, config)
    % Calculates various theoretical and empirical timescales.
    half_idx = round(length(sim.time) / 2);
    
    mean_c = mean(sim.concentration(half_idx:end));
    timescales.tau_char_theory = 1 / (config.k_on * mean_c + config.k_off);
    timescales.mean_occupation = mean(sim.occupation(half_idx:end));
    
    timescales.tau_n_values = get_autocorr_time(sim.occupation(half_idx:end), config.dt);
    timescales.tau_c_values = get_autocorr_time(sim.concentration(half_idx:end), config.dt);
    
    dtsample = (config.T / config.Nres) * config.memorywindowlength;
    timescales.tau_rc_samples = get_autocorr_time(reshape(reservoir_states, 1, []), dtsample);
end

%--------------------------------------------------------------------------

function tau = get_autocorr_time(signal, dt)
    % Helper to compute the 1/e autocorrelation time.
    if isempty(signal) || std(signal) < 1e-9, tau = NaN; return; end
    centered = signal - mean(signal);
    [acf, lags] = xcorr(centered, 'coeff');
    pos_lags = lags >= 0;
    idx_e = find(acf(pos_lags) < exp(-1), 1, 'first');
    if isempty(idx_e), tau = lags(end) * dt; else, tau = lags(find(pos_lags,1)+idx_e-1) * dt; end
end

%--------------------------------------------------------------------------

function results = initialize_results_struct(nP1, nP2)
    % Creates a struct to hold all results, pre-allocating with NaNs.
    results.nrmse_train     = NaN(nP1, nP2);
    results.nrmse_test      = NaN(nP1, nP2);
    results.mean_occupation = NaN(nP1, nP2);
    results.tau_char_theory = NaN(nP1, nP2);
    results.tau_n_values    = NaN(nP1, nP2);
    results.tau_c_values    = NaN(nP1, nP2);
    results.tau_rc_samples  = NaN(nP1, nP2);
end

%--------------------------------------------------------------------------

function plot_sweep_results(results, config)
    % Generates all 2D heatmap plots from the final results.
    [X, Y] = meshgrid(config.param1_values, config.param2_values);
    
    plot_heatmap(X, Y, results.nrmse_train', 'NRMSE (Train)', config);
    plot_heatmap(X, Y, results.nrmse_test', 'NRMSE (Test)', config);
    plot_heatmap(X, Y, results.mean_occupation', 'Mean Receptor Occupation', config);
    plot_heatmap(X, Y, log10(results.tau_char_theory'), 'Log10(Theoretical Timescale)', config, true);
    plot_heatmap(X, Y, results.tau_n_values', 'Timescale from n-values', config);
    plot_heatmap(X, Y, results.tau_c_values', 'Timescale from c-values', config);
end

%--------------------------------------------------------------------------

function plot_heatmap(X, Y, Z, title_str, config, use_log_z)
    if nargin < 6, use_log_z = false; end

    figure('Name', title_str);
    contourf(X, Y, Z, 20, 'LineColor', 'none');
    shading interp;
    cb = colorbar;
    
    if use_log_z
        colormap(jet); ylabel(cb, 'Log10(Value)');
    else
        colormap(flipud(parula)); ylabel(cb, 'Value');
    end
    
    if max(config.param1_values)/min(config.param1_values) > 50, set(gca, 'XScale', 'log'); end
    if max(config.param2_values)/min(config.param2_values) > 50, set(gca, 'YScale', 'log'); end
    
    xlabel(strrep(config.param1_name, '_', ' '));
    ylabel(strrep(config.param2_name, '_', ' '));
    title(title_str);
end

%--------------------------------------------------------------------------

function plot_interim_results(sim, y_test_hat, y_test_true, run_num, total_runs)
    % Plots key results from a single run, useful for debugging.
    fig_title = sprintf('Interim Results (Run %d/%d)', run_num, total_runs);
    figure('Name', fig_title, 'NumberTitle', 'off');
    
    % Plot c(t) and n(t)
    subplot(2,1,1);
    plot(sim.time, sim.concentration, 'b-', 'DisplayName', 'c(t)');
    hold on;
    plot(sim.time, sim.occupation, 'r-', 'DisplayName', 'n(t)');
    grid on; title('Channel Dynamics'); xlabel('Time (s)'); legend;

    % Plot predicted vs true test data
    subplot(2,1,2);
    plot(y_test_true, 'b-o', 'LineWidth', 1.5, 'DisplayName', 'True Target');
    hold on;
    plot(y_test_hat, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Prediction');
    grid on; title('Test Set Prediction'); xlabel('Sample Index'); ylabel('Value'); legend;
end

%--------------------------------------------------------------------------
function plot_narma_autocorrelation(q_series)
    % Plots the autocorrelation of the generated NARMA10 series.
    figure('Name', 'NARMA10 Autocorrelation');
    maxLag = 200;
    centered_q = q_series - mean(q_series);
    [acf, lags] = xcorr(centered_q, maxLag, 'coeff');
    pos_lags = lags >= 0;
    
    plot(lags(pos_lags), acf(pos_lags), 'LineWidth', 2);
    grid on;
    xlabel('Lag (samples)');
    ylabel('Autocorrelation');
    title('NARMA10 Target Series Autocorrelation');
end