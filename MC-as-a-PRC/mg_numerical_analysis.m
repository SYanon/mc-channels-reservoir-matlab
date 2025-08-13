% =========================================================================
% mg_numerical_analysis
%
% DESCRIPTION:
%   A modular and refactored script to perform a two-parameter sweep for
%   the Mackey-Glass reservoir computing benchmark.
%
% =========================================================================

%% 0. Initialize Environment & Define Configuration
close all;
clear;
fprintf('Starting modular Mackey-Glass parameter sweep...\n');

% --- Central Configuration Struct ---
% All parameters are defined in one place for easy management.
config = get_default_config();

%% 1. Load and Prepare Data (Done ONCE)
% Data is loaded outside the main loop to avoid redundant disk access.
mg_data = load_mg_data(config.MG_FILENAME, config.num_tot_points, config.predictlength);

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
fprintf('Starting sweep over %s and %s...\n', p1_name, p2_name);
for i2 = 1:nP2
    for i1 = 1:nP1
        tic;
        
        % --- Update parameters for the current run ---
        run_config = config; % Start with default config
        run_config.(p2_name) = p2_vals(i2);
        run_config.(p1_name) = p1_vals(i1);
        
        fprintf('Run (%d/%d): %s=%g, %s=%g\n', ...
                (i2-1)*nP1 + i1, nP1*nP2, p1_name, run_config.(p1_name), p2_name, run_config.(p2_name));

        %% 4. Run Core Simulation
        % This function encapsulates the physics of the channel.
        [sim, reservoir_states] = run_channel_simulation(mg_data.input_series, run_config);
        
        %% 5. Train and Test Readout
        % These functions perform the machine learning steps.
        W_out = train_readout(reservoir_states, mg_data.train_target, run_config);
        [nrmse_train, nrmse_test] = test_readout(W_out, reservoir_states, mg_data, run_config);

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
    end
end

%% 8. Visualize Results
fprintf('Sweep complete. Generating plots...\n');
plot_sweep_results(results, config);

fprintf('All done.\n');

%% 9. Extras: Autocorrelation plots for MG Series and c(t) vs time/n(t) vs time
% Add this block to the end of run_mg_sweep_modular.m

%% =========== MG Autocorrelation (Example Plot) ===========
fprintf('Plotting input series autocorrelation...\n');

% Use the data loaded at the beginning
full_series = mg_data.input_series;

maxLagMG = 200;
mg_centered = full_series - mean(full_series);
[acfMG, lagsMG] = xcorr(mg_centered, maxLagMG, 'coeff');
acfMG  = acfMG(lagsMG>=0);
lagsMG = lagsMG(lagsMG>=0);

idx1eMG = find(acfMG < exp(-1), 1, 'first');
if isempty(idx1eMG), idx1eMG = maxLagMG; end
tauMG_1e = lagsMG(idx1eMG);

figure('Name','MG Autocorrelation');
plot(lagsMG, acfMG, 'LineWidth',2); hold on; grid on;
plot(tauMG_1e, acfMG(idx1eMG), 'ro','MarkerSize',8,'LineWidth',2);
xlabel('Lag (samples)'); ylabel('Autocorrelation');
title('Mackey-Glass Autocorrelation');
legend('ACF','1/e crossing','Location','Best');

% Plot the results from the final simulation run
figure('Name', 'Simulation Trace Analysis');

subplot(2, 1, 1); % Top plot for concentration
plot(sim.time, sim.concentration, 'b');
grid on;
title('Ligand Concentration c(t)');
xlabel('Time (s)');
ylabel('Concentration');

subplot(2, 1, 2); % Bottom plot for occupation
plot(sim.time, sim.occupation, 'r');
grid on;
title('Receptor Occupation Fraction n(t)');
xlabel('Time (s)');
ylabel('Occupation Fraction');

%==========================================================================
%                    --- LOCAL HELPER FUNCTIONS ---
%==========================================================================

function config = get_default_config()
    % Central location for all default parameters. Returns a config struct.
    
    % --- Sweep Parameters ---
    % Also examples such as
    % param1_name   = 'k_on';      % e.g. 'k_on', 'N', 'T', 'k_off', ...
    % param1_values = [5e-20, 1e-19, 2e-19, 5e-19, 1e-18, 2e-18, 5e-18, 1e-17, 2e-17]; % optimal range for the paper (spans 0.1-0.9 occupation ratio). 

    % param2_name   = 'k_off'; 
    % param2_values = [0.1, 0.2, 0.5, 1, 2, 5, 10]; % optimal range for the paper (spans 0.1-0.9 occupation ratio).

    % param2_name   = 'T';
    % param2_values = [0.5, 0.8, 1, 1.5, 2];

    % param2_name   = 'distance';
    % param2_values = [1e-6, 2e-6, 5e-6, 10e-6, 20e-6, 50e-6];

    % param2_name   = 'memorywindowlength';
    % param2_values = [1, 2, 3, 4, 5];

    % param2_name   = 'N_max';
    % param2_values = [200, 500, 1000, 3000, 10000, 20000];

    % param2_name   = 'D';
    % param2_values = [0.5e-11, 1e-11, 2e-11, 5e-11, 1e-10, 2e-10];
    config.param1_name   = 'k_on';
    config.param1_values = [5e-20, 1e-19, 2e-19, 5e-19, 1e-18, 2e-18, 5e-18, 1e-17, 2e-17];
    config.param2_name   = 'k_off';
    config.param2_values = [0.1, 0.2, 0.5, 1, 2, 5, 10];
    
    % --- Receptor & Binding ---
    config.N     = 500;      % Total number of receptors
    config.k_on  = 1e-18;    % (1/(M*s)) Binding rate constant
    config.k_off = 1;        % (1/s) Unbinding rate constant

    % --- Input Normalization ---
    config.N_min = 100;      % Min # of molecules after normalization
    config.N_max = 3000;     % Max # of molecules after normalization

    % --- Reservoir & Readout ---
    config.lambda = 1e-6;    % Regularization parameter for ridge regression
    config.Nres   = 50;      % Number of virtual reservoir nodes
    config.memorywindowlength = 1; % Multiplier for Nres (typically 1)

    % --- Communication Channel ---
    config.distance = 10e-6; % (m) Transmitter-receiver distance
    config.D        = 1e-11; % (m^2/s) Diffusion coefficient

    % --- Time & Data Lengths ---
    config.T  = 1;           % (s) Symbol duration
    config.predictlength = 6; % How many steps ahead to predict for MG
    config.washout1         = 500;
    config.num_train_points = 500;
    config.wheretostarttest = 1200;
    config.num_test_points  = 500;
    
    % Derived data lengths
    config.washout2 = config.wheretostarttest - (config.washout1 + config.num_train_points);
    config.num_tot_points = config.washout1 + config.washout2 + config.num_train_points + config.num_test_points;

    % --- Simulation Control ---
    config.dt = 0.001;       % (s) Numerical solution time step
    config.memlengthsweep = 100; % Factor for calculating channel memory
    
    % --- Data Source ---
    config.MG_FILENAME = 'MGseries_RK4_tau17_beta0.20_gamma0.1_n10_len5000_dt1.0.mat';
end

%--------------------------------------------------------------------------

function mg_data = load_mg_data(filename, required_len, predict_len)
    % Loads the Mackey-Glass series from a file and prepares all necessary
    % data splits (input, target, train, test).
    
    try
        loaded_data = load(filename, 'mackey_glass_series');
        full_series_raw = loaded_data.mackey_glass_series;
    catch
        error('Failed to load MG data from "%s". Please ensure the file exists.', filename);
    end
    
    if length(full_series_raw) < required_len + predict_len
        error('Loaded series is too short. Required length: %d.', required_len + predict_len);
    end
    
    % --- Normalize the required part of the series ---
    series_segment = full_series_raw(1 : required_len + predict_len);
    series_normalized = (series_segment - min(series_segment)) / (max(series_segment) - min(series_segment));
    
    % --- Create input and target signals ---
    % mg_data.input_series (1 x N double): The signal fed to the reservoir.
    % mg_data.target_series (1 x N double): The signal the reservoir must predict.
    mg_data.input_series  = series_normalized(1:end - predict_len);
    mg_data.target_series = series_normalized((predict_len+1):end);

    % --- Split into training and testing sets ---
    config = get_default_config(); % Use default lengths for splitting
    mg_data.train_target = mg_data.target_series(config.washout1+1 : config.washout1 + config.num_train_points);
    mg_data.test_target  = mg_data.target_series(config.washout1 + config.num_train_points + config.washout2 + 1 : end);
end

%--------------------------------------------------------------------------

function [sim, reservoir_states] = run_channel_simulation(input_series, config)
    % Core simulation of the molecular communication channel.
    
    % --- Map input signal to number of molecules ---
    inp_min = min(input_series);
    inp_max = max(input_series);
    N_i = config.N_min + (input_series - inp_min)/(inp_max - inp_min) * (config.N_max - config.N_min);

    % --- Setup time vectors for simulation ---
    t_total = length(N_i) * config.T;
    % sim.time (1 x M double): High-resolution time vector for the ODE solution.
    sim.time = 0:config.dt:t_total;
    
    % --- Calculate Concentration at Receiver ---
    % sim.concentration (1 x M double): Ligand concentration c(t) over time.
    sim.concentration = zeros(size(sim.time));
    Tpeak = config.distance^2/(6*config.D);
    memory_length = round((config.memlengthsweep * Tpeak)/config.T)*config.T;
    
    for iSym = 1:length(N_i)
        t_symbol_start = (iSym - 1) * config.T;
        t_memory_end   = t_symbol_start + memory_length;
        
        % PERFORMANCE NOTE: This 'find' can be slow. For uniform time steps,
        % indices can be calculated directly, which is much faster.
        idxRange = find(sim.time > t_symbol_start & sim.time <= t_memory_end);
        if isempty(idxRange), continue; end

        t_local = sim.time(idxRange) - t_symbol_start;
        
        % Vectorized calculation for the current symbol's contribution
        pulse = (N_i(iSym) ./ ((4*pi*config.D*t_local).^(3/2))) .* exp(-config.distance^2./(4*config.D*t_local));
        sim.concentration(idxRange) = sim.concentration(idxRange) + pulse;
    end

    % --- Solve Receptor Binding ODE (Euler method) ---
    % sim.occupation (1 x M double): Receptor occupation fraction n(t) over time.
    sim.occupation = zeros(size(sim.time));
    for idxT = 2:length(sim.time)
        c_t = sim.concentration(idxT - 1);
        n_t = sim.occupation(idxT - 1);
        dn_dt = config.k_on*(config.N - n_t*config.N)*c_t - config.k_off*n_t*config.N;
        sim.occupation(idxT) = sim.occupation(idxT-1) + (dn_dt/config.N)*config.dt;
    end
    
    % --- Sample Reservoir States ---
    % reservoir_states (Nres x N_symbols): The state matrix X used for learning.
    steps_per_symbol = config.T/config.dt;
    num_symbols = config.num_tot_points;
    reservoir_states = zeros(config.Nres, num_symbols);
    
    for iSym = 1:num_symbols
        t_symbol_start = (iSym - 1) * config.T;
        start_idx = round(t_symbol_start/config.dt) + 1;
        
        % Generate indices for sampling within the symbol window
        sample_indices_float = start_idx - (config.memorywindowlength-1)*steps_per_symbol + (0:(config.Nres-1))*(steps_per_symbol/config.Nres)*config.memorywindowlength;
        sample_indices = round(sample_indices_float);
        
        % Ensure indices are valid
        valid_indices = sample_indices > 0 & sample_indices <= length(sim.occupation);
        if all(valid_indices)
            reservoir_states(:, iSym) = sim.occupation(sample_indices);
        end
    end
end

%--------------------------------------------------------------------------

function W_out = train_readout(reservoir_states, train_target, config)
    % Trains the linear readout layer using ridge regression.
    
    % (Nres+1 x N_train) matrix, with bias term.
    train_states = reservoir_states(:, config.washout1+1 : config.washout1+config.num_train_points);
    X_train = [train_states; ones(1, config.num_train_points)];
    
    % (N_train x 1) vector.
    y_train = train_target(:);

    % W_out ((Nres+1) x 1): The trained weight vector.
    W_out = pinv(X_train * X_train' + config.lambda * eye(size(X_train, 1))) * (X_train * y_train);
end

%--------------------------------------------------------------------------

function [nrmse_train, nrmse_test] = test_readout(W_out, reservoir_states, mg_data, config)
    % Uses the trained weights to make predictions and calculates NRMSE.
    
    % --- Test on Training Data ---
    train_states = reservoir_states(:, config.washout1+1 : config.washout1+config.num_train_points);
    X_train = [train_states; ones(1, config.num_train_points)];
    y_train_hat = X_train' * W_out;
    nrmse_train = sqrt(mean((y_train_hat - mg_data.train_target(:)).^2)) / std(mg_data.train_target);
    
    % --- Test on Testing Data ---
    test_indices = config.washout1+config.num_train_points+config.washout2+1 : config.num_tot_points;
    test_states = reservoir_states(:, test_indices);
    X_test = [test_states; ones(1, config.num_test_points)];
    y_test_hat = X_test' * W_out;
    nrmse_test = sqrt(mean((y_test_hat - mg_data.test_target(:)).^2)) / std(mg_data.test_target);
end

%--------------------------------------------------------------------------

function timescales = calculate_timescales(sim, reservoir_states, config)
    % Calculates various theoretical and empirical timescales from simulation data.
    
    half_idx = round(length(sim.time) / 2);
    
    % --- Theoretical Timescale ---
    mean_c = mean(sim.concentration(half_idx:end));
    timescales.tau_char_theory = 1 / (config.k_on * mean_c + config.k_off);
    timescales.mean_occupation = mean(sim.occupation(half_idx:end));
    
    % --- Autocorrelation Timescales ---
    timescales.tau_n_values = get_autocorr_time(sim.occupation(half_idx:end), config.dt);
    timescales.tau_c_values = get_autocorr_time(sim.concentration(half_idx:end), config.dt);
    
    % --- Reservoir Sample Timescale ---
    % PERFORMANCE NOTE: The original code grew this matrix inside the main loop.
    % Here, we use the pre-computed reservoir_states matrix, which is far more efficient.
    res_sequence = reshape(reservoir_states, 1, []);
    dtsample = (config.T / config.Nres) * config.memorywindowlength;
    timescales.tau_rc_samples = get_autocorr_time(res_sequence, dtsample);
end

%--------------------------------------------------------------------------

function tau = get_autocorr_time(signal, dt)
    % Helper to compute the 1/e autocorrelation time for a given signal.
    if isempty(signal) || std(signal) == 0
        tau = NaN;
        return;
    end
    centered_signal = signal - mean(signal);
    [acf, lags] = xcorr(centered_signal, 'coeff');
    positive_lags = lags >= 0;
    
    % Find the first time the ACF drops below 1/e
    idx_e = find(acf(positive_lags) < exp(-1), 1, 'first');
    if isempty(idx_e)
        tau = (sum(positive_lags)-1) * dt; % Return max lag if it never drops
    else
        tau = lags(find(positive_lags,1,'first')+idx_e-1) * dt;
    end
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
    % Generates all the 2D heatmap plots from the final results.
    [X, Y] = meshgrid(config.param1_values, config.param2_values);
    
    plot_heatmap(X, Y, results.nrmse_test', 'NRMSE (Test)', config);
    plot_heatmap(X, Y, results.mean_occupation', 'Mean Receptor Occupation', config);
    plot_heatmap(X, Y, log10(results.tau_char_theory'), 'Log10(Theoretical Timescale)', config, true);
    % --- ADDED THESE LINES TO REPRODUCE THE OTHER PLOTS ---
    plot_heatmap(X, Y, results.nrmse_train', 'NRMSE (Train)', config);
    plot_heatmap(X, Y, results.tau_n_values', 'Timescale from n-values', config, true);
    plot_heatmap(X, Y, results.tau_c_values', 'Timescale from c-values', config, true);

end

%--------------------------------------------------------------------------

function plot_heatmap(X, Y, Z, title_str, config, use_log_z)
    if nargin < 6, use_log_z = false; end

    figure('Name', title_str);
    contourf(X, Y, Z, 20, 'LineColor', 'none');
    shading interp;
    cb = colorbar;
    
    if use_log_z
        colormap(jet);
        ylabel(cb, 'Log10(Timescale (s))');
    else
        colormap(flipud(parula));
        ylabel(cb, title_str);
    end
    
    if max(config.param1_values)/min(config.param1_values) > 50
        set(gca, 'XScale', 'log');
    end
    if max(config.param2_values)/min(config.param2_values) > 50
        set(gca, 'YScale', 'log');
    end
    
    xlabel(strrep(config.param1_name, '_', ' '));
    ylabel(strrep(config.param2_name, '_', ' '));
    title(title_str);
end