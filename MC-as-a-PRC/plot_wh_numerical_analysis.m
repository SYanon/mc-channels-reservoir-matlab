% =========================================================================
% plot_wh_numerical_analysis.m
%
% DESCRIPTION:
%   Performs a two-parameter sweep for the Wiener-Hammerstein
%   nonlinear transformation task and plots intermediate results.
%   This is a complete, runnable, and updated script.
% =========================================================================

%% 0. Initialize Environment & Define Configuration
close all;
clear;
fprintf('Starting Wiener-Hammerstein transformation task parameter sweep...\n');

config = get_default_config();

%% 1. Load and Prepare Data
try
    task_data = load_wh_data(config);
    fprintf('Wiener-Hammerstein data loaded successfully.\n');
catch ME
    fprintf('Error loading data file. Please run createWHseries.m first.\n');
    rethrow(ME);
end

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
        
        run_config = config;
        run_config.(p2_name) = p2_vals(i2);
        run_config.(p1_name) = p1_vals(i1);

        fprintf('Run (%d/%d): %s=%g, %s=%g\n', ...
                (i2-1)*nP1 + i1, nP1*nP2, p1_name, run_config.(p1_name), p2_name, run_config.(p2_name));
        
        % Run the physical reservoir simulation
        [~, reservoir_states] = run_channel_simulation(task_data.input_series, run_config);
        
        % Train and test using the virtual nodes
            %[nrmse_train, nrmse_test, y_test_hat] = test_readout(W_out, reservoir_states, task_data.target, run_config);

        W_out = train_readout(reservoir_states, task_data.train_target, run_config);
        [nrmse_train, nrmse_test, y_test_hat] = test_readout(W_out, reservoir_states, task_data, run_config);
        
        results.nrmse_train(i1, i2) = nrmse_train;
        results.nrmse_test(i1, i2) = nrmse_test;
        
        fprintf('  %s=%.2e, %s=%.2e -> NRMSE=%.4f (%.2f s)\n', p1_name, p1_vals(i1), p2_name, p2_vals(i2), nrmse_test, toc);
        fprintf('  -> NRMSE(Test)=%.3g. (took %.2fs)\n', nrmse_test, toc);

        % In-loop plotting
        plot_intermediate_results(y_test_hat, task_data, run_config, nrmse_test);
    end
end

fprintf('Sweep finished.\n');


%% 4. Final Plotting
plot_sweep_results(results, config);


% =========================================================================
%                       HELPER FUNCTIONS
% =========================================================================

function config = get_default_config()
    % Central location for all default parameters.
    config.param1_name   = 'k_on';
    config.param1_values = [5e-20, 2e-19,  1e-18, 2e-18,  1e-17, 2e-17];
    config.param2_name   = 'k_off';
    config.param2_values = [0.1, 0.5, 1,  5];   

    % --- System Parameters ---
    config.N     = 500;
    config.k_on  = 1e-18;
    config.k_off = 1;
    config.N_min = 100; %10 in gemini
    config.N_max = 3000; %500 in gemini
    config.lambda = 1e-6;
    config.Nres   = 100; % Increased nodes, often better for transformation
    config.memorywindowlength = 1;
    config.distance = 10e-6;
    config.D        = 1e-11; %1e-10 i gemini code
    config.T  = 1;
    
    % --- Data Lengths ---
    config.washout1         = 500;
    config.num_train_points = 2000; % Using more data for training
    config.wheretostarttest = 3000;
    config.num_test_points  = 1000;
    config.washout2 = config.wheretostarttest - (config.washout1 + config.num_train_points);
    config.num_tot_points = config.washout1 + config.washout2 + config.num_train_points + config.num_test_points;
            
    % % Default Reservoir physical parameters
    % config.r = 5e-6;
    % config.N_T = 1000;

    % --- Simulation Control ---
    config.dt = 0.001;
    config.memlengthsweep = 100;

    % --- Data Source ---
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
    train_states = reservoir_states(:, config.washout1+1 : config.washout1+config.num_train_points);
    X_train = [train_states; ones(1, config.num_train_points)];
    y_train = train_target(:);
    W_out = pinv(X_train * X_train' + config.lambda * eye(size(X_train, 1))) * (X_train * y_train);
end

function [nrmse_train, nrmse_test, y_test_hat] = test_readout(W_out, reservoir_states, task_data, config)
    train_states = reservoir_states(:, config.washout1+1 : config.washout1+config.num_train_points);
    X_train = [train_states; ones(1, config.num_train_points)];
    y_train_hat = X_train' * W_out;
    nrmse_train = sqrt(mean((y_train_hat - task_data.train_target(:)).^2)) / std(task_data.train_target);
    test_indices = config.wheretostarttest+1 : config.wheretostarttest + config.num_test_points;
    test_states = reservoir_states(:, test_indices);
    X_test = [test_states; ones(1, config.num_test_points)];
    y_test_hat = X_test' * W_out;
    nrmse_test = sqrt(mean((y_test_hat - task_data.test_target(:)).^2)) / std(task_data.test_target);
end

function results = initialize_results_struct(nP1, nP2)
    results.nrmse_train     = NaN(nP1, nP2);
    results.nrmse_test      = NaN(nP1, nP2);
end

function plot_intermediate_results(y_test_hat, task_data, config, nrmse_test)
    % Plots the performance for a single run within the sweep.
    figure(1); clf;

    y_test_true = task_data.test_target(:);

    plot(y_test_true, 'b-', 'LineWidth', 1.5); hold on;
    plot(y_test_hat,  'r--', 'LineWidth', 1.5);
    title(sprintf('%s=%.2e, %s=%.2e, NRMSE=%.4f', ...
        config.param1_name, config.(config.param1_name), ...
        config.param2_name, config.(config.param2_name), nrmse_test));
    legend('Target', 'Reservoir Output', 'Location', 'southeast');
    grid on; ylim([-0.1 1.1]); drawnow;
end

% function plot_sweep_results(results, config)
%     % Plots the final heatmap of the parameter sweep results.
%     figure('Name', 'Parameter Sweep Heatmap');
%     [Y, X] = meshgrid(config.param2_values, config.param1_values);
% 
%     surf(X, Y, results.nrmse_test);
%     shading interp;
%     xlabel(config.param1_name);
%     ylabel(config.param2_name);
%     zlabel('NRMSE');
%     title('NRMSE vs. Parameters');
%     set(gca, 'XScale', 'log');
%     set(gca, 'YScale', 'log');
%     colorbar;
%     view(2); % Top-down view
% end

function plot_sweep_results(results, config)
    [X, Y] = meshgrid(config.param1_values, config.param2_values);
    plot_heatmap(X, Y, results.nrmse_test', 'NRMSE (Test)', config);
end

function plot_heatmap(X, Y, Z, title_str, config)
    figure('Name', title_str);
    contourf(X, Y, Z, 20, 'LineColor', 'none');
    shading interp;
    cb = colorbar;
    colormap(flipud(parula));
    ylabel(cb, title_str);
    set(gca, 'XScale', 'log');
    set(gca, 'YScale', 'log');
    xlabel(strrep(config.param1_name, '_', ' '));
    ylabel(strrep(config.param2_name, '_', ' '));
    title(title_str);
end