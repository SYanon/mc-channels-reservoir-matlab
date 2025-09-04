% =========================================================================
% plot_sine_numerical_analysis.m
%
% DESCRIPTION:
%   Performs a two-parameter sweep for the sine-to-square wave
%   nonlinear transformation task and plots intermediate results.
%   ADAPTED FROM: plot_mg_numerical_analysis.m
% =========================================================================

%% 0. Initialize Environment & Define Configuration
close all;
clear;
fprintf('Starting sine-to-square transformation task parameter sweep...\n');

config = get_default_config();

%% 1. Load and Prepare Data
% MODIFIED: Now loads the sine/square data.
task_data = load_sine_data(config.SINE_FILENAME, config.num_tot_points);

%% 2. Prepare for Sweep
nP1 = length(config.param1_values);
nP2 = length(config.param2_values);
results = initialize_results_struct(nP1, nP2); % Reusing this helper

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

        %% 4. Run Core Simulation (NO CHANGE NEEDED in the function itself)
        [~, reservoir_states] = run_channel_simulation(task_data.input_series, run_config);
        
        %% 5. Train and Test Readout (MODIFIED test_readout to output predictions)
        W_out = train_readout(reservoir_states, task_data.train_target, run_config);
        [nrmse_train, nrmse_test, y_test_hat] = test_readout(W_out, reservoir_states, task_data, run_config);

        %% 6. Store Results
        results.nrmse_train(i1, i2) = nrmse_train;
        results.nrmse_test(i1, i2)  = nrmse_test;
        
        fprintf('  -> NRMSE(Test)=%.3g. (took %.2fs)\n', nrmse_test, toc);

        %% 7. In-loop plotting block (As requested)
        if run_config.plot_in_loop
            figure('Name', sprintf('Sine-to-Square Prediction: %s=%.2e, %s=%.2f', p1_name, p1_vals(i1), p2_name, p2_vals(i2)));
            plot(task_data.test_target, 'r--', 'LineWidth', 2, 'DisplayName', 'True Target (Square)');
            hold on;
            plot(y_test_hat, 'k-', 'LineWidth', 1.5, 'DisplayName', 'Reservoir Prediction');
            hold off;
            grid on;
            title(sprintf('NRMSE: %.4f', nrmse_test));
            xlabel('Sample Index in Test Set');
            ylabel('Normalized Value');
            legend('Location', 'best');
            ylim([-0.1 1.1]);
            drawnow; % Forces the plot to render immediately
        end
    end
end

%% 8. Visualize Final Heatmap
fprintf('Sweep complete. Generating heatmap...\n');
plot_sweep_results(results, config);

fprintf('All done.\n');

%==========================================================================
%                    --- LOCAL HELPER FUNCTIONS ---
%==========================================================================

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
    config.N_min = 100;
    config.N_max = 3000;
    config.lambda = 1e-6;
    config.Nres   = 100; % Increased nodes, often better for transformation
    config.memorywindowlength = 1;
    config.distance = 10e-6;
    config.D        = 1e-11;
    config.T  = 1;
    
    % --- Data Lengths ---
    config.washout1         = 500;
    config.num_train_points = 2000; % Using more data for training
    config.wheretostarttest = 3000;
    config.num_test_points  = 1000;
    config.washout2 = config.wheretostarttest - (config.washout1 + config.num_train_points);
    config.num_tot_points = config.washout1 + config.washout2 + config.num_train_points + config.num_test_points;
    
    % --- Simulation Control ---
    config.dt = 0.001;
    config.memlengthsweep = 100;
    
    % --- Data Source ---
    config.SINE_FILENAME = 'SINEseries_len5000_p25.mat'; % MODIFIED
    
    % --- Plotting flag ---
    config.plot_in_loop = false; % Set to true to see plots for each run
end

%--------------------------------------------------------------------------

% NEW HELPER FUNCTION
function task_data = load_sine_data(filename, required_len)
    % Loads the sine/square series and prepares data splits.
    
    try
        loaded_data = load(filename, 'input_sine_series', 'target_square_series');
        input_series  = loaded_data.input_sine_series;
        target_series = loaded_data.target_square_series;
    catch
        error('Failed to load SINE data from "%s". Run createSINEseries first.', filename);
    end
    
    if length(input_series) < required_len
        error('Loaded series is too short. Required length: %d.', required_len);
    end
    
    task_data.input_series = input_series(1:required_len);
    task_data.target_series = target_series(1:required_len);

    % Split into training and testing sets
    config = get_default_config();
    task_data.train_target = task_data.target_series(config.washout1+1 : config.washout1 + config.num_train_points);
    task_data.test_target  = task_data.target_series(config.wheretostarttest + 1 : config.wheretostarttest + config.num_test_points);
end

%--------------------------------------------------------------------------
% The functions below are copied from your original script.
% No changes are needed for them to work with the new task,
% which shows the power of your modular code design!
% I have only added 'y_test_hat' as an output to test_readout.
%--------------------------------------------------------------------------

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