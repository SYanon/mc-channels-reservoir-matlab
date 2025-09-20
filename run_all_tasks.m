% =========================================================================
% run_all_tasks_and_plot_clusters.m
%
% DESCRIPTION:
%   A unified script to analyze the hyperparameter spaces for three distinct
%   PRC tasks. It sequentially runs Bayesian optimization for each task,
%   selects the top 10 best-performing trials from each, and then
%   visualizes all 30 elite trials on a single parallel coordinates plot
%   to identify task-specific hyperparameter clustering.
%
%   **VERSION 2**: Corrected the legend creation method to be more robust.
% =========================================================================

%% 0. Initialize Environment
close all;
clear;
rng('default');
fprintf('Starting Unified Bayesian Optimization and Cluster Analysis...\n\n');

%% 1. Define Tasks and Optimization Parameters
tasks = {
    struct('name', 'Forecasting (MG)', 'objective', @objectiveFunction_mg, 'config', @get_default_config_mg, 'loader', @load_mg_data);
    struct('name', 'Transformation (Sine-Sq)', 'objective', @objectiveFunction_sine, 'config', @get_default_config_sine, 'loader', @load_sine_data);
    struct('name', 'Hybrid (MG-Cubed)', 'objective', @objectiveFunction_mg_cubed, 'config', @get_default_config_mg_cubed, 'loader', @load_mg_cubed_data);
};

num_trials = 50;
num_top_to_plot = 10;

% Define the optimizable variables (same for all tasks)
vars = [
    optimizableVariable('k_on', [5e-20, 2e-17], 'Transform', 'log');
    optimizableVariable('k_off', [0.1, 10]);
    optimizableVariable('T', [0.5, 2]);
    optimizableVariable('distance', [1e-6, 50e-6], 'Transform', 'log');
    optimizableVariable('memorywindowlength', [1, 5], 'Type', 'integer');
    optimizableVariable('N_max', [200, 20000]);
    optimizableVariable('D', [0.5e-11, 2e-10], 'Transform', 'log');
];

all_top_params = [];
all_top_groups = [];

%% 2. Run Optimization Sequentially for Each Task
for i = 1:length(tasks)
    task = tasks{i};
    fprintf('===== Running Optimization for Task: %s =====\n', task.name);
    
    % Load data and config for the current task
    config = task.config();
    task_data = task.loader(config);
    objFun = @(params) task.objective(params, task_data, config);
    
    % Run Bayesian Optimization
    results = bayesopt(objFun, vars, ...
        'MaxObjectiveEvaluations', num_trials, ...
        'IsObjectiveDeterministic', true, ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'Verbose', 1);
        
    % Find the top N best trials for this task
    [~, sorted_indices] = sort(results.ObjectiveTrace, 'ascend');
    top_indices = sorted_indices(1:num_top_to_plot);
    
    % Get the hyperparameter sets for the top trials
    top_params_table = results.XTrace(top_indices, :);
    
    % Append to our master list
    all_top_params = [all_top_params; top_params_table];
    all_top_groups = [all_top_groups; repmat(i, num_top_to_plot, 1)];
    
    fprintf('Finished optimization for %s.\n\n', task.name);
end

%% 3. Generate the Unified Parallel Coordinates Plot (CORRECTED SECTION)
fprintf('===== Generating Unified Plot for Top Hyperparameters =====\n');

% A. Extract and normalize the combined data from all tasks
data_matrix = table2array(all_top_params);
param_names = all_top_params.Properties.VariableNames;

min_vals = min(data_matrix, [], 1);
max_vals = max(data_matrix, [], 1);
range_vals = max_vals - min_vals;
range_vals(range_vals == 0) = 1; % Avoid division by zero
normalized_data = (data_matrix - min_vals) ./ range_vals;

% B. Define plot styles for each task
plot_styles = {
    struct('color', 'b', 'marker', 'x', 'name', 'Forecasting (Top 10)');
    struct('color', 'r', 'marker', 'o', 'name', 'Transformation (Top 10)');
    struct('color', [0 0.6 0], 'marker', '^', 'name', 'Hybrid (Top 10)'); % Dark Green
};

% C. Create the plot
figure('Name', 'Hyperparameter Clustering by Task', 'Position', [100, 100, 1200, 700]);
hold on;

for i = 1:size(normalized_data, 1)
    group_idx = all_top_groups(i);
    style = plot_styles{group_idx};
    
    % Plot line with markers AND a DisplayName for the legend
    plot(1:length(param_names), normalized_data(i, :), ...
        'LineStyle', '-', ...
        'Marker', style.marker, ...
        'Color', style.color, ...
        'LineWidth', 1.5, ...
        'MarkerSize', 8, ...
        'DisplayName', style.name); % <-- This DisplayName is key
end

hold off;

% D. Format the plot
xticks(1:length(param_names));
xticklabels(param_names);
xtickangle(30);
ylabel('Normalized Value');
title('Clustering of Top 10 Optimal Hyperparameters by Task');
grid on;
box on;
ylim([0, 1]);

% E. Add a legend (This will now work correctly and automatically)
legend('Location', 'best');

fprintf('Unified plot generated successfully.\n');


% =========================================================================
% =========================================================================
%                    --- LOCAL HELPER FUNCTIONS ---
%             (Helper functions are unchanged from before)
% =========================================================================
% =========================================================================

% -------------------------------------------------------------------------
% --- Functions for Mackey-Glass Forecasting Task ---
% -------------------------------------------------------------------------
function nrmse_test = objectiveFunction_mg(params, data, config)
    run_config = config;
    run_config.k_on = params.k_on; run_config.k_off = params.k_off;
    run_config.T = params.T; run_config.distance = params.distance;
    run_config.memorywindowlength = params.memorywindowlength;
    run_config.N_max = params.N_max; run_config.D = params.D;
    run_config.Nres = config.Nres * run_config.memorywindowlength;
    [~, r_states] = run_channel_simulation(data.input_series, run_config);
    W_out = train_readout(r_states, data.train_target, run_config);
    [~, nrmse_test] = test_readout(W_out, r_states, data, run_config);
end

function config = get_default_config_mg()
    config.task_name = 'mg_forecasting';
    config.N = 500; config.N_min = 100; config.lambda = 1e-6; config.Nres = 50;
    config.predictlength = 6; config.washout1 = 500; config.num_train_points = 500;
    config.wheretostarttest = 1200; config.num_test_points = 500;
    config.washout2 = config.wheretostarttest - (config.washout1 + config.num_train_points);
    config.num_tot_points = config.washout1 + config.washout2 + config.num_train_points + config.num_test_points;
    config.dt = 0.001; config.memlengthsweep = 100;
    config.MG_FILENAME = 'MGseries_RK4_tau17_beta0.20_gamma0.1_n10_len5000_dt1.0.mat';
end

function data = load_mg_data(config)
    loaded = load(config.MG_FILENAME, 'mackey_glass_series');
    series_raw = loaded.mackey_glass_series;
    seg = series_raw(1 : config.num_tot_points + config.predictlength);
    norm_seg = (seg - min(seg)) / (max(seg) - min(seg));
    data.input_series = norm_seg(1:end - config.predictlength);
    data.target_series = norm_seg((config.predictlength+1):end);
    data.train_target = data.target_series(config.washout1+1 : config.washout1 + config.num_train_points);
    data.test_target = data.target_series(config.wheretostarttest+1 : config.wheretostarttest + config.num_test_points);
end

% -------------------------------------------------------------------------
% --- Functions for Sine-to-Square Transformation Task ---
% -------------------------------------------------------------------------
function nrmse_test = objectiveFunction_sine(params, data, config)
    run_config = config;
    run_config.k_on = params.k_on; run_config.k_off = params.k_off;
    run_config.T = params.T; run_config.distance = params.distance;
    run_config.memorywindowlength = params.memorywindowlength;
    run_config.N_max = params.N_max; run_config.D = params.D;
    run_config.Nres = config.Nres * run_config.memorywindowlength;
    [~, r_states] = run_channel_simulation(data.input_series, run_config);
    W_out = train_readout(r_states, data.train_target, run_config);
    [~, nrmse_test] = test_readout(W_out, r_states, data, run_config);
end

function config = get_default_config_sine()
    config.task_name = 'sine_transform';
    config.N = 500; config.N_min = 100; config.lambda = 1e-6; config.Nres = 100;
    config.washout1 = 500; config.num_train_points = 2000; config.wheretostarttest = 3000;
    config.num_test_points = 1000;
    config.washout2 = config.wheretostarttest - (config.washout1 + config.num_train_points);
    config.num_tot_points = config.washout1 + config.washout2 + config.num_train_points + config.num_test_points;
    config.dt = 0.001; config.memlengthsweep = 100;
    config.SINE_FILENAME = 'SINEseries_len5000_p25.mat';
end

function data = load_sine_data(config)
    loaded = load(config.SINE_FILENAME, 'input_sine_series', 'target_square_series');
    data.input_series = loaded.input_sine_series(1:config.num_tot_points);
    data.target_series = loaded.target_square_series(1:config.num_tot_points);
    data.train_target = data.target_series(config.washout1+1 : config.washout1 + config.num_train_points);
    data.test_target = data.target_series(config.wheretostarttest + 1 : config.wheretostarttest + config.num_test_points);
end

% -------------------------------------------------------------------------
% --- Functions for Mackey-Glass Cubed Hybrid Task ---
% -------------------------------------------------------------------------
function nrmse_test = objectiveFunction_mg_cubed(params, data, config)
    run_config = config;
    run_config.k_on = params.k_on; run_config.k_off = params.k_off;
    run_config.T = params.T; run_config.distance = params.distance;
    run_config.memorywindowlength = params.memorywindowlength;
    run_config.N_max = params.N_max; run_config.D = params.D;
    run_config.Nres = config.Nres * run_config.memorywindowlength;
    [~, r_states] = run_channel_simulation(data.input_series, run_config);
    W_out = train_readout(r_states, data.train_target, run_config);
    [~, nrmse_test] = test_readout(W_out, r_states, data, run_config);
end

function config = get_default_config_mg_cubed()
    config.task_name = 'mg_cubed_hybrid';
    config.N = 500; config.N_min = 100; config.lambda = 1e-6; config.Nres = 50;
    config.washout1 = 500; config.num_train_points = 500; config.wheretostarttest = 1200;
    config.num_test_points = 500;
    config.washout2 = config.wheretostarttest - (config.washout1 + config.num_train_points);
    config.num_tot_points = config.washout1 + config.washout2 + config.num_train_points + config.num_test_points;
    config.dt = 0.001; config.memlengthsweep = 100;
    config.MG_CUBED_FILENAME = 'MGCubed_series_k10.mat';
end

function data = load_mg_cubed_data(config)
    loaded = load(config.MG_CUBED_FILENAME, 'input_series', 'target_series');
    data.input_series = loaded.input_series(1:config.num_tot_points);
    data.target_series = loaded.target_series(1:config.num_tot_points);
    data.train_target = data.target_series(config.washout1+1 : config.washout1 + config.num_train_points);
    data.test_target = data.target_series(config.wheretostarttest+1 : config.wheretostarttest + config.num_test_points);
end

% -------------------------------------------------------------------------
% --- COMMON SIMULATION AND LEARNING FUNCTIONS ---
% -------------------------------------------------------------------------
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
    test_indices = config.wheretostarttest+1 : config.wheretostarttest + config.num_test_points;
    test_states = reservoir_states(:, test_indices);
    X_test = [test_states; ones(1, config.num_test_points)];
    y_test_hat = X_test' * W_out;
    nrmse_test = sqrt(mean((y_test_hat - mg_data.test_target(:)).^2)) / std(mg_data.test_target);
end