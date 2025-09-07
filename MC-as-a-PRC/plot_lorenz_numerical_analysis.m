% =========================================================================
% plot_lorenz_numerical_analysis.m
%
% DESCRIPTION:
%   Performs a two-parameter sweep for the Lorenz '63 prediction task.
%   **MODIFIED** to include intermediate 3D attractor plots for each run.
% =========================================================================

%% 0. Initialize Environment & Define Configuration
close all;
clear;
fprintf('Starting Lorenz ''63 parameter sweep...\n');

config = get_default_config();

%% 1. Load and Prepare Data (Done ONCE)
lorenz_data = load_lorenz_data(config.LORENZ_FILENAME, config.num_tot_points, config.predictlength);

%% 2. Prepare for Sweep
nP1 = length(config.param1_values);
nP2 = length(config.param2_values);
results.nrmse_test = NaN(nP1, nP2);

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

        % Run the core simulation with serialized 3D input
        [sim, reservoir_states] = run_channel_simulation(lorenz_data.input_series, run_config);
        
        % Train the readout to map to a 3D target
        W_out = train_readout(reservoir_states, lorenz_data.train_target, run_config);
        
        % Test the readout and get the 3D prediction
        [~, nrmse_test, y_test_hat] = test_readout(W_out, reservoir_states, lorenz_data, run_config);

        results.nrmse_test(i1, i2)  = nrmse_test;
        
        fprintf('  -> NRMSE(Test)=%.4f. (took %.2fs)\n', nrmse_test, toc);

        % --- MODIFIED SECTION: In-loop 3D Attractor Plotting ---
        if run_config.plot_in_loop
            figure('Name', sprintf('Run %d: %s=%g, %s=%g', (i2-1)*nP1 + i1, p1_name, run_config.(p1_name), p2_name, run_config.(p2_name)));
            % Plot the true attractor trajectory
            plot3(lorenz_data.test_target(:,1), lorenz_data.test_target(:,2), lorenz_data.test_target(:,3), 'b-', 'LineWidth', 1.5, 'DisplayName', 'True Attractor');
            hold on;
            % Overlay the predicted attractor trajectory
            plot3(y_test_hat(:,1), y_test_hat(:,2), y_test_hat(:,3), 'r--', 'LineWidth', 1, 'DisplayName', 'Predicted Attractor');
            title(sprintf('Prediction | NRMSE: %.4f', nrmse_test));
            xlabel('x'); ylabel('y'); zlabel('z');
            legend('Location', 'northeast');
            grid on;
            axis tight;
            view(35, 25); % Set a consistent 3D viewing angle
            drawnow; % Force MATLAB to render the plot immediately
        end
        % --- END OF MODIFIED SECTION ---
    end
end

%% 4. Visualize Sweep Results
fprintf('Sweep complete. Generating heatmap...\n');
[X, Y] = meshgrid(config.param1_values, config.param2_values);
plot_heatmap(X, Y, results.nrmse_test', 'NRMSE (Test)', config);

fprintf('All done.\n');

%==========================================================================
%                    --- LOCAL HELPER FUNCTIONS ---
%==========================================================================

function config = get_default_config()
    % Central configuration for the Lorenz task
    config.param1_name   = 'k_on';
    config.param1_values = [1e-19, 5e-19, 1e-18, 5e-18, 1e-17];
    config.param2_name   = 'k_off';
    config.param2_values = [0.5, 1, 2, 5, 10];
    
    config.N     = 500;
    config.N_min = 100;
    config.N_max = 5000;
    config.lambda = 1e-7;
    config.Nres   = 150;
    config.distance = 10e-6;
    config.D        = 1e-11;
    config.T  = 0.5;
    config.predictlength = 1;
    
    config.washout1         = 500;
    config.num_train_points = 2000;
    config.wheretostarttest = 3000;
    config.num_test_points  = 1500;
    config.washout2 = config.wheretostarttest - (config.washout1 + config.num_train_points);
    config.num_tot_points = config.washout1 + config.washout2 + config.num_train_points + config.num_test_points;
    
    config.dt = 0.001;
    config.memlengthsweep = 100;
    config.LORENZ_FILENAME = 'lorenz_series_len8000.mat';

    % --- ADDED: Flag to control intermediate plotting ---
    config.plot_in_loop = true; % Set to 'false' to disable per-run plots
end

function lorenz_data = load_lorenz_data(filename, required_len, predict_len)
    loaded = load(filename, 'lorenz_series');
    full_series = loaded.lorenz_series;
    if size(full_series, 1) < required_len + predict_len
        error('Loaded Lorenz series is too short.');
    end
    series_segment = full_series(1 : required_len + predict_len, :);
    lorenz_data.input_series  = series_segment(1:end - predict_len, :);
    lorenz_data.target_series = series_segment(predict_len + 1 : end, :);
    config = get_default_config();
    tr_end = config.washout1 + config.num_train_points;
    te_start = config.wheretostarttest + 1;
    te_end = te_start + config.num_test_points - 1;
    lorenz_data.train_target = lorenz_data.target_series(config.washout1+1 : tr_end, :);
    lorenz_data.test_target  = lorenz_data.target_series(te_start : te_end, :);
end

function [sim, reservoir_states] = run_channel_simulation(input_matrix, config)
    [num_points, num_dims] = size(input_matrix);
    serialized_input = reshape(input_matrix', 1, []);
    
    sub_symbol_T = config.T; 
    t_total = num_points * num_dims * sub_symbol_T;
    
    N_i = config.N_min + (serialized_input - 0)/(1 - 0) * (config.N_max - config.N_min);
    
    sim.time = 0:config.dt:t_total;
    sim.concentration = zeros(size(sim.time));
    Tpeak = config.distance^2/(6*config.D);
    memory_length = round((config.memlengthsweep * Tpeak)/sub_symbol_T)*sub_symbol_T;
    
    for iSym = 1:length(N_i)
        t_symbol_start = (iSym - 1) * sub_symbol_T;
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
    
    reservoir_states = zeros(config.Nres, num_points);
    steps_per_full_symbol = (num_dims * sub_symbol_T) / config.dt;
    for iPoint = 1:num_points
        t_symbol_start = (iPoint - 1) * num_dims * sub_symbol_T;
        start_idx = round(t_symbol_start/config.dt) + 1;
        sample_indices_float = start_idx + (0:(config.Nres-1)) * (steps_per_full_symbol / config.Nres);
        sample_indices = round(sample_indices_float);
        if all(sample_indices > 0 & sample_indices <= length(sim.occupation))
            reservoir_states(:, iPoint) = sim.occupation(sample_indices);
        end
    end
end

function W_out = train_readout(reservoir_states, train_target, config)
    train_states = reservoir_states(:, config.washout1+1 : config.washout1+config.num_train_points);
    X_train = [train_states; ones(1, size(train_states, 2))];
    Y_train = train_target;
    W_out = pinv(X_train * X_train' + config.lambda * eye(size(X_train, 1))) * (X_train * Y_train);
end

function [nrmse_train, nrmse_test, y_test_hat] = test_readout(W_out, reservoir_states, lorenz_data, config)
    train_indices = config.washout1+1 : config.washout1+config.num_train_points;
    train_states = reservoir_states(:, train_indices);
    X_train = [train_states; ones(1, size(train_states, 2))];
    y_train_hat = X_train' * W_out;
    train_error = y_train_hat - lorenz_data.train_target;
    nrmse_train = sqrt(mean(train_error(:).^2)) / std(lorenz_data.train_target(:));
    
    test_indices = config.wheretostarttest+1 : config.wheretostarttest+config.num_test_points;
    test_states = reservoir_states(:, test_indices);
    X_test = [test_states; ones(1, size(test_states, 2))];
    y_test_hat = X_test' * W_out;
    test_error = y_test_hat - lorenz_data.test_target;
    nrmse_test = sqrt(mean(test_error(:).^2)) / std(lorenz_data.test_target(:));
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