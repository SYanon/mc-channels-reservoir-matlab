% =========================================================================
% generate_steady_state_response.m
%
% DESCRIPTION:
%   Generates a plot comparing the steady-state receptor occupation vs.
%   input strength for two biophysical parameter sets: one promoting
%   nonlinearity and one promoting a more linear response.
% =========================================================================

%% 0. Initialize Environment
close all;
clear;
fprintf('--- Generating Steady-State Response for Nonlinearity Analysis ---\n');

%% 1. Define Optimal Parameter Configurations
% Using the same optimal sets as the impulse response script.

% --- High-Nonlinearity Config (Optimized for Sine-to-Square Transformation) ---
config_nonlinear = get_default_config();
config_nonlinear.k_on     = 1.5494e-17;
config_nonlinear.k_off    = 2.7868;
config_nonlinear.T        = 1.2279;
config_nonlinear.distance = 4.0983e-06;
config_nonlinear.N_max    = 19925;
config_nonlinear.D        = 1.8232e-10;

% --- Low-Nonlinearity Config (Optimized for Mackey-Glass Forecasting) ---
config_linear = get_default_config();
config_linear.k_on      = 6.6402e-19;
config_linear.k_off     = 4.1491;
config_linear.T         = 1.9917;
config_linear.distance  = 5.127e-06;
config_linear.N_max     = 19400;
config_linear.D         = 1.0183e-11;

% List of configurations to test
configs_to_test = {config_nonlinear, config_linear};
config_names    = {'High-Nonlinearity (Transformation)', 'Low-Nonlinearity (Forecasting)'};
results         = cell(1, length(configs_to_test));

%% 2. Define Inputs and Loop
input_strengths = 0:0.05:1.0; % Test 21 different constant input levels
num_symbols = 200; % Long enough duration to ensure steady state is reached

for iConfig = 1:length(configs_to_test)
    current_config = configs_to_test{iConfig};
    fprintf('\nProcessing config: %s\n', config_names{iConfig});
    
    steady_state_outputs = zeros(size(input_strengths));
    
    % Use parfor for potential speedup if you have the Parallel Computing Toolbox
    for i_strength = 1:length(input_strengths)
        strength = input_strengths(i_strength);
        fprintf('  Testing input strength: %.2f\n', strength);
        
        % Create a long, constant input signal
        input_series = ones(1, num_symbols) * strength;
        
        % Run the simulation
        [sim, ~] = run_channel_simulation(input_series, current_config);
        
        % Store the final value as the steady-state output
        steady_state_outputs(i_strength) = sim.occupation(end);
    end
    results{iConfig} = steady_state_outputs;
end

%% 3. Plot the Results
fprintf('\nAll simulations complete. Generating plot...\n');
figure('Name', 'Nonlinearity: Steady-State Response', 'Position', [100, 100, 800, 600]);

ss_outputs_nonlinear = results{1};
ss_outputs_linear    = results{2};

plot(input_strengths, ss_outputs_nonlinear, 'r-o', 'LineWidth', 2.5, 'DisplayName', 'High-Nonlinearity (Transformation Params)');
hold on;
plot(input_strengths, ss_outputs_linear, 'b-s', 'LineWidth', 2.5, 'DisplayName', 'Low-Nonlinearity (Forecasting Params)');
hold off;

grid on;
title('Tuning Nonlinearity via Biophysical Parameters', 'FontSize', 16);
xlabel('Normalized Input Signal Strength', 'FontSize', 12);
ylabel('Steady-State Receptor Occupation n(t)', 'FontSize', 12);
legend('show', 'Location', 'southeast', 'FontSize', 11);
ylim([0, max([ss_outputs_nonlinear, ss_outputs_linear])*1.1]);
set(gca, 'FontSize', 11);

fprintf('Plot generated successfully.\n');

%% 4. Output Data to Command Window for Analysis
fprintf('\n--- COPY-PASTE THIS DATA FOR ANALYSIS ---\n');
fprintf('input_strengths = %s;\n', mat2str(input_strengths));
fprintf('ss_outputs_high_nonlinearity = %s;\n', mat2str(ss_outputs_nonlinear));
fprintf('ss_outputs_low_nonlinearity = %s;\n', mat2str(ss_outputs_linear));
fprintf('--- END OF DATA ---\n');

%==========================================================================
%                    --- LOCAL HELPER FUNCTIONS ---
% (Copied from your script to make this file self-contained)
%==========================================================================

function config = get_default_config()
    % Provides a base set of parameters.
    config.N     = 500;
    config.N_min = 100;
    config.N_max = 20000; % Set a wide default N_max
    config.dt = 0.001;
    config.memlengthsweep = 100;
    config.T = 1; % Default T, will be overwritten
    % Other specific parameters like Nres are not needed for this simulation
end

function [sim, reservoir_states] = run_channel_simulation(input_series, config)
    % Core simulation of the molecular communication channel.
    % Modified slightly to handle variable input lengths gracefully.
    inp_min = 0; inp_max = 1; % Assume input is already normalized [0, 1]
    if inp_max == inp_min
         N_i = repmat(config.N_min + (input_series(1) - inp_min) * (config.N_max - config.N_min), size(input_series));
    else
        N_i = config.N_min + (input_series - inp_min)/(inp_max - inp_min) * (config.N_max - config.N_min);
    end
    t_total = length(N_i) * config.T;
    sim.time = 0:config.dt:t_total;
    sim.concentration = zeros(size(sim.time));
    Tpeak = config.distance^2/(6*config.D);
    memory_length = round((config.memlengthsweep * Tpeak)/config.T)*config.T;
    if memory_length == 0, memory_length = t_total; end % Ensure it's not zero
    
    for iSym = 1:length(N_i)
        t_symbol_start = (iSym - 1) * config.T;
        t_memory_end   = t_symbol_start + memory_length;
        if t_memory_end > t_total, t_memory_end = t_total; end
        
        start_idx = round(t_symbol_start/config.dt) + 1;
        end_idx = round(t_memory_end/config.dt);
        if start_idx > length(sim.time) || end_idx > length(sim.time) || start_idx > end_idx, continue; end
        idxRange = start_idx:end_idx;
        
        t_local = sim.time(idxRange) - t_symbol_start;
        t_local(t_local <= 0) = 1e-9; % Avoid division by zero
        
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
    
    reservoir_states = []; % Not needed for this analysis
end