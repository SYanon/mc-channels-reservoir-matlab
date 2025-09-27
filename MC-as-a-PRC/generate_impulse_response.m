% =========================================================================
% generate_impulse_response.m
%
% DESCRIPTION:
%   Generates a plot comparing the impulse response of the MC channel for
%   two different biophysical parameter sets: one optimized for memory
%   (Forecasting) and one for a fast response (Transformation).
% =========================================================================

%% 0. Initialize Environment
close all;
clear;
fprintf('--- Generating Impulse Response for Memory Analysis ---\n');

%% 1. Define Optimal Parameter Configurations
% These values are taken directly from your Bayesian optimization results.

% --- High-Memory Config (Optimized for Mackey-Glass Forecasting) ---
config_memory = get_default_config(); % Start with a base config
config_memory.k_on      = 6.6402e-19;
config_memory.k_off     = 4.1491;
config_memory.T         = 1.9917;
config_memory.distance  = 5.127e-06;
config_memory.N_max     = 19400;
config_memory.D         = 1.0183e-11;

% --- Low-Memory Config (Optimized for Sine-to-Square Transformation) ---
config_nonlinear = get_default_config(); % Start with a base config
config_nonlinear.k_on     = 1.5494e-17;
config_nonlinear.k_off    = 2.7868;
config_nonlinear.T        = 1.2279;
config_nonlinear.distance = 4.0983e-06;
config_nonlinear.N_max    = 19925;
config_nonlinear.D        = 1.8232e-10;

%% 2. Create the Impulse Input Signal
% A single pulse of '1' at the beginning, followed by zeros.
num_symbols = 500; % Long enough to observe the full decay
input_series = [1, zeros(1, num_symbols - 1)];
fprintf('Created an impulse input signal of length %d.\n', num_symbols);

%% 3. Run Simulations for Both Configurations
fprintf('Running simulation for HIGH-MEMORY configuration...\n');
[sim_memory, ~] = run_channel_simulation(input_series, config_memory);

fprintf('Running simulation for LOW-MEMORY configuration...\n');
[sim_nonlinear, ~] = run_channel_simulation(input_series, config_nonlinear);

%% 4. Plot the Results
fprintf('Simulations complete. Generating plot...\n');
figure('Name', 'Fading Memory: Impulse Response Comparison', 'Position', [100, 100, 800, 600]);

plot(sim_memory.time, sim_memory.occupation, 'b-', 'LineWidth', 2.5, 'DisplayName', 'High-Memory (Forecasting Params)');
hold on;
plot(sim_nonlinear.time, sim_nonlinear.occupation, 'r-', 'LineWidth', 2.5, 'DisplayName', 'Low-Memory (Transformation Params)');
hold off;

grid on;
title('Tuning Fading Memory via Biophysical Parameters', 'FontSize', 16);
xlabel('Time (s)', 'FontSize', 12);
ylabel('Receptor Occupation Fraction n(t)', 'FontSize', 12);
legend('show', 'Location', 'northeast', 'FontSize', 11);
xlim([0, 50]); % Adjust x-limit to focus on the decay
set(gca, 'FontSize', 11);

fprintf('Plot generated successfully.\n');

%% 5. Output Data to Command Window for Analysis
% You can copy this output and share it for further discussion.
fprintf('\n--- COPY-PASTE THIS DATA FOR ANALYSIS ---\n');
fprintf('time_high_memory = %s;\n', mat2str(sim_memory.time));
fprintf('occupation_high_memory = %s;\n', mat2str(sim_memory.occupation));
fprintf('time_low_memory = %s;\n', mat2str(sim_nonlinear.time));
fprintf('occupation_low_memory = %s;\n', mat2str(sim_nonlinear.occupation));
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
    inp_min = min(input_series);
    inp_max = max(input_series);
    if inp_max == inp_min
        N_i = config.N_min + (input_series - inp_min) * (config.N_max - config.N_min);
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
        if start_idx > length(sim.time) || end_idx > length(sim.time), continue; end
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