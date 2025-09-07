% =========================================================================
% lorenz_generate_smoldyn_data.m
%
% DESCRIPTION:
%   Generates and runs a Smoldyn simulation for the Lorenz '63 task using
%   a pre-defined set of optimal hyperparameters.
%
% HOW TO USE:
%   1. Paste your best hyperparameters from the Bayesian optimization into
%      the "Optimal Hyperparameters" section below.
%   2. Ensure Smoldyn is installed and accessible from your system's path.
%   3. Press "Run" in the MATLAB editor.
% =========================================================================

%% 0. Initialize Environment
close all;
clear;
fprintf('Starting Lorenz ''63 Smoldyn data generation...\n');

%% 1. Optimal Hyperparameters
% --- UPDATED with recommended parameters for a stable Smoldyn run (Iter #89) ---
run_config.k_on     = 6.01e-18;
run_config.k_off    = 0.225;
run_config.T        = 1.140;       % Sub-symbol duration
run_config.distance = 3.70e-06;
run_config.N_max    = 7394;
run_config.D        = 9.71e-11;
fprintf('Using optimized hyperparameters chosen for Smoldyn stability.\n');

%% 2. Load Default Config and Data
% Load fixed parameters (like data lengths, Nres, etc.)
default_config = get_default_config();
run_config.N_min = default_config.N_min;
run_config.N     = default_config.N;
run_config.Nres  = default_config.Nres;
run_config.dt    = default_config.dt;
run_config.memlengthsweep = default_config.memlengthsweep;

% Load the Lorenz time series data
lorenz_data = load(default_config.LORENZ_FILENAME, 'lorenz_series');
full_series = lorenz_data.lorenz_series(1:default_config.num_tot_points, :);

%% 3. Prepare Serialized Input for Smoldyn
[num_points, num_dims] = size(full_series);

% Serialize the 3D input into a 1D stream: [x1,y1,z1, x2,y2,z2,...]
serialized_input = reshape(full_series', 1, []);

% Normalize this 1D stream to get the number of molecules for each pulse
N_i = run_config.N_min + (serialized_input - 0)/(1 - 0) * (run_config.N_max - run_config.N_min);

% Generate the corresponding time points for each serialized pulse
base_times = (0:num_points-1)' * (num_dims * run_config.T);
offsets = 0:run_config.T:(num_dims-1)*run_config.T;
input_time_points_matrix = base_times + offsets;
input_time_points = reshape(input_time_points_matrix', 1, []);

%% 4. Set Smoldyn Simulation Parameters
% Convert MATLAB units to Smoldyn units (micrometers, etc.)
KON = run_config.k_on * 1e18;      % NOTE: This conversion factor is based on your MG script.
KOFF = run_config.k_off;
DIFFMESSENGER = run_config.D * 1e12; % m^2/s to um^2/s

START_TIME = 0;
STOP_TIME = num_points * num_dims * run_config.T; % Total simulation time
TIME_STEP = 0.01;         % Smoldyn's internal time step
SAMPLING_PERIOD = 0.1;    % How often to save Smoldyn output (seconds)
TXRXDISTANCE = run_config.distance * 1e6; % m to um

fprintf('Smoldyn simulation will run for %.2f seconds.\n', STOP_TIME);

%% 5. Generate Smoldyn Configuration File
% This calls a helper function to write the .txt file Smoldyn will use
generateLorenzSimConfig(N_i, input_time_points);
fprintf('Smoldyn configuration file generated.\n');

%% 6. Run Smoldyn Simulation
% Create a time-stamped directory for the output files
dirname = sprintf('lorenz_smoldyn_%s/', datestr(now,'mm-dd-yyyy_HH-MM-SS'));
mkdir(dirname);
copyfile('MC_lorenz_smoldyn_config.txt', dirname);

% Build and execute the system command to run Smoldyn
% Use '-w' for visualization (slower) or '-wt' for no visualization (faster)
command = sprintf(['smoldyn %sMC_lorenz_smoldyn_config.txt -wt ', ...
    '--define KON=%d ', '--define KOFF=%d ', '--define DIFFMESSENGER=%d ', ...
    '--define START_TIME=%d ', '--define STOP_TIME=%d ', '--define TIME_STEP=%f ', ...
    '--define SAMPLING_PERIOD=%f ', '--define TXRXDISTANCE=%d '], ...
    dirname, KON, KOFF, DIFFMESSENGER, START_TIME, STOP_TIME, TIME_STEP, ...
    SAMPLING_PERIOD, TXRXDISTANCE);

fprintf('Running Smoldyn... (This may take a while)\n');
system(command);
fprintf('Smoldyn simulation complete.\n');

%% 7. Read and Process Smoldyn Results
fprintf('Reading Smoldyn output files...\n');
allMoleculesFilename = sprintf('%sallmolecules.txt', dirname);
data_temp = importdata(allMoleculesFilename,' ',1); % Skip header
data = data_temp.data;
smoldyn_time = data(:, 1);
smoldyn_active_receptors = data(:, end) / run_config.N; % Normalize to a fraction

%% 8. Run Deterministic Simulation for Comparison
fprintf('Running deterministic simulation for comparison...\n');
[sim, ~] = run_channel_simulation(full_series, run_config);

%% 9. Plot and Compare Results
fprintf('Plotting results...\n');
figure('Name', 'Smoldyn vs. Deterministic Model for Lorenz Task');
% Plot the deterministic (mean-field) result
plot(sim.time, sim.occupation, 'b-', 'LineWidth', 2, 'DisplayName', 'Deterministic (MATLAB)');
hold on;
% Plot the stochastic (Smoldyn) result
plot(smoldyn_time, smoldyn_active_receptors, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Stochastic (Smoldyn)');
title('Reservoir State (Receptor Occupation)');
xlabel('Time (s)');
ylabel('Fraction of Bound Receptors');
legend;
grid on;
xlim([0, STOP_TIME]);

fprintf('✅ Workflow complete. Check the plot and the "%s" directory for data.\n', dirname);

%==========================================================================
%                       LOCAL HELPER FUNCTIONS
%==========================================================================

function config = get_default_config()
    % Fixed (non-optimized) parameters
    config.N     = 500;
    config.N_min = 100;
    config.Nres   = 150;
    config.dt = 0.001;
    config.memlengthsweep = 100;
    config.num_tot_points = 4000; % Number of points to use from the Lorenz series
    config.LORENZ_FILENAME = 'lorenz_series_len8000.mat';
end

function lorenz_data = load_lorenz_data(filename, required_len)
    loaded = load(filename, 'lorenz_series');
    lorenz_data = loaded.lorenz_series;
    if size(lorenz_data, 1) < required_len
        error('Loaded Lorenz series is too short.');
    end
    lorenz_data = lorenz_data(1:required_len, :);
end

% This is the same deterministic simulation function from your other scripts
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
        epsilon = 1e-12;
        pulse = (N_i(iSym) ./ ((4*pi*config.D*t_local + epsilon).^(3/2))) .* exp(-config.distance^2./(4*config.D*t_local + epsilon));
        sim.concentration(idxRange) = sim.concentration(idxRange) + pulse;
    end
    sim.occupation = zeros(size(sim.time));
    for idxT = 2:length(sim.time)
        c_t = sim.concentration(idxT - 1);
        n_t = sim.occupation(idxT - 1);
        dn_dt = config.k_on*(config.N - n_t*config.N)*c_t - config.k_off*n_t*config.N;
        sim.occupation(idxT) = sim.occupation(idxT-1) + (dn_dt/config.N)*config.dt;
    end
    reservoir_states = zeros(config.Nres, num_points); % Not needed for this script, but kept for consistency
end