%% sine_smoldyn_data.m
%
% DESCRIPTION:
%   Generates Smoldyn data for the sine-to-square wave transformation task.
%   This script is configured with the "realistic" and high-performing 
%   hyperparameters found in Iteration 23 of the Bayesian Optimization search
%   for this specific task.
%
% PREREQUISITES:
%   1. 'SINEseries_len5000_p25.mat' file must be present (run createSINEseries.m).
%   2. Smoldyn executable must be in the system's PATH.
%   3. 'generateSimConfig.m' helper function must be available.
%%

%% Initialize the MATLAB environment
close all;   % Close all open figure windows
clear;       % Clear all variables from the workspace

%% ==========================================================
%                  1. PARAMETER SETUP
% ===========================================================
% The parameters below are from Iteration 23, identified as the best
% realistic configuration for the Sine-to-Square task.

fprintf('Setting up parameters from Bayesian Optimization: Sine-to-Square Task (Iteration 23)\n');

% Receptor and binding parameters
N = 500;                % Total number of receptors (fixed default)
k_on = 1.98e-17;       % Binding rate constant (s^-1 M^-1) -- OPTIMIZED
k_off = 6.44;          % Unbinding rate constant (s^-1) -- OPTIMIZED

% Time parameters
T = 1.64;              % Symbol duration (seconds) -- OPTIMIZED

% Data lengths (matching the optimization script)
washout1         = 500;
num_train_points = 2000;
wheretostarttest = 3000;
num_test_points  = 1000;
washout2 = wheretostarttest - (washout1 + num_train_points);
num_tot_points = washout1 + washout2 + num_train_points + num_test_points;

% Communication channel parameters
distance = 4.53e-06;   % Distance (m) -- OPTIMIZED
D = 1.17e-11;        % Diffusion coefficient (m^2/s) -- OPTIMIZED

% Input normalization parameters
N_min = 100;           % Minimum value for input normalization (fixed default)
N_max = 9564;          % Maximum value for input normalization -- OPTIMIZED

% Step size control & channel memory calculation
dt = 0.001;
memlengthsweep = 100;
Tpeak = distance^2/(6*D);
memory_length = round((memlengthsweep * Tpeak)/T)*T;


%% ==========================================================
%                  2. LOAD AND PREPARE DATA
% ===========================================================
SINE_FILENAME = 'SINEseries_len5000_p25.mat';
try
    % Load the sine wave input data
    loaded_data = load(SINE_FILENAME, 'input_sine_series');
    input_series = loaded_data.input_sine_series;
    fprintf('Loaded "input_sine_series" from "%s".\n', SINE_FILENAME);
catch ME
    fprintf('ERROR: Could not load data from "%s".\n', SINE_FILENAME);
    fprintf('Please ensure you have run createSINEseries.m first.\n');
    rethrow(ME);
end

if length(input_series) < num_tot_points
    error('Loaded input series is too short. Required length: %d, Found: %d', num_tot_points, length(input_series));
end
% Truncate to the exact required length for the simulation
input_series = input_series(1:num_tot_points);

% Compute the number of molecules to be released for each symbol
% The input_series is already normalized to [0, 1]
N_i = N_min + input_series * (N_max - N_min);


%% ==========================================================
%               3. DETERMINISTIC MODEL CALCULATION
% ===========================================================
fprintf('Calculating deterministic model response for comparison...\n');
t_total = length(N_i) * T;
t_values = 0:dt:t_total;
c_values = zeros(size(t_values));
n_values = zeros(size(t_values));
n_values(1) = 0; % Initial occupation is zero

% Superposition of concentration pulses from each symbol
for i = 1:length(N_i)
    t_symbol_start = (i - 1) * T;
    t_memory_end = t_symbol_start + memory_length;
    indices = find(t_values > t_symbol_start & t_values <= t_memory_end);
    if isempty(indices), continue; end
    t_local = t_values(indices) - t_symbol_start;
    pulse = (N_i(i) ./ ((4 * pi * D * t_local).^(3/2))) .* exp(-distance^2 ./ (4 * D * t_local));
    c_values(indices) = c_values(indices) + pulse;
end

% Solve for receptor occupation using the rate equation
for i = 2:length(t_values)
    c_t = c_values(i - 1);
    n_t = n_values(i - 1); % n_t is occupation NUMBER here
    dn_dt = k_on * (N - n_t) * c_t - k_off * n_t;
    n_values(i) = n_t + dn_dt * dt;
end
occupation_fraction_deterministic = n_values / N; % Normalize to get fraction


%% ==========================================================
%                  4. SMOLDYN SIMULATION SETUP
% ===========================================================
fprintf('Setting up Smoldyn simulation...\n');
tic;

%% Set Simulation Parameters
% Convert parameters to Smoldyn units (micrometers, seconds)
KON = k_on * 1e18;         % Smoldyn wants M^-1 s^-1, but uses um^3 internally
KOFF = k_off;
DIFFMESSENGER = D * 1e12;  % Convert m^2/s to um^2/s

% Simulation time parameters
START_TIME = 0;
BIT_INTERVAL = T;
STOP_TIME = START_TIME + length(N_i) * BIT_INTERVAL; 
disp(['Smoldyn Simulation Stop Time: ', num2str(STOP_TIME)]);
TIME_STEP = 0.01; % Smoldyn simulation time step

% Receptor and Spatial parameters (reusing from template)
numRECEPTOR = N;
TXRXDISTANCE = distance * 1e6; % Convert m to um
BOUNDARYLENGTH = 25;
TXPOSITION = -10;
RXRADIUS = 3;
RXRECEPTIONSPACETHICKNESS = 0.2;

%% Create Smoldyn Configuration File
input_time_points = START_TIME:BIT_INTERVAL:STOP_TIME - eps*1e10;
generateSimConfig(N_i, input_time_points);

%% Create Directory for Simulation Output
dirname = sprintf('Sine_Smoldyn_Iter23_%s/', datestr(now,'mm-dd-yyyy--HH-MM-SS'));
mkdir(dirname);
copyfile('MCpointTXsphRX_generated.txt', dirname);

%% Run Smoldyn Simulation
% Using '-wt' for a non-visual (headless) run for speed.
SAMPLING_PERIOD = 1; % Define the sampling period for output
command = sprintf(['smoldyn %sMCpointTXsphRX_generated.txt -wt -s ', ...
'--define index=%d ', ...
    '--define KON=%d ', ...
    '--define KOFF=%d ', ...
    '--define DIFFMESSENGER=%d ', ...
    '--define DIFFRECEPTOR=%d ', ...
    '--define DIFFRECEPTORACT=%d ', ...
    '--define START_TIME=%d ', ...
    '--define STOP_TIME=%d ', ...
    '--define TIME_STEP=%f ', ...
    '--define BIT_INTERVAL=%f ', ...
    '--define numRECEPTOR=%d ', ...
    '--define TXRXDISTANCE=%d ', ...
    '--define BOUNDARYLENGTH=%d ', ...
    '--define TXPOSITION=%d ', ...
    '--define RXRADIUS=%d ', ...
    '--define RXRECEPTIONSPACETHICKNESS=%f ', ...
    '--define SAMPLING_PERIOD=%d '], ...
    dirname, 1, KON, KOFF, DIFFMESSENGER, ...
    0, 0, START_TIME, round(STOP_TIME), TIME_STEP, BIT_INTERVAL, numRECEPTOR, ...
    round(TXRXDISTANCE), BOUNDARYLENGTH, TXPOSITION, RXRADIUS, RXRECEPTIONSPACETHICKNESS, SAMPLING_PERIOD);

fprintf('Executing Smoldyn... This may take a while.\n');
system(command);
fprintf('Smoldyn simulation complete.\n');


%% ==========================================================
%               5. READ AND PLOT SIMULATION RESULTS
% ===========================================================
output_filename = sprintf('%soutput.txt', dirname);
data_temp = importdata(output_filename, ' ', 1);
smoldyn_data = data_temp.data;
smoldyn_time = smoldyn_data(:,1);
active_receptors = smoldyn_data(:,2);
occupation_fraction_smoldyn = active_receptors / N;

figure('Position',[400 400 1200 400]);
plot(t_values, occupation_fraction_deterministic, 'LineWidth', 2, 'DisplayName', 'Deterministic Model');
hold on;
plot(smoldyn_time, occupation_fraction_smoldyn, '-', 'DisplayName', 'Stochastic (Smoldyn) Run');
grid on;
xlabel('Time (s)');
ylabel('Occupation Fraction <n(t)>');
title('Sine-to-Square Task (Iter 23): Deterministic vs. Stochastic Response');
legend('Location', 'best');

toc;