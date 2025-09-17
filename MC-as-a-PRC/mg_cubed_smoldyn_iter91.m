%%
% Code to generate Smoldyn data for the Mackey-Glass CUBED task.
%
% This script is configured with the "realistic" and high-performing 
% hyperparameters found in Iteration 91 of the Bayesian Optimization search
% for the MG Cubed task.
%%


%% Initialize the MATLAB environment
close all   % Close all open figure windows
clear       % Clear all variables from the workspace

%% 1. Parameter Setup
% The parameters below are from Iteration 91, identified as the best
% realistic configuration for the MG Cubed task.

fprintf('Setting up parameters from Bayesian Optimization: Iteration 91\n');

% Receptor and binding parameters
N = 500;                % Total number of receptors at the receiver
k_on = 8.0858e-18;     % Binding rate constant -- OPTIMIZED
k_off = 3.5861;        % Unbinding rate constant -- OPTIMIZED

% Time parameters
T = 1.9251;            % Symbol duration (seconds) -- OPTIMIZED

% Data lengths
washout1 = 500;
num_train_points = 500;
wheretostarttest = 1200;
num_test_points  = 500;
washout2 = wheretostarttest - (washout1 + num_train_points);
num_tot_points = washout1 + washout2 + num_train_points + num_test_points;

% Communication channel parameters
distance = 8.0929e-06;   % Distance (m) -- OPTIMIZED
D = 2.9381e-11;        % Diffusion coefficient (m^2/s) -- OPTIMIZED

% Input normalization parameters
N_min = 100;           % Minimum value for input normalization
N_max = 11345;         % Maximum value for input normalization -- OPTIMIZED

% Step size control & channel memory calculation
dt = 0.001;
memlengthsweep = 100;
Tpeak = distance^2/(6*D);
memory_length = round((memlengthsweep * Tpeak)/T)*T;

% NOTE: The prediction length for the MG Cubed task is 10 steps.
predictlength = 10;

%% --- Load Mackey-Glass CUBED Series ---
% We load the pre-processed input and target series for this task.
MG_CUBED_FILENAME = 'MGCubed_series_k10.mat';
try
    loaded_data = load(MG_CUBED_FILENAME, 'input_series');
    input_series = loaded_data.input_series;
    fprintf('Loaded "input_series" from "%s".\n', MG_CUBED_FILENAME);
catch ME
    fprintf('ERROR: Could not load data from "%s".\n', MG_CUBED_FILENAME);
    fprintf('Please ensure you have run createMGcubedseries.m first.\n');
    rethrow(ME);
end

if length(input_series) < num_tot_points
    error('Loaded input series is too short.');
end
% Truncate to required length for the simulation
input_series = input_series(1:num_tot_points);

%% 4. Compute Input Molecule Counts
% N_i will be the number of molecules released for each symbol
N_i = N_min + (input_series - min(input_series)) / (max(input_series) - min(input_series)) * (N_max - N_min);

%% 7. Deterministic Model Calculation (for plotting comparison)
fprintf('Calculating deterministic model response for comparison...\n');
t_total = length(N_i) * T;
t_values = 0:dt:t_total;
c_values = zeros(size(t_values));
n_values = zeros(size(t_values));
n_values(1) = 0;
for i = 1:length(N_i)
    t_symbol_start = (i - 1) * T;
    t_memory_end = t_symbol_start + memory_length;
    indices = find(t_values > t_symbol_start & t_values <= t_memory_end);
    if isempty(indices), continue; end
    t_local = t_values(indices) - t_symbol_start;
    pulse = (N_i(i) ./ ((4 * pi * D * t_local).^(3/2))) .* exp(-distance^2 ./ (4 * D * t_local));
    c_values(indices) = c_values(indices) + pulse;
end
for i = 2:length(t_values)
    c_t = c_values(i - 1);
    n_t = n_values(i - 1);
    dn_dt = k_on * (N - n_t) * c_t - k_off * n_t;
    n_values(i) = n_t + dn_dt * dt;
end
n_values = n_values / N;

%% ==========================================================
%                  SMOLDYN SIMULATION SETUP
% ===========================================================
fprintf('Setting up Smoldyn simulation...\n');
tic;

%% Set Simulation Parameters
% Convert parameters to Smoldyn units
KON = k_on*1e18;
KOFF = k_off;
DIFFMESSENGER = D*1e12;

% Simulation time parameters
START_TIME = 0;
BIT_INTERVAL = T;
STOP_TIME = START_TIME + length(N_i)*BIT_INTERVAL; 
disp(['Smoldyn Simulation Stop Time: ', num2str(STOP_TIME)]);
TIME_STEP = 0.01;

% Receptor and Spatial parameters
numRECEPTOR = N;
TXRXDISTANCE = distance*1e6;
BOUNDARYLENGTH = 25;
TXPOSITION = -10;
RXRADIUS = 3;
RXRECEPTIONSPACETHICKNESS = 0.2;

%% Create Smoldyn Configuration File
input_time_points = START_TIME:BIT_INTERVAL:STOP_TIME-eps*1e10;
generateSimConfig(N_i, input_time_points);

%% Create Directory for Simulation Output
dirname = sprintf('MG_Cubed_Smoldyn_Iter91_%s/', datestr(now,'mm-dd-yyyy--HH-MM-SS'));
mkdir(dirname);
copyfile('MCpointTXsphRX_generated.txt', dirname);

%% Run Smoldyn Simulation
% Using '-wt' for a non-visual (headless) run for speed.
% --- THIS IS THE NEW, CORRECTED BLOCK ---
SAMPLING_PERIOD = 1; % Define the sampling period variable
command = sprintf(['smoldyn %sMCpointTXsphRX_generated.txt -wt -s ', ...
'--define index=%d ', ...
    '--define KON=%d ', ...
    '--define KOFF=%d ', ...
    '--define DIFFMESSENGER=%d ', ...
    '--define DIFFRECEPTOR=%d ', ...
    '--define DIFFRECEPTORACT=%d ', ...
    '--define START_TIME=%d ', ...
    '--define STOP_TIME=%d ', ...
    '--define TIME_STEP=%f ', ... % Time step can be a float
    '--define BIT_INTERVAL=%f ', ... % Bit interval can be a float
    '--define numRECEPTOR=%d ', ...
    '--define TXRXDISTANCE=%d ', ...
    '--define BOUNDARYLENGTH=%d ', ...
    '--define TXPOSITION=%d ', ...
    '--define RXRADIUS=%d ', ...
    '--define RXRECEPTIONSPACETHICKNESS=%f ', ... % Thickness can be a float
    '--define SAMPLING_PERIOD=%d '], ... % ADDED THIS MISSING LINE
    dirname, 1, KON, KOFF, DIFFMESSENGER, ...
    0, 0, START_TIME, round(STOP_TIME), TIME_STEP, BIT_INTERVAL, numRECEPTOR, ...
    round(TXRXDISTANCE), BOUNDARYLENGTH, TXPOSITION, RXRADIUS, RXRECEPTIONSPACETHICKNESS, SAMPLING_PERIOD);

fprintf('Executing Smoldyn... This may take a while.\n');
system(command);
fprintf('Smoldyn simulation complete.\n');

%% Read and Plot Simulation Results
output_filename = sprintf('%soutput.txt', dirname);
data_temp = importdata(output_filename, ' ', 1);
smoldyn_data = data_temp.data;
smoldyn_time = smoldyn_data(:,1);
active_receptors = smoldyn_data(:,2);
occupation_fraction_smoldyn = active_receptors / N;

figure('Position',[400 400 1200 400]);
plot(t_values, n_values, 'LineWidth', 2, 'DisplayName', 'Deterministic Model');
hold on;
plot(smoldyn_time, occupation_fraction_smoldyn, '-', 'DisplayName', 'Stochastic (Smoldyn) Run');
grid on;
xlabel('Time (s)');
ylabel('Occupation Fraction <n(t)>');
title('MG Cubed Task (Iter 91): Deterministic vs. Stochastic Response');
legend;

toc;