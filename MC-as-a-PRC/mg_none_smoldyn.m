%%
% Code to generate Smoldyn data given the MC system settings and input
% sequence (in this case MG Series generated beforehand)
% MODIFIED to use the Bayesian Optimized hyperparameters for the baseline MG task.
%%


%% Initialize the MATLAB environment
close all   % Close all open figure windows
clear       % Clear all variables from the workspace

%% 1. Parameter Setup
% The parameters below have been updated with the best configuration found
% by the Bayesian Optimizer for the baseline ('none') Mackey-Glass task.

% Receptor and binding parameters
N = 500;                % Total number of receptors at the receiver
k_on = 6.8778e-18;     % Binding rate constant (1/(M*s)) -- BAYESIAN OPTIMIZED
k_off = 9.616;         % Unbinding rate constant (1/s) -- BAYESIAN OPTIMIZED
KD = k_off / k_on;      % Dissociation constant

% Time parameters
T = 0.99696;           % Symbol duration (seconds) -- BAYESIAN OPTIMIZED

% Data lengths (using standard values from optimization script config)
washout1 = 500;
num_train_points = 500;
wheretostarttest = 1200;
num_test_points  = 500;
washout2 = wheretostarttest - (washout1 + num_train_points);
num_tot_points = washout1 + washout2 + num_train_points + num_test_points;

% Communication channel parameters
distance = 6.6544e-06;   % Distance (m) -- BAYESIAN OPTIMIZED
D = 3.4772e-11;        % Diffusion coefficient (m^2/s) -- BAYESIAN OPTIMIZED

% Input normalization parameters
N_min = 100;           % Minimum value for input normalization
N_max = 10396;         % Maximum value for input normalization -- BAYESIAN OPTIMIZED

% Step size control
dt = 0.001;  % Time step for deterministic/numerical analysis %fixed
memlengthsweep = 100;
Tpeak = distance^2/(6*D);
memory_length = round((memlengthsweep * Tpeak)/T)*T; % memory of MC channel
offset = 0;

% Mackey-Glass time series parameters
predictlength = 6; % Prediction length (number of steps ahead) -- FROM OPTIMIZED CONFIG

%% --- Load Mackey-Glass Series ---
MG_filename = 'MGseries_RK4_tau17_beta0.20_gamma0.1_n10_len5000_dt1.0.mat';
load(MG_filename, 'mackey_glass_series');
disp('Loaded "mackey_glass_series" from MAT file.');
if length(mackey_glass_series) < num_tot_points + predictlength
    error('Loaded series is too short. Increase MG_length or reduce num_tot_points.');
end

series_segment = mackey_glass_series(1 : num_tot_points + predictlength);
mg_min = min(series_segment);
mg_max = max(series_segment);
mackey_glass_series = (series_segment - mg_min)/(mg_max - mg_min);

%% 4. Compute Input Molecule Counts
input_series  = mackey_glass_series(1:end - predictlength);
N_i = N_min + (input_series - min(input_series)) / (max(input_series) - min(input_series)) * (N_max - N_min);

%% 6. Generate Time Vector for Deterministic Comparison Plot
t_total = length(N_i) * T;
t_values = 0:dt:t_total;

%% 7. Deterministic Model Calculation (for plotting comparison)
c_values = zeros(size(t_values));
n_values = zeros(size(t_values));
n_values(1) = 0;
for i = 1:length(N_i)
    t_symbol_start = (i - 1) * T;
    t_memory_end = t_symbol_start + memory_length;
    indices = find(t_values > t_symbol_start & t_values <= t_memory_end);
    for j = indices'
        t = t_values(j) - t_symbol_start;
        c_values(j) = c_values(j) + (N_i(i) ./ ((4 * pi * D * t).^(3/2))) .* exp(-distance^2 ./ (4 * D * t));
    end
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
fprintf('Setting up Smoldyn simulation with optimized parameters...\n');
tic;

%% Set Simulation Parameters
numIterations = 1; % Run a single, high-fidelity simulation

% Convert parameters to Smoldyn units
KON = k_on*1e18;
KOFF = k_off;
DIFFMESSENGER = D*1e12;
DIFFRECEPTOR = 0;
DIFFRECEPTORACT = 0;

% Simulation time parameters
START_TIME = 0;
BIT_INTERVAL = T;
STOP_TIME = START_TIME + length(N_i)*BIT_INTERVAL; 
disp(['Smoldyn Simulation Stop Time: ', num2str(STOP_TIME)]);
TIME_STEP = 0.01;
SAMPLING_PERIOD = 1;

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
dirname = sprintf('MG_Optimized_Baseline_%s/', datestr(now,'mm-dd-yyyy--HH-MM-SS'));
mkdir(dirname);
copyfile('MCpointTXsphRX_generated.txt', dirname);

%% Run Smoldyn Simulation
% Note: Using '-w' to visualize, change to '-wt' for faster non-visual run
command = sprintf(['smoldyn %sMCpointTXsphRX_generated.txt -wt -s ', ...
'--define index=%d ', ...
    '--define KON=%d ', ...
    '--define KOFF=%d ', ...
    '--define DIFFMESSENGER=%d ', ...
    '--define DIFFRECEPTOR=%d ', ...
    '--define DIFFRECEPTORACT=%d ', ...
    '--define START_TIME=%d ', ...
    '--define STOP_TIME=%d ', ...
    '--define TIME_STEP=%d ', ...
    '--define BIT_INTERVAL=%d ', ...
    '--define numRECEPTOR=%d ', ...
    '--define TXRXDISTANCE=%d ', ...
    '--define BOUNDARYLENGTH=%d ', ...
    '--define TXPOSITION=%d ', ...
    '--define RXRADIUS=%d ', ...
    '--define RXRECEPTIONSPACETHICKNESS=%d ', ...            
    '--define SAMPLING_PERIOD=%d '], ...
    dirname, 1, KON, KOFF, DIFFMESSENGER, ...
    DIFFRECEPTOR, DIFFRECEPTORACT, START_TIME, STOP_TIME, TIME_STEP, BIT_INTERVAL, numRECEPTOR, ...
    TXRXDISTANCE, BOUNDARYLENGTH, TXPOSITION, RXRADIUS, RXRECEPTIONSPACETHICKNESS, SAMPLING_PERIOD);

fprintf('Executing Smoldyn... This may take a while.\n');
system(command);
fprintf('Smoldyn simulation complete.\n');

%% Read and Plot Simulation Results
allMoleculesFilename = sprintf('%sallmolecules_varNo_2_varValue_1_iter_1.txt', dirname);
data2_temp = importdata(allMoleculesFilename,' ',1);
data2 = data2_temp.data;
Active_rec_fin = data2(:, end) / N;

figure('Position',[400 400 1200 400]);
plot(t_values, n_values, 'LineWidth', 2, 'DisplayName', 'Deterministic Model');
hold on;
plot(linspace(START_TIME, STOP_TIME, length(Active_rec_fin)), Active_rec_fin, '-', 'DisplayName', 'Stochastic (Smoldyn) Run');
grid on;
xlabel('Time (s)');
ylabel('Occupation Fraction <n(t)>');
title('Optimized Parameters: Deterministic vs. Stochastic Response');
legend;

toc;