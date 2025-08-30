% =========================================================================
% NewMG_Generate_Smoldyn_Data_MultiRandom.m
%
% DESCRIPTION:
%   Generates Smoldyn data for the Mackey-Glass task using the best
%   hyperparameters found via Bayesian Optimization when MULTI-RANDOM
%   post-processing was active.
% =========================================================================

%% Initialize the MATLAB environment
close all;
clear;

%% 1. Parameter Setup
% Updated with the Bayesian Optimized values for the 'multi-random' method.

% Receptor and binding parameters
N = 500;
k_on = 1.8248e-17;     % -- BAYESIAN OPTIMIZED (MULTI-RANDOM)
k_off = 7.6434;        % -- BAYESIAN OPTIMIZED (MULTI-RANDOM)
KD = k_off / k_on;

% Time parameters
T = 1.5163;            % Symbol duration (s) -- BAYESIAN OPTIMIZED (MULTI-RANDOM)

% Data lengths (using standard values from optimization script config)
washout1 = 500;
num_train_points = 500;
wheretostarttest = 1200;
num_test_points  = 500;
washout2 = wheretostarttest - (washout1 + num_train_points);
num_tot_points = washout1 + washout2 + num_train_points + num_test_points;

% Communication channel parameters
distance = 5.0777e-06;   % Distance (m) -- BAYESIAN OPTIMIZED (MULTI-RANDOM)
D = 1.7926e-10;        % Diffusion coefficient (m^2/s) -- BAYESIAN OPTIMIZED (MULTI-RANDOM)

% Input normalization parameters
N_min = 100;
N_max = 16135;         % -- BAYESIAN OPTIMIZED (MULTI-RANDOM)

% Step size control & Mackey-Glass parameters
dt = 0.001;
memlengthsweep = 100;
Tpeak = distance^2/(6*D);
memory_length = round((memlengthsweep * Tpeak)/T)*T;
offset = 0;
predictlength = 6; % From optimized config

%% --- Load Mackey-Glass Series ---
MG_filename = 'MGseries_RK4_tau17_beta0.20_gamma0.1_n10_len5000_dt1.0.mat';
load(MG_filename, 'mackey_glass_series');
disp('Loaded "mackey_glass_series" from MAT file.');
series_segment = mackey_glass_series(1 : num_tot_points + predictlength);
series_normalized = (series_segment - min(series_segment))/(max(series_segment) - min(series_segment));
input_series  = series_normalized(1:end - predictlength);
N_i = N_min + (input_series - min(input_series)) / (max(input_series) - min(input_series)) * (N_max - N_min);

%% ==========================================================
%                  SMOLDYN SIMULATION SETUP
% ===========================================================
fprintf('Setting up Smoldyn simulation for MULTI-RANDOM optimized parameters...\n');
tic;

%% Set Simulation Parameters
KON = k_on*1e18;
KOFF = k_off;
DIFFMESSENGER = D*1e12;
START_TIME = 0;
BIT_INTERVAL = T;
STOP_TIME = START_TIME + length(N_i)*BIT_INTERVAL; 
disp(['Smoldyn Simulation Stop Time: ', num2str(STOP_TIME)]);
TIME_STEP = 0.01;
numRECEPTOR = N;
TXRXDISTANCE = distance*1e6;
BOUNDARYLENGTH = 25;
TXPOSITION = -10;
RXRADIUS = 3;
RXRECEPTIONSPACETHICKNESS = 0.2;

%% Create Smoldyn Configuration and Directory
input_time_points = START_TIME:BIT_INTERVAL:STOP_TIME-eps*1e10;
generateSimConfig(N_i, input_time_points);
dirname = sprintf('MG_Optimized_MultiRandom_%s/', datestr(now,'mm-dd-yyyy--HH-MM-SS'));
mkdir(dirname);
copyfile('MCpointTXsphRX_generated.txt', dirname);

%% Run Smoldyn Simulation
command = sprintf(['smoldyn %sMCpointTXsphRX_generated.txt -wt -s ', ...
'--define index=%d --define KON=%d --define KOFF=%d --define DIFFMESSENGER=%d ', ...
'--define START_TIME=%d --define STOP_TIME=%d --define TIME_STEP=%d --define BIT_INTERVAL=%d ', ...
'--define numRECEPTOR=%d --define TXRXDISTANCE=%d --define BOUNDARYLENGTH=%d --define TXPOSITION=%d ', ...
'--define RXRADIUS=%d --define RXRECEPTIONSPACETHICKNESS=%d --define DIFFRECEPTOR=0 --define DIFFRECEPTORACT=0 --define SAMPLING_PERIOD=1'], ...
    dirname, 1, KON, KOFF, DIFFMESSENGER, START_TIME, STOP_TIME, TIME_STEP, BIT_INTERVAL, numRECEPTOR, ...
    TXRXDISTANCE, BOUNDARYLENGTH, TXPOSITION, RXRADIUS, RXRECEPTIONSPACETHICKNESS);

fprintf('Executing Smoldyn... This may take a while.\n');
system(command);
fprintf('Smoldyn simulation complete.\n');
toc;