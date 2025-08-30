% =========================================================================
% NARMA10_Generate_Smoldyn_Uniform.m
%
% DESCRIPTION:
%   Generates Smoldyn data for the NARMA10 task using the best
%   hyperparameters found via Bayesian Optimization when UNIFORM
%   post-processing was active.
% =========================================================================

%% Initialize the MATLAB environment
close all;
clear;
rng(0); % Fix the random sequence

%% 1. Parameter Setup
% Updated with the Bayesian Optimized values for the 'uniform' method.

% Receptor and binding parameters
N = 500;
k_on = 1.9444e-18;     % -- BAYESIAN OPTIMIZED (UNIFORM)
k_off = 6.756;         % -- BAYESIAN OPTIMIZED (UNIFORM)
KD = k_off / k_on;

% Time parameters
T = 0.62553;           % Symbol duration (s) -- BAYESIAN OPTIMIZED (UNIFORM)

% Data lengths (using standard values from optimization script config)
washout1 = 300;
num_train_points = 2000;
wheretostarttest = 2500;
num_test_points  = 2000;
washout2 = wheretostarttest - (washout1 + num_train_points);
num_tot_points = washout1 + washout2 + num_train_points + num_test_points;

% Communication channel parameters
distance = 1.57e-05;     % Distance (m) -- BAYESIAN OPTIMIZED (UNIFORM)
D = 1.9371e-10;        % Diffusion coefficient (m^2/s) -- BAYESIAN OPTIMIZED (UNIFORM)

% Input normalization parameters
N_min = 100;
N_max = 15842;         % -- BAYESIAN OPTIMIZED (UNIFORM)

%% 2. Generate the NARMA10 Time Series
narma_len = num_tot_points + 10 + 1;
u = 0.5 * rand(narma_len, 1);
q = zeros(narma_len, 1);
for n = 10:num_tot_points-1
    q(n+1) = 0.3 * q(n) + 0.05 * q(n) * sum(q(n-9:n)) + 1.5 * u(n-9) * u(n) + 0.1;
end
N_i = N_min + (u - min(u)) / (max(u) - min(u)) * (N_max - N_min);

%% ==========================================================
%                  SMOLDYN SIMULATION SETUP
% ===========================================================
fprintf('Setting up Smoldyn simulation for UNIFORM optimized parameters...\n');
tic;

% Set Simulation Parameters
KON = k_on*1e18; KOFF = k_off; DIFFMESSENGER = D*1e12;
START_TIME = 0; BIT_INTERVAL = T;
STOP_TIME = START_TIME + length(N_i)*BIT_INTERVAL; 
disp(['Smoldyn Simulation Stop Time: ', num2str(STOP_TIME)]);
TIME_STEP = 0.01; numRECEPTOR = N; TXRXDISTANCE = distance*1e6;
BOUNDARYLENGTH = 25; TXPOSITION = -10; RXRADIUS = 3; RXRECEPTIONSPACETHICKNESS = 0.2;

% Create Smoldyn Configuration and Directory
input_time_points = START_TIME:BIT_INTERVAL:STOP_TIME-eps*1e10;
generateSimConfig(N_i, input_time_points);
dirname = sprintf('NARMA_Optimized_Uniform_%s/', datestr(now,'mm-dd-yyyy--HH-MM-SS'));
mkdir(dirname);
copyfile('MCpointTXsphRX_generated.txt', dirname);

% Run Smoldyn Simulation
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