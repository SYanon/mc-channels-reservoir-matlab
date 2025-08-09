%%
% Code to generate Smoldyn data given the MC system settings and input
% sequence (in this case MG Series generated beforehand)
%%


%% Initialize the MATLAB environment
close all   % Close all open figure windows
clear       % Clear all variables from the workspace

movmean_on = 0; % = 0 by default
movmean_window = 500; % only meaningful if movmean_on = 1

%% 1. Parameter Setup
% Receptor and binding parameters
N = 500;                % Total number of receptors at the receiver
k_on = 1e-18;          % Binding rate constant (in 1/(M*s))
k_off = 1;            % Unbinding rate constant (in 1/s)
KD = k_off / k_on;   % Dissociation constant

% Time parameters
T = 1;                   % Symbol duration (seconds)

washout1 = 100;          % Number of symbols to wash out before training
num_train_points = 2000;  % Number of symbols used for training % CHECK!!!
wheretostarttest = 2400;
washout2 = wheretostarttest-(washout1+num_train_points);          % Number of symbols to wash out before testing
num_test_points = 2500;   % Number of symbols used for testing


num_tot_points = washout1 + washout2 + num_train_points + num_test_points;  % Total number of symbols

% Communication channel parameters
distance = 10e-6;        % Distance between transmitter and receiver (10 microns)
D = 1e-11;               % Diffusion coefficient (in m^2/s)

% Input normalization parameters
N_min = 100;            % Minimum value for input normalization
N_max = 3000;           % Maximum value for input normalization


% Step size control
dt = 0.001;  % Time step for deterministic/numerical analysis %fixed
memlengthsweep = 100;
Tpeak = distance^2/(6*D);
memory_length = round((memlengthsweep * Tpeak)/T)*T; % memory of MC channel
offset = 0;  % if you have an offset param, update above

% Mackey-Glass time series parameters
beta = 0.2;              % Parameter beta in Mackey-Glass equation
tau = 17;                % Time delay
n = 10;                  % Nonlinearity parameter
gamma = 0.1;             % Parameter gamma in Mackey-Glass equation
dtMG = 1;                % Time step for Mackey-Glass equation
predictlength = 12;      % Prediction length (number of steps ahead)

%% --- Generate or Load Mackey-Glass Series ---

% tt_end = (num_tot_points + predictlength + tau - 1) * dtMG;
% tt = 0:dtMG:tt_end;

MG_filename = 'MGseries_RK4_tau17_beta0.20_gamma0.1_n10_len5000_dt1.0.mat';
load(MG_filename, 'mackey_glass_series');
disp('Loaded "mackey_glass_series" from MAT file.');
if length(mackey_glass_series) < num_tot_points + predictlength
    error('Loaded series is too short. Increase MG_length or reduce num_tot_points.');
end

mackey_glass_series_non_norm = mackey_glass_series(1 : num_tot_points + predictlength);
mg_min = min(mackey_glass_series_non_norm);
mg_max = max(mackey_glass_series_non_norm);
mackey_glass_series = (mackey_glass_series_non_norm - mg_min)/(mg_max - mg_min);

figure;
plot(mackey_glass_series, 'LineWidth',1.5);
title('Loaded Mackey-Glass Series (First num\_tot\_points Samples)');
xlabel('Index (k)');
ylabel('MG value (normalized)');
grid on;

%% 4. Compute Input Molecule Counts
input_series  = mackey_glass_series(1:end - predictlength);
% Normalize input series to obtain molecule counts N_i
N_i = N_min + (input_series - min(input_series)) / (max(input_series) - min(input_series)) * (N_max - N_min);

figure;
plot(N_i, 'k-', 'LineWidth', 1.5);
title('N_i');


%% 6. Generate Time Vector
t_total = length(N_i) * T;      % Total simulation time
t_values = 0:dt:t_total;           % Time vector from 0 to t_total with step dt

% Number of time steps per symbol duration
steps_per_symbol = T / dt;

%% 7. Pre-allocate Concentration and Receptor Occupation Vectors
c_values = zeros(size(t_values));  % Pre-allocate concentration vector
n_values = zeros(size(t_values));  % Pre-allocate receptor occupation vector

% Initial condition for receptor occupation
n_values(1) = 0;  % No bound receptors at time t = 0

%% 8. Compute Ligand Concentration c(t) at the Receiver
tic  % Start timing
% Compute concentration c(t) considering memory effect
%for i = 1:num_tot_points - predictlength
for i = 1:length(N_i)
    
    t_symbol_start = (i - 1) * T;            % Start time of current symbol
    t_memory_end = t_symbol_start + memory_length;  % End time of memory window

    % Find indices within the memory window
    indices = find(t_values > t_symbol_start & t_values <= t_memory_end);

    % Calculate the concentration due to this symbol within the memory window
    for j = indices'
        t = t_values(j) - t_symbol_start;  % Time since symbol transmission
        c_values(j) = c_values(j) + (N_i(i) ./ ((4 * pi * D * t).^(3/2))) .* ...
                      exp(-distance^2 ./ (4 * D * t));
    end
end
toc  % Stop timing

%% 9. Simulate Receptor Binding Process Using Euler's Method
% Simulate receptor dynamics over time
for i = 2:length(t_values)
    c_t = c_values(i - 1);  % Concentration at previous time step
    n_t = n_values(i - 1);  % Number of bound receptors at previous time step

    % Differential equation for receptor binding dynamics
    dn_dt = k_on * (N - n_t) * c_t - k_off * n_t;

    % Update the number of bound receptors
    n_values(i) = n_t + dn_dt * dt;
end

% Normalize receptor occupation to obtain average occupation ratio [0, 1]
n_values = n_values / N;

% Plot the average number of bound receptors over time
figure('Position',[400 400 1200 400]);  % [left bottom width height]
plot(t_values, n_values, 'LineWidth', 2);
xlabel('Time (s)');
ylabel('<n(t)> - Average Number of Bound Receptors');
title('Average Number of Bound Receptors vs. Time');
grid on;

% Plot the ligand concentration c(t) over time at the receiver
figure('Position',[400 400 1200 400]);  % [left bottom width height]
plot(t_values, c_values, 'LineWidth', 2);
xlabel('Time (s)');
ylabel('c(t) - Ligand Concentration');
title('Ligand Concentration vs. Time at the Receiver');
grid on;


%% SMOLDYN

%% Initialization
% Start a timer
tic

%% Set Simulation Parameters

% Number of iterations for Monte Carlo simulation
numIterations = 1; % number of iterations for Monte Carlo


% Reaction rates and diffusion coefficients
KON = k_on*1e18;               % On rate constant
KOFF = k_off;             % Off rate constant
DIFFMESSENGER = D*1e12;     % Diffusion coefficient of messenger molecules
DIFFRECEPTOR = 0;       % Diffusion coefficient of receptors
DIFFRECEPTORACT = 0;    % Diffusion coefficient of activated receptors

% Simulation time parameters
START_TIME = 0;         % Simulation start time

BIT_INTERVAL = T;           % Time interval between bits

%STOP_TIME = t_end;        % Simulation stop time
STOP_TIME = START_TIME + length(N_i)*BIT_INTERVAL; 

disp(['Smoldyn Simulation Stop Time: ', num2str(STOP_TIME)]);



TIME_STEP = 0.01;       % Simulation time step

SAMPLING_PERIOD = 1;           % Sampling period in terms of time steps

%Nres

% Communication parameters


% Receptor parameters
numRECEPTOR = N;          % Number of receptors

% Spatial parameters
TXRXDISTANCE = distance*1e6;          % Distance between transmitter and receiver
BOUNDARYLENGTH = 25;       % Length of the boundary
TXPOSITION = -10;           % Position of the transmitter
RXRADIUS = 3;               % Radius of the receiver
RXRECEPTIONSPACETHICKNESS = 0.2; % Thickness of the receiver reception space




%% Create Smoldyn Configuration
% This creates the .txt smoldyn configuration file that will generate MG
% series in accordance with the parametetrs above.


input_time_points = START_TIME:BIT_INTERVAL:STOP_TIME-eps*1e10;

generateSimConfig(N_i, input_time_points)



%% Create Time-Stamped Directory for Simulation Input and Output Files

% Generate a directory name with the current date and time
dirname = sprintf('sample_%s/', datestr(now,'mm-dd-yyyy--HH-MM-SS'));
% Create the directory
mkdir(dirname)

% Copy the Smoldyn execution file to the new directory
status = copyfile('MCpointTXsphRX_generated.txt', dirname);  % copy Smoldyn execution file
if ~status
    disp('Smoldyn execution file could not be copied to new directory.')
    return
end
%% Define Variables for Parameter Sweep

% Create an array of variable names for parameters
variableNameArray = [...
    "KON",...                            % 1
    "KOFF",...                           % 2
    "DIFFMESSENGER",...                  % 3
    "DIFFRECEPTOR",...                   % 4
    "DIFFRECEPTORACT",...                % 5
    "START_TIME",...                     % 6
    "STOP_TIME",...                      % 7
    "TIME_STEP",...                      % 8
    "BIT_INTERVAL",...                   % 9
    "numRECEPTOR",...                    % 10
    "TXRXDISTANCE",...                   % 11
    "BOUNDARYLENGTH",...                 % 12
    "TXPOSITION",...                     % 13
    "RXRADIUS",...                       % 14
    "RXRECEPTIONSPACETHICKNESS",...      % 15    
    "SAMPLING_PERIOD"...                 % 16
    ];

variableNo = 2;           % Choose which parameter to sweep (index in variableNameArray)
variableValues = KOFF;    % Specify the parameter values to simulate

%% Run Smoldyn Simulations

for j = 1:numel(variableValues)
    % Assign the current variable value to the parameter being swept
    assignin('base', variableNameArray(variableNo), variableValues(j));
    for i = 1:numIterations      
        % Build the command to run Smoldyn with the current parameters
        % If you want to visualize the simulation, replace -wt with -w 
        % (but it becomes much more computationally demanding)
        command = sprintf(['smoldyn %sMCpointTXsphRX_generated.txt -wt -s ', ...
        '--define index=%d ', ...
            '--define variableNo=%d ', ...
            '--define variableValue=%d ', ...
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
            dirname, i, variableNo, variableValues(j), KON, KOFF, DIFFMESSENGER, ...
            DIFFRECEPTOR, DIFFRECEPTORACT, START_TIME, STOP_TIME, TIME_STEP, BIT_INTERVAL, numRECEPTOR, ...
            TXRXDISTANCE, BOUNDARYLENGTH, TXPOSITION, RXRADIUS, RXRECEPTIONSPACETHICKNESS, SAMPLING_PERIOD);

        system(command);
        
    end
end

%% Read Simulation Results

% Initialize cell arrays to store the data
received_signals = cell(numel(variableValues), numIterations);
times1 = cell(numel(variableValues), numIterations);
ReceptorActives = cell(numel(variableValues), numIterations);
times2 = cell(numel(variableValues), numIterations);
bitsequences = cell(numel(variableValues), numIterations);

for j = 1:numel(variableValues)
    % Assign the current variable value to the parameter being swept
    assignin('base', variableNameArray(variableNo), variableValues(j));

    for k = 1:numIterations
        fprintf('varName = %s -- varValue = %d -- numIter = %d \n', variableNameArray(variableNo), variableValues(j), k);

        % Read the received signal data
        receivedSignalFilename = sprintf('%sreceivedsignal_varNo_%d_varValue_%d_iter_%d.txt', dirname, variableNo, variableValues(j), k);
        data1 = importdata(receivedSignalFilename,' ');
        received_signals{j, k} = data1(:, 3);   % The third column is the received signal
        times1{j, k} = data1(:, 1);             % The first column is time

        % Read all molecules data
        allMoleculesFilename = sprintf('%sallmolecules_varNo_%d_varValue_%d_iter_%d.txt', dirname, variableNo, variableValues(j), k);
        data2_temp = importdata(allMoleculesFilename,' ',1); % Skip the first line (header)
        data2 = data2_temp.data;
        times2{j, k} = data2(:, 1);             % Time data
        ReceptorActives{j, k} = data2(:, end);  % The last column is ReceptorActive count

    end
end

%%
% You can access the data later, for example:
received_signal_fin = received_signals{1, 1}; % For variableValues(1) and iteration 1
Active_rec_fin = ReceptorActives{1,1}/N;


if movmean_on == 1 
%Active_rec_fin = movmean(Active_rec_fin,[movmean_window 0]); % average over only past values
Active_rec_fin = movmean(Active_rec_fin,movmean_window); % centered average
end


time_axis_smoldyn = 0:TIME_STEP*SAMPLING_PERIOD:length(Active_rec_fin)*TIME_STEP*SAMPLING_PERIOD-1;


figure('Position',[400 400 1200 400]);  % [left bottom width height]
plot(t_values, n_values, 'LineWidth', 2);
hold on
plot(linspace(START_TIME, STOP_TIME, length(Active_rec_fin)), Active_rec_fin,"-")
grid on;
xlabel('Time (s)');
ylabel('<n(t)> - Average Percentage of Bound Receptors');
title('Percentage of bound receptors - Numerical vs. Smoldyn');


figure('Position',[400 400 1200 400]);  % [left bottom width height]
plot(linspace(START_TIME, STOP_TIME, length(received_signal_fin)), received_signal_fin,"-")
grid on;
xlabel('Time (s)');
ylabel('Received signal (number of ligands in virtual space) in Smoldyn');




toc


%%


%% Appendix

% (For Mac users) Use the following if MATLAB can't run Smoldyn
% This adds the Smoldyn path to the set of default paths
% setenv('PATH', getenv('PATH')+":/usr/local/bin")
% system('echo $PATH')


