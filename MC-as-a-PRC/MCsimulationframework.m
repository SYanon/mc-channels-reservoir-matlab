% MATLAB script to run Smoldyn simulations and process results

%% Initialization
% Clear the workspace and close all figures
clear
close all

% Start a timer
tic

%% Create Time-Stamped Directory for Simulation Input and Output Files

% Generate a directory name with the current date and time
dirname = sprintf('sample_%s/', datestr(now,'mm-dd-yyyy--HH-MM-SS'));
% Create the directory
mkdir(dirname)

% Copy the Smoldyn execution file to the new directory
status = copyfile('simulation/MCpointTXsphRX.txt', dirname);  % copy Smoldyn execution file
if ~status
    disp('Smoldyn execution file could not be copied to new directory.')
    return
end

%% Set Simulation Parameters

% Number of iterations for Monte Carlo simulation
numIterations = 1; % number of iterations for Monte Carlo

% Reaction rates and diffusion coefficients
KON = 10;               % On rate constant
KOFF = 1;               % Off rate constant
DIFFMESSENGER = 100;    % Diffusion coefficient of messenger molecules
DIFFRECEPTOR = 0;       % Diffusion coefficient of receptors
DIFFRECEPTORACT = 0;    % Diffusion coefficient of activated receptors

% Simulation time parameters
START_TIME = 0;         % Simulation start time
STOP_TIME = 50;         % Simulation stop time
TIME_STEP = 0.01;       % Simulation time step

% Communication parameters
BIT_INTERVAL = 5;           % Time interval between bits
BITZERO_MOLCOUNT = 2000;    % Molecule count for bit zero
BITONE_MOLCOUNT = 5000;     % Molecule count for bit one
BITONE_PROB = 0.5;          % Probability of bit one

% Receptor parameters
numRECEPTOR = 200;          % Number of receptors
ADAPT_OFFSET = 10;          % Adaptation offset

% Spatial parameters
TXRXDISTANCE = 20;          % Distance between transmitter and receiver
BOUNDARYLENGTH = 50;        % Length of the boundary
TXPOSITION = -20;           % Position of the transmitter
RXRADIUS = 3;               % Radius of the receiver

% Phase shift parameters
PHASE_SHIFT_BIT = 10; 
PHASE_SHIFT_TIME = PHASE_SHIFT_BIT * BIT_INTERVAL; % Phase shift time

% Additional parameters
BIT_INTERVAL_NEW = 2;
RXRECEPTIONSPACETHICKNESS = 1; % Thickness of the receiver reception space
SAMPLING_PERIOD = 5;           % Sampling period in terms of time steps

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
    "BITZERO_MOLCOUNT",...               % 10
    "BITONE_MOLCOUNT",...                % 11
    "BITONE_PROB",...                    % 12
    "numRECEPTOR",...                    % 13
    "ADAPT_OFFSET",...                   % 14
    "TXRXDISTANCE",...                   % 15
    "BOUNDARYLENGTH",...                 % 16
    "TXPOSITION",...                     % 17
    "RXRADIUS",...                       % 18
    "RXRECEPTIONSPACETHICKNESS",...      % 19
    "SAMPLING_PERIOD"...                 % 20
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
        command = sprintf(['smoldyn %sMCpointTXsphRX.txt -w ', ...
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
            '--define BITZERO_MOLCOUNT=%d ', ...
            '--define BITONE_MOLCOUNT=%d ', ...
            '--define BITONE_PROB=%d ', ...
            '--define numRECEPTOR=%d ', ...
            '--define ADAPT_OFFSET=%d ', ...
            '--define TXRXDISTANCE=%d ', ...
            '--define BOUNDARYLENGTH=%d ', ...
            '--define TXPOSITION=%d ', ...
            '--define RXRADIUS=%d ', ...
            '--define RXRECEPTIONSPACETHICKNESS=%d ', ...
            '--define SAMPLING_PERIOD=%d ', ...
            '--define PHASE_SHIFT_TIME=%d ', ...
            '--define BIT_INTERVAL_NEW=%d'], ...
            dirname, i, variableNo, variableValues(j), KON, KOFF, DIFFMESSENGER, ...
            DIFFRECEPTOR, DIFFRECEPTORACT, START_TIME, STOP_TIME, TIME_STEP, BIT_INTERVAL, ...
            BITZERO_MOLCOUNT, BITONE_MOLCOUNT, BITONE_PROB, numRECEPTOR, ADAPT_OFFSET, ...
            TXRXDISTANCE, BOUNDARYLENGTH, TXPOSITION, RXRADIUS, RXRECEPTIONSPACETHICKNESS, ...
            SAMPLING_PERIOD, PHASE_SHIFT_TIME, BIT_INTERVAL_NEW);

        % Execute the Smoldyn command
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

        % Read the bit sequence
        bitSequenceFilename = sprintf('%sbitsequence_varNo_%d_varValue_%d_iter_%d.txt', dirname, variableNo, variableValues(j), k);
        data3 = importdata(bitSequenceFilename,' ');
        bitsequences{j, k} = data3(:, 1);
    end
end

% You can access the data later, for example:
% someReceivedSignal = received_signals{1, 1}; % For variableValues(1) and iteration 1

%% Appendix

% (For Mac users) Use the following if MATLAB can't run Smoldyn
% This adds the Smoldyn path to the set of default paths
% setenv('PATH', getenv('PATH')+":/usr/local/bin")
% system('echo $PATH')
