%% Initialize the MATLAB environment
close all   % Close all open figure windows
clear       % Clear all variables from the workspace


smold = 1; % enable cases where new Smoldyn sim is run, or Smoldyn data is read.
smold_only_data = 1; % only Smoldyn data is read without running a new sim, thereby requiring datadirname for smoldyn data
datadirname = 'MG_5000_T1_koff1'; % only meaningful if smold = 1 && smold_only_data = 1

movmean_on = 0; % = 0 by default
movmean_window = 2000; % only meaningful if movmean_on = 1 % SHOULD RUN ONLY OVER THE PAST VALUES. 
memorywindowlength = 1; % = 1 by default
movmean_readout_test = 0; % = 0 by default
movmean_readout_test_window = 5; % only meaningful if movmean_readout_test = 1

plot_in_loop = false;   % set to true to see intermediate loop figures


%% Primary & Secondary Swept Parameters
param1_name   = 'movmean_window';      % e.g. 'Nres_init', 'num_train_points', 'T', 'predictlength', ...
param1_values = [1 100 200 500 1000 2000];

param2_name   = 'predictlength';   % e.g. 'Nres_init', 'num_train_points', 'T', 'predictlength', ...
%param2_values = [1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 50 60 70 80 90 100];
param2_values = [1 2 3 5 10 20 50 100];


%% 1. Parameter Setup
% Receptor and binding parameters
N = 500;                % Total number of receptors at the receiver
k_on = 1e-18;          % Binding rate constant (in 1/(M*s))
k_off = 1;            % Unbinding rate constant (in 1/s)
KD = k_off / k_on;   % Dissociation constant

% Time parameters
T = 1;                   % Symbol duration (seconds)
predictlength = 12;      % Prediction length (number of steps ahead)


% Data lengths
washout1 = 250;          % Number of symbols to wash out before training
num_train_points = 2000;  % Number of symbols used for training 
wheretostarttest = 2500;
washout2 = wheretostarttest-(washout1+num_train_points);          % Number of symbols to wash out before testing
num_test_points = 2000;   % Number of symbols used for testing


num_tot_points = washout1 + washout2 + num_train_points + num_test_points;  % Total number of symbols

% Communication channel parameters
distance = 10e-6;        % Distance between transmitter and receiver (10 microns)
D = 1e-11;               % Diffusion coefficient (in m^2/s)

% Input normalization parameters
N_min = 100;            % Minimum value for input normalization
N_max = 3000;           % Maximum value for input normalization

% Regularization and reservoir parameters
lambda = 1e-6;          % Regularization parameter for training
lambda_smol = 1e-6;     % Regularization parameter for training for Smoldyn data

Nres_init = 100;               % Number of reservoir nodes
Nres = Nres_init * memorywindowlength;

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


%% 2D arrays to store results:
nP1 = length(param1_values);
nP2 = length(param2_values);

nrmse_train_NUM_array                              = zeros(nP1, nP2);
nrmse_test_NUM_array                               = zeros(nP1, nP2);
nrmse_train_SMOL_array                              = zeros(nP1, nP2);
nrmse_test_SMOL_array                               = zeros(nP1, nP2);

% ---- NEW array to hold correlation time from the reservoir samples ----

%% Automatic log-scale checks
minP1   = min(param1_values);
maxP1   = max(param1_values);
ratioP1 = maxP1 / minP1;
useLogX = (minP1>0) && (ratioP1>50);

minP2   = min(param2_values);
maxP2   = max(param2_values);
ratioP2 = maxP2 / minP2;
useLogY = (minP2>0) && (ratioP2>50);


for i2 = 1:nP2
    val2 = param2_values(i2);

    % Assign param2 (outer loop) to correct variable
    switch param2_name
        case 'washout1'
            washout1 = val2;
        case 'num_train_points'
            num_train_points = val2;
        case 'wheretostarttest'
            wheretostarttest = val2;
            washout2 = wheretostarttest - (washout1 + num_train_points);
        case 'washout2'
            washout2 = val2;
        case 'num_test_points'
            num_test_points = val2;
        case 'offset'
            offset = val2;
        case 'lambda'
            lambda = val2;
        case 'lambda_smol'
            lambda_smol = val2;
        case 'movmean_window'
            movmean_window = val2;  
        case 'movmean_readout_test_window'
            movmean_readout_test_window = val2;  
        case 'Nres_init'
            Nres_init = val2;
            Nres = Nres_init * memorywindowlength;
        case 'memorywindowlength'
            memorywindowlength = val2;
            Nres = Nres_init * memorywindowlength;
        case 'memlengthsweep'
            memlengthsweep = val2;
            memory_length = round((memlengthsweep * Tpeak)/T)*T;
        case 'predictlength'
            predictlength = val2;
        otherwise
            error('Parameter2 name (%s) not recognized.', param2_name);
    end

    for i1 = 1:nP1
        val1 = param1_values(i1);

        tic
        % Assign param1 (inner loop) to correct variable
        switch param1_name
            case 'washout1'
                washout1 = val1;
            case 'num_train_points'
                num_train_points = val1;
            case 'wheretostarttest'
                wheretostarttest = val1;
                washout2 = wheretostarttest - (washout1 + num_train_points);
            case 'washout2'
                washout2 = val1;
            case 'num_test_points'
                num_test_points = val1;
            case 'offset'
                offset = val1;
            case 'lambda'
                lambda = val1;
            case 'lambda_smol'
                lambda_smol = val1;
            case 'movmean_window'
                movmean_window = val1;  
            case 'movmean_readout_test_window'
                movmean_readout_test_window = val1;                  
            case 'Nres_init'
                Nres_init = val1;
                Nres = Nres_init * memorywindowlength;
            case 'memorywindowlength'
                memorywindowlength = val1;
                Nres = Nres_init * memorywindowlength;
            case 'memlengthsweep'
                memlengthsweep = val1;
                memory_length = round((memlengthsweep * Tpeak)/T)*T;
            case 'predictlength'
                predictlength = val1;
            otherwise
                error('Parameter1 name (%s) not recognized.', param1_name);
        end


        num_tot_points = washout1 + washout2 + num_train_points + num_test_points;




        %% --- Generate or Load Mackey-Glass Series ---

        MG_filename = 'MGseries_RK4_tau17_beta0.20_gamma0.1_n10_len5000_dt1.0.mat';
        load(MG_filename, 'mackey_glass_series');
        disp('Loaded "mackey_glass_series" from MAT file.');
        if length(mackey_glass_series) < num_tot_points + predictlength
            error('Loaded series is too short. Increase MG_length or reduce num_tot_points.');
        end
        
        mackey_glass_series = mackey_glass_series(1:num_tot_points + predictlength);

        if plot_in_loop
            figure;
            plot(mackey_glass_series, 'LineWidth',1.5);
            title('Loaded Mackey-Glass Series (First num\_tot\_points Samples)');
            xlabel('Index (k)');
            ylabel('MG value (normalized)');
            grid on;
        end

        % Training/Testing sets
        input_series  = mackey_glass_series(1:end - predictlength);
        target_series = mackey_glass_series((predictlength+1):end);

        u_train = input_series(washout1+1 : washout1 + num_train_points);
        y_train = target_series(washout1+1 : washout1 + num_train_points);

        u_test = input_series(washout1 + num_train_points + washout2 + 1 : end);
        y_test = target_series(washout1 + num_train_points + washout2 + 1 : end);


        %% 4. Compute Input Molecule Counts
        % Normalize input series to obtain molecule counts N_i
        N_i = N_min + (input_series - min(input_series)) / (max(input_series) - min(input_series)) * (N_max - N_min);

        if plot_in_loop
            figure;
            plot(N_i, 'k-', 'LineWidth', 1.5);
            title('N_i');
        end

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

        if plot_in_loop
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
        end

        %% 10. Training Phase
        % Initialize matrix to store sampled values for training
        sampled_matrix = zeros(Nres, num_train_points);

        sample_indices_to_be_deleted = [];

        % Sampling the number of bound receptors for the training set
        for i = washout1 + 1 : washout1 + num_train_points
            t_symbol_start = (i - 1) * T + offset;  % Start time with offset
            start_idx = find(t_values >= t_symbol_start, 1);  % Find corresponding index

            % Generate indices for sampling within the symbol duration
            sample_indices = start_idx - (memorywindowlength-1) * steps_per_symbol + (0:Nres - 1) * (steps_per_symbol / Nres) * memorywindowlength; % NEW2

            % Ensure indices are within bounds
            sample_indices(sample_indices > length(t_values)) = [];

            if length(sample_indices) >= Nres % NEW - offset control - eliminate the responses that are partially complete due to offset shift
                sampled_matrix(:, i - washout1) = n_values(sample_indices(1:Nres));
            else
                %sampled_matrix(:, i - washout1) = [];
                sample_indices_to_be_deleted = [sample_indices_to_be_deleted i - washout1];

            end

        end

        sampled_matrix(:, sample_indices_to_be_deleted) = [];
        
        % % Plot sample receptor occupation profiles from the training set
        % figure;
        % subplot(2, 2, 1);
        % plot(sampled_matrix(:, 1));
        % title('Sample 1');
        % subplot(2, 2, 2);
        % plot(sampled_matrix(:, floor(end * 0.4)));
        % %plot(sampled_matrix(1:end-1, 2));
        % title('Sample 40%');
        % subplot(2, 2, 3);
        % plot(sampled_matrix(:, floor(end * 0.8)));
        % %plot(sampled_matrix(1:end-1, 3));
        % title('Sample 80%');
        % subplot(2, 2, 4);
        % plot(sampled_matrix(1:end-1, end));
        % title('Sample End');


        sampled_matrix = [sampled_matrix; ones(1, size(sampled_matrix, 2))];

        y_train = y_train(:);
        y_tilda = y_train(1:size(sampled_matrix,2)); % offset control


        % Ensure dimensions match for regression
        if size(sampled_matrix, 2) ~= length(y_tilda)
            error('Dimension mismatch: Check the dimensions of sampled_matrix and y_tilda.');
        end

        % Transpose sampled_matrix for regression
        sampled_matrix = sampled_matrix';

        W_out = pinv(sampled_matrix' * sampled_matrix + lambda * eye(size(sampled_matrix, 2))) ...
            * (sampled_matrix' * y_tilda);

        % Display the calculated weights
        %disp('Calculated Weights W_out:');
        %disp(W_out);

        % Calculate the estimated output y_train_hat for training data
        y_train_hat = sampled_matrix * W_out;

        if plot_in_loop
            % Plot the original and estimated target values
            figure('Position',[400 400 1200 400]);  % [left bottom width height]
            plot(y_train, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);  % Original target values
            hold on;
            plot(y_train_hat, 'r-', 'LineWidth', 2);                % Estimated target values
            xlabel('Index');
            ylabel('Target and Estimated Values');
            title('Training Data: Original and Estimated Targets');
            legend('Original Target', 'Estimated Target');
            grid on;
            set(gca, 'XTick', []);
        end

        %% 11. Compute NRMSE for Training Data
        % Ensure y_train_hat and y_train have the same size
        y_train = y_train(:);         % Ensure column vector
        y_train_hat = y_train_hat(:); % Ensure column vector


        % Check dimensions
        if length(y_train_hat) ~= length(y_train)
            error('The vectors y_train_hat and y_train must be the same length.');
        end

        % Calculate RMSE and NRMSE
        rmse_train = sqrt(mean((y_train_hat - y_train).^2));
        mse_train = mean((y_train_hat - y_train).^2);
        std_y_train = std(y_train);
        mean_y_train = mean(y_train);
        var_y_train = mean((y_train - mean_y_train).^2);

        nrmse_train = rmse_train / std_y_train;
        nmse_train = mse_train / var_y_train;

        % Display result
        %disp(['NRMSE (Training): ', num2str(nrmse_train)]);
        %disp(['NMSE (Training): ', num2str(nmse_train)]);

        %% 12. Testing Phase
        % Initialize matrix to store sampled values for testing
        sampled_matrix_test = zeros(Nres, num_test_points);
        sample_indices_to_be_deleted = [];

        % Sampling the number of bound receptors for the testing set
        for i = washout1 + num_train_points + washout2 + 1 : num_tot_points
            t_symbol_start = (i - 1) * T + offset;  % Start time with offset
            start_idx = find(t_values >= t_symbol_start, 1);  % Find corresponding index

            % Generate indices for sampling within the symbol duration
            sample_indices = start_idx - (memorywindowlength-1) * steps_per_symbol + (0:Nres - 1) * (steps_per_symbol / Nres) * memorywindowlength;

            % Ensure indices are within bounds
            sample_indices(sample_indices > length(t_values)) = [];

            if length(sample_indices) >= Nres % NEW - offset control - eliminate the responses that are partially complete due to offset shift
                sampled_matrix_test(:, i - (washout1 + num_train_points + washout2)) = n_values(sample_indices(1:Nres));
            else
                %sampled_matrix_test(:, i - (washout1 + num_train_points + washout2)) = [];
                sample_indices_to_be_deleted = [sample_indices_to_be_deleted i - (washout1 + num_train_points + washout2)];
            end
        end

        sampled_matrix_test(:, sample_indices_to_be_deleted) = [];


        % % Plot sample receptor occupation profiles from the testing set
        % figure;
        % subplot(2, 2, 1);
        % plot(sampled_matrix_test(:, 1));
        % title('Sample 1');
        % subplot(2, 2, 2);
        % plot(sampled_matrix_test(:, floor(end * 0.4)));
        % title('Sample 40%');
        % subplot(2, 2, 3);
        % plot(sampled_matrix_test(:, floor(end * 0.8)));
        % title('Sample 80%');
        % subplot(2, 2, 4);
        % plot(sampled_matrix_test(:, end));
        % title('Sample End');

        sampled_matrix_test = [sampled_matrix_test; ones(1, size(sampled_matrix_test, 2))];

        % Transpose for computation
        sampled_matrix_test = sampled_matrix_test';

        % Calculate the estimated output y_test_hat for testing data
        y_test_hat = sampled_matrix_test * W_out;

        %% 13. Compute NRMSE for Testing Data
        % Prepare target output for testing
        y_segment_test = y_test;  % Adjust length to match y_test_hat

        % Ensure column vectors
        y_segment_test = y_segment_test(:);
        y_test_hat = y_test_hat(:);

        y_segment_test = y_segment_test(1:length(y_test_hat)); % offset control


        % Check dimensions
        if length(y_test_hat) ~= length(y_segment_test)
            error('The vectors y_test_hat and y_segment_test must be the same length.');
        end

        % Calculate RMSE, NRMSE, and NMSE
        rmse_test = sqrt(mean((y_test_hat - y_segment_test).^2));
        mse_test = mean((y_test_hat - y_segment_test).^2);
        std_y_test = std(y_segment_test);
        mean_y_test = mean(y_segment_test);
        var_y_test = mean((y_segment_test - mean_y_test).^2);

        nrmse_test = rmse_test / std_y_test;
        nmse_test = mse_test / var_y_test;

        % Display results
        %disp(['NRMSE (Test): ', num2str(nrmse_test)]);
        %disp(['NMSE (Test): ', num2str(nmse_test)]);


        if plot_in_loop
            % Plot the original and estimated target values for testing data
            figure('Position',[400 400 1200 400]);  % [left bottom width height]
            plot(y_segment_test, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);  % Original target values
            hold on;
            plot(y_test_hat, 'r-', 'LineWidth', 2);                        % Estimated target values
            xlabel('Index');
            ylabel('Target and Estimated Values');
            title('Testing Data: Original and Estimated Targets');
            legend('Original Target', 'Estimated Target');
            grid on;
            set(gca, 'XTick', []);
        end

        %% --- Save results ---
        nrmse_train_NUM_array(i1, i2) = nrmse_train;
        nrmse_test_NUM_array(i1, i2)  = nrmse_test;

        fprintf('%s=%g, %s=%g -> NRMSE-NUM(Tr)=%.3g, NRMSE-NUM(Te)=%.3g\n',...
            param1_name, val1, param2_name, val2, nrmse_train, nrmse_test);

        if smold == 0
            continue
        end


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
        STOP_TIME = START_TIME + length(N_i)*BIT_INTERVAL;
        %disp(['Smoldyn Simulation Stop Time: ', num2str(STOP_TIME)]);
        TIME_STEP = 0.01;       % Simulation time step
        SAMPLING_PERIOD = 1;           % Sampling period in terms of time steps

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
        %generateSimConfig2(N_i, input_time_points)


        %% Create Time-Stamped Directory for Simulation Input and Output Files

        % Generate a directory name with the current date and time
        dirname = sprintf('sample_%s/', datestr(now,'mm-dd-yyyy--HH-MM-SS'));
        % Create the directory
        %mkdir(dirname)

        % Copy the Smoldyn execution file to the new directory
        %status = copyfile('MCpointTXsphRX_generated.txt', dirname);  % copy Smoldyn execution file
        % if ~status
        %     disp('Smoldyn execution file could not be copied to new directory.')
        %     return
        % end
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

                % Execute the Smoldyn command
                if smold_only_data ~= 1
                    system(command);
                end
            end
        end

        %% Read Simulation Results

        if smold_only_data == 1
            dirname = sprintf('%s/', datadirname); % CHECK!
        end

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

                % % Read the bit sequence
                % bitSequenceFilename = sprintf('%sbitsequence_varNo_%d_varValue_%d_iter_%d.txt', dirname, variableNo, variableValues(j), k);
                % data3 = importdata(bitSequenceFilename,' ');
                % bitsequences{j, k} = data3(:, 1);
            end
        end

        %%
        % You can access the data later, for example:
        received_signal_fin = received_signals{1, 1}; % For variableValues(1) and iteration 1
        Active_rec_fin = ReceptorActives{1,1}/N;
       

        if movmean_on == 1
            Active_rec_fin = movmean(Active_rec_fin,[movmean_window 0]); % average over only past values
            %Active_rec_fin = movmean(Active_rec_fin,movmean_window); % centered average
        end


        %time_axis_smoldyn = 0:TIME_STEP*SAMPLING_PERIOD:length(Active_rec_fin)*TIME_STEP*SAMPLING_PERIOD-1;

        if plot_in_loop
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
        end

        toc


        %% TRAINING - SMOLDYN DATA

        sampled_matrix_smol = zeros(Nres, num_train_points);

        time_index_smol = START_TIME:TIME_STEP*SAMPLING_PERIOD:STOP_TIME;
        %time_index_smol = time_index_smol(1:end-1);

        steps_per_symbol_smol = T/TIME_STEP;

        sample_indices_to_be_deleted = [];

        for i = washout1 + 1 : washout1 + num_train_points
            t_symbol_start_smol = (i - 1) * T + offset;
            start_idx = find(time_index_smol>= t_symbol_start_smol, 1);  % Find corresponding index

            sample_indices_smol = start_idx  - (memorywindowlength-1) * steps_per_symbol_smol + (0:Nres - 1) * (steps_per_symbol_smol / Nres) * memorywindowlength;
            sample_indices_smol(sample_indices_smol > length(time_index_smol)) = [];

            if length(sample_indices_smol) >= Nres % NEW - offset control - eliminate the responses that are partially complete due to offset shift
                sampled_matrix_smol(:, i - washout1) = Active_rec_fin(sample_indices_smol(1:Nres));
            else
                %sampled_matrix_smol(:, i - (washout1 + num_train_points + washout2)) = [];
                sample_indices_to_be_deleted = [sample_indices_to_be_deleted i - washout1];

            end

        end

        sampled_matrix_smol(:, sample_indices_to_be_deleted) = [];


        % % Plot sample receptor occupation profiles from the training set
        % figure;
        % subplot(2, 2, 1);
        % plot(sampled_matrix_smol(:, 1));
        % title('Sample 1');
        % subplot(2, 2, 2);
        % plot(sampled_matrix_smol(:, floor(end * 0.4)));
        % title('Sample 40%');
        % subplot(2, 2, 3);
        % plot(sampled_matrix_smol(:, floor(end * 0.8)));
        % title('Sample 80%');
        % subplot(2, 2, 4);
        % plot(sampled_matrix_smol(:, end));
        % title('Sample End');

        sampled_matrix_smol = [sampled_matrix_smol; ones(1, size(sampled_matrix_smol, 2))]; 

        y_tilda = y_train(1:size(sampled_matrix_smol,2)); % offset control

        % Ensure dimensions match for regression
        if size(sampled_matrix_smol, 2) ~= length(y_tilda)
            error('Dimension mismatch: Check the dimensions of sampled_matrix and y_tilda.');
        end

        % Transpose sampled_matrix for regression
        sampled_matrix_smol = sampled_matrix_smol';


        % Compute the regularized linear regression weights W_out
        W_out_smol = pinv(sampled_matrix_smol' * sampled_matrix_smol + lambda_smol * eye(size(sampled_matrix_smol, 2))) ...
                * (sampled_matrix_smol' * y_tilda);
        
        % Display the calculated weights
        %disp('Calculated Weights W_out_smol:');
        %disp(W_out_smol);

        % Calculate the estimated output y_train_hat for training data
        y_train_hat_smol = sampled_matrix_smol * W_out_smol;

        if plot_in_loop
            % Plot the original and estimated target values
            figure('Position',[400 400 1200 400]);  % [left bottom width height]
            plot(y_train, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);  % Original target values
            hold on;
            plot(y_train_hat_smol, 'r-', 'LineWidth', 2);                % Estimated target values
            xlabel('Index');
            ylabel('Target and Estimated Values');
            title('Training Data: Original and Estimated Targets - Smoldyn');
            legend('Original Target', 'Estimated Target');
            grid on;
            set(gca, 'XTick', []);
        end

        % Compute NRMSE for Training Data - Smoldyn
        % Ensure y_train_hat and y_train have the same size
        y_train = y_train(:);         % Ensure column vector
        y_train_hat_smol = y_train_hat_smol(:); % Ensure column vector

        % Check dimensions
        if length(y_train_hat_smol) ~= length(y_train)
            error('The vectors y_train_hat and y_train must be the same length - Smol.');
        end

        % Calculate RMSE, NRMSE, and NMSE
        rmse_train_smol = sqrt(mean((y_train_hat_smol - y_train).^2));
        mse_train_smol = mean((y_train_hat_smol - y_train).^2);
        std_y_train_smol = std(y_train);
        mean_y_train_smol = mean(y_train);
        var_y_train_smol = mean((y_train - mean_y_train_smol).^2);

        nrmse_train_smol = rmse_train_smol / std_y_train_smol;
        nmse_train_smol = mse_train_smol / var_y_train_smol;

        % Display results
        %disp(['NRMSE (Training / Smoldyn): ', num2str(nrmse_train_smol)]);
        %disp(['NMSE (Training / Smoldyn): ', num2str(nmse_train_smol)]);


        %% 12. Testing Phase - SMOLDYN
        % Initialize matrix to store sampled values for testing
        sampled_matrix_test_smol = zeros(Nres, num_test_points);

        sample_indices_to_be_deleted = [];

        % Sampling the number of bound receptors for the testing set
        for i = washout1 + num_train_points + washout2 + 1 : num_tot_points
            t_symbol_start_smol = (i - 1) * T + offset;  % Start time with offset
            start_idx = find(time_index_smol >= t_symbol_start_smol, 1);  % Find corresponding index

            % Generate indices for sampling within the symbol duration
            sample_indices_smol = start_idx - (memorywindowlength-1) * steps_per_symbol_smol + (0:Nres - 1) * (steps_per_symbol_smol / Nres) * memorywindowlength;

            % Ensure indices are within bounds
            sample_indices_smol(sample_indices_smol > length(time_index_smol)) = [];

            if length(sample_indices_smol) >= Nres % NEW - offset control - eliminate the responses that are partially complete due to offset shift
                sampled_matrix_test_smol(:, i - (washout1 + num_train_points + washout2)) = Active_rec_fin(sample_indices_smol(1:Nres));
            else
                %sampled_matrix_test_smol(:, i - (washout1 + num_train_points + washout2)) = [];
                sample_indices_to_be_deleted = [sample_indices_to_be_deleted i - (washout1 + num_train_points + washout2)];
            end
        end

        sampled_matrix_test_smol(:, sample_indices_to_be_deleted) = [];

        % % Plot sample receptor occupation profiles from the testing set
        % figure;
        % subplot(2, 2, 1);
        % plot(sampled_matrix_test_smol(:, 1));
        % title('Sample 1');
        % subplot(2, 2, 2);
        % plot(sampled_matrix_test_smol(:, floor(end * 0.4)));
        % title('Sample 40%');
        % subplot(2, 2, 3);
        % plot(sampled_matrix_test_smol(:, floor(end * 0.8)));
        % title('Sample 80%');
        % subplot(2, 2, 4);
        % plot(sampled_matrix_test_smol(:, end));
        % title('Sample End');

        sampled_matrix_test_smol = [sampled_matrix_test_smol; ones(1, size(sampled_matrix_test_smol, 2))];

        % Transpose for computation
        sampled_matrix_test_smol = sampled_matrix_test_smol';

        % Calculate the estimated output y_test_hat for testing data
        y_test_hat_smol = sampled_matrix_test_smol * W_out_smol;

        %% 13. Compute NRMSE for Testing Data
        % Prepare target output for testing
        %y_segment_test_smol = y_test(1:end - 2);  % Adjust length to match y_test_hat
        %y_test_hat_smol = y_test_hat_smol(1:end - 2);  % Ensure same length

        y_segment_test_smol = y_test;

        % Ensure column vectors
        y_segment_test_smol = y_segment_test_smol(:);
        y_test_hat_smol = y_test_hat_smol(:);

        y_segment_test_smol = y_segment_test_smol(1:length(y_test_hat_smol)); % offset control

        % Check dimensions
        if length(y_test_hat_smol) ~= length(y_segment_test_smol)
            error('The vectors y_test_hat and y_segment_test must be the same length.');
        end

        if movmean_readout_test == 1
            y_test_hat_smol = movmean(y_test_hat_smol,movmean_readout_test_window); % centered average
            % y_test_hat_smol = movmean(y_test_hat_smol,[movmean_readout_test_window 0]); % average over only past values
            % y_test_hat_smol = movmean(y_test_hat_smol,[0 movmean_readout_test_window]); % average over only future values
        end

        % Calculate RMSE, NRMSE, and NMSE
        rmse_test_smol = sqrt(mean((y_test_hat_smol - y_segment_test_smol).^2));
        mse_test_smol = mean((y_test_hat_smol - y_segment_test_smol).^2);
        std_y_test_smol = std(y_segment_test_smol);
        mean_y_test_smol = mean(y_segment_test_smol);
        var_y_test_smol = mean((y_segment_test_smol - mean_y_test_smol).^2);

        nrmse_test_smol = rmse_test_smol / std_y_test_smol;
        nmse_test_smol = mse_test_smol / var_y_test_smol;

        % Display results
        %disp(['NRMSE (Test / Smoldyn): ', num2str(nrmse_test_smol)]);
        %disp(['NMSE (Test / Smoldyn): ', num2str(nmse_test_smol)]);

        %if plot_in_loop
            % Plot the original and estimated target values for testing data
            figure('Position',[400 400 1200 400]);  % [left bottom width height]
            plot(y_segment_test_smol, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);  % Original target values
            hold on;
            plot(y_test_hat_smol, 'r-', 'LineWidth', 2);                        % Estimated target values
            xlabel('Index');
            ylabel('Target and Estimated Values');
            title('Testing Data: Original and Estimated Targets - SMOLDYN');
            legend('Original Target', 'Estimated Target');
            grid on;
            set(gca, 'XTick', []);
        %end

        %% --- Save results ---
        nrmse_train_SMOL_array(i1, i2) = nrmse_train_smol;
        nrmse_test_SMOL_array(i1, i2)  = nrmse_test_smol;

        fprintf('%s=%g, %s=%g -> NRMSE-SMOL(Tr)=%.3g, NRMSE-SMOL(Te)=%.3g\n',...
            param1_name, val1, param2_name, val2, nrmse_train_smol, nrmse_test_smol);

    end
end % END sweeping loops.

%%

%% ========== 2D Heatmaps (Param1 on X, Param2 on Y) ==========
[X, Y] = meshgrid(param1_values, param2_values);  % X: param1, Y: param2

figureOpts = {'LineColor','none','LevelStepMode','auto'};

% 1) NRMSE(NUM)
figure('Name','NRMSE Heatmap for Numerical Model');
contourf(X, Y, nrmse_test_NUM_array', 20, figureOpts{:});
shading interp; colorbar;
colormap(flipud(parula));
if useLogX, set(gca,'XScale','log'); end
if useLogY, set(gca,'YScale','log'); end
xlabel(param1_name);
ylabel(param2_name);
title('NRMSE Heatmap for Numerical Model');

% 2) NRMSE(SMOL)
figure('Name','NRMSE Heatmap for Stochastic Model');
contourf(X, Y, nrmse_test_SMOL_array', 20, figureOpts{:});
shading interp; colorbar;
colormap(flipud(parula));
if useLogX, set(gca,'XScale','log'); end
if useLogY, set(gca,'YScale','log'); end
xlabel(param1_name);
ylabel(param2_name);
title('NRMSE Heatmap for Stochastic Model');

%figure; plot(param2_values, nrmse_test_SMOL_array)


%% Appendix

% (For Mac users) Use the following if MATLAB can't run Smoldyn
% This adds the Smoldyn path to the set of default paths
% setenv('PATH', getenv('PATH')+":/usr/local/bin")
% system('echo $PATH')


