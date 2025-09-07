%% createWHseries.m
% This script generates and saves the input/target data for the
% Wiener-Hammerstein nonlinear transformation task.
%
% A Wiener-Hammerstein system is a block model consisting of a
% linear dynamic block, followed by a static nonlinearity, followed by
% another linear dynamic block (LTI -> NL -> LTI).

function createWHseries(nSamples, fileName)
    % Set default arguments if none are provided
    if nargin < 1, nSamples = 5000; end
    if nargin < 2, fileName = 'WHseries_len5000.mat'; end

    fprintf('Generating Wiener-Hammerstein transformation task data...\n');

    % --- 1. Set up parameters ---
    T_symbol = 1.0; % Symbol duration in seconds
    t = (0:nSamples-1)' * T_symbol; % Time vector

    % --- 2. Generate the signals ---
    
    % Create a random input signal. Using a band-limited pseudorandom 
    % binary signal is a standard approach for system identification.
    input_signal = idinput(nSamples, 'prbs');

    % Define the Wiener-Hammerstein model structure using the System Identification Toolbox
    % LTI Block 1 (Input): A second-order low-pass filter
    B1 = [0.1, 0.1];
    A1 = [1, -0.8, 0.2];
    lti_input = idtf(B1, A1, T_symbol);

    % Static Nonlinearity: A saturation function
    nonlinearity = idSaturation('LinearInterval', [-0.8, 0.8]);

    % LTI Block 2 (Output): A first-order high-pass filter
    B2 = [1, -1];
    A2 = [1, -0.9];
    lti_output = idtf(B2, A2, T_symbol);

    % --- FIX: Simulate the model stages sequentially ---
    % Instead of creating an idnlhw object, we will pass the signal through each component.
    % This correctly models the LTI -> NL -> LTI structure and avoids the syntax error.
    
    % Stage 1: Pass input through the first LTI filter
    signal_after_lti1 = lsim(lti_input, input_signal, t);
    
    % Stage 2: Apply the static nonlinearity
    signal_after_nl = evaluate(nonlinearity, signal_after_lti1);
    
    % Stage 3: Pass the result through the second LTI filter
    target_signal_data = lsim(lti_output, signal_after_nl, t);

    % --- 3. Normalize to [0,1] ---
    % This is crucial for the reservoir computer, which maps input values
    % to a number of molecules.
    input_WH_series = (input_signal - min(input_signal)) / (max(input_signal) - min(input_signal));
    target_WH_series = (target_signal_data - min(target_signal_data)) / (max(target_signal_data) - min(target_signal_data));
    
    % --- 4. Save the result ---
    save(fileName, 'input_WH_series', 'target_WH_series', 't', 'nSamples', 'T_symbol');
    fprintf('✅ Created Wiener-Hammerstein series with length=%d, saved to "%s".\n', nSamples, fileName);
    
    % --- 5. Verification Plot ---
    figure('Name', 'Wiener-Hammerstein Generated Data');
    plot(t, input_WH_series, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Input (Random)');
    hold on;
    plot(t, target_WH_series, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Target (WH Output)');
    title('Wiener-Hammerstein Input and Target Time Series');
    xlabel('Time Step');
    ylabel('Normalized Amplitude');
    legend show;
    grid on;
    xlim([0, 200]); % Show only the first 200 steps for clarity
end