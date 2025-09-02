%% createSAWseries.m
% This script generates and saves the input/target data for the
% sine-to-sawtooth wave nonlinear transformation task.

function createSAWseries(nSamples, num_periods, fileName)
    % Set default arguments if none are provided
    if nargin < 1, nSamples = 5000; end
    if nargin < 2, num_periods = 25; end
    if nargin < 3, fileName = 'SAWseries_len5000_p25.mat'; end

    fprintf('Generating sine-to-sawtooth transformation task data...\n');

    % --- 1. Set up parameters ---
    T_symbol = 1.0; % Symbol duration in seconds
    t = (0:nSamples-1)' * T_symbol; % Time vector

    total_duration = nSamples * T_symbol;
    freq = num_periods / total_duration;

    % --- 2. Generate the signals ---
    % Generate a sine wave for input
    input_sine_series = sin(2 * pi * freq * t);

    % Generate a sawtooth wave for the target using MATLAB's sawtooth() function.
    % The second argument '1' creates a full rising ramp, which is a standard sawtooth.
    target_sawtooth_series = sawtooth(2 * pi * freq * t, 1);

    % --- 3. Normalize to [0,1] ---
    input_sine_series = (input_sine_series + 1) / 2;
    target_sawtooth_series = (target_sawtooth_series + 1) / 2;
    
    % --- 4. Save the result ---
    save(fileName, 'input_sine_series', 'target_sawtooth_series', 't', 'nSamples', 'T_symbol');
    fprintf('✅ Created sine/sawtooth series with length=%d, saved to "%s".\n', nSamples, fileName);
    
    % --- 5. Verification Plot ---
    figure;
    plot(t, input_sine_series, 'b-', 'LineWidth', 1.5);
    hold on;
    plot(t, target_sawtooth_series, 'r--', 'LineWidth', 2);
    hold off;
    title('Generated Transformation Task Data');
    xlabel('Time (s)');
    ylabel('Normalized Amplitude');
    legend('Input (Sine)', 'Target (Sawtooth)');
    grid on;
    xlim([0, (3/freq)]); % Show the first 3 periods for clarity
end