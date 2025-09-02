%% createSINEseries.m
% This script generates and saves the input/target data for the
% sine-to-square wave nonlinear transformation task.
% VERSION 2: Does not require the Signal Processing Toolbox.

function createSINEseries(nSamples, num_periods, fileName)
    % Set default arguments if none are provided
    if nargin < 1, nSamples = 5000; end
    if nargin < 2, num_periods = 25; end
    if nargin < 3, fileName = 'SINEseries_len5000_p25.mat'; end

    fprintf('Generating sine-to-square transformation task data...\n');

    % --- 1. Set up parameters ---
    T_symbol = 1.0; % Symbol duration in seconds, consistent with your config
    t = (0:nSamples-1)' * T_symbol; % Time vector (as a column vector)

    % Calculate frequency to achieve the desired number of periods over the total duration
    total_duration = nSamples * T_symbol;
    freq = num_periods / total_duration;

    % --- 2. Generate the signals ---
    % Generate a sine wave for input
    sine_wave = sin(2 * pi * freq * t);
    input_sine_series = sine_wave;

    % Generate a square wave for the target
    % MODIFIED LINE: Replaced square() with sign() to avoid toolbox dependency.
    % The sign of a sine wave is a square wave.
    target_square_series = sign(sine_wave);

    % --- 3. Normalize to [0,1] ---
    % This is crucial, as your channel simulation maps the input signal
    % to a number of molecules (e.g., between N_min and N_max).
    input_sine_series = (input_sine_series + 1) / 2;
    target_square_series = (target_square_series + 1) / 2;
    
    % Address potential NaN for the first element if target_square_series(1) was 0
    if isnan(target_square_series(1))
        target_square_series(1) = 0.5; % Or whatever neutral value you prefer
    end

    % --- 4. Save the result ---
    save(fileName, 'input_sine_series', 'target_square_series', 't', 'nSamples', 'T_symbol');
    fprintf('✅ Created sine/square series with length=%d, saved to "%s".\n', nSamples, fileName);
    
    % --- 5. Verification Plot ---
    figure;
    plot(t, input_sine_series, 'b-', 'LineWidth', 1.5);
    hold on;
    plot(t, target_square_series, 'r--', 'LineWidth', 2);
    hold off;
    title('Generated Transformation Task Data');
    xlabel('Time (s)');
    ylabel('Normalized Amplitude');
    legend('Input (Sine)', 'Target (Square)');
    grid on;
    xlim([0, (3/freq)]); % Show the first 3 periods for clarity
end