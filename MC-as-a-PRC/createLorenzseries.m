% =========================================================================
% createLorenzseries.m
%
% DESCRIPTION:
%   Generates and saves a Lorenz '63 chaotic time series. This script is
%   runnable by pressing 'Play' in the MATLAB editor.
% =========================================================================

%% 1. Configuration
clear;
close all;

series_length = 8000; % Total number of points for the final series
fileName = 'lorenz_series_len8000.mat'; % Output file name

% Standard chaotic parameters for the Lorenz system
sigma = 10;
rho   = 28;
beta  = 8/3;

% Simulation parameters
dt = 0.01;            % Time step for the final sampled series
transient_time = 100; % Time to run to let the attractor stabilize
initial_state = [0.1, 0.1, 0.1]; % Initial conditions [x, y, z]

%% 2. Integration and Sampling
total_time = series_length * dt + transient_time;
tspan = [0, total_time];

fprintf('Integrating Lorenz ''63 system...\n');

% Define the Lorenz ODE system for the solver
lorenz_ode = @(t, state) [sigma * (state(2) - state(1)); ...
                         state(1) * (rho - state(3)) - state(2); ...
                         state(1) * state(2) - beta * state(3)];

% Use MATLAB's robust ode45 solver for efficiency and accuracy
options = odeset('RelTol', 1e-6, 'AbsTol', 1e-9);
[t_dense, state_dense] = ode45(lorenz_ode, tspan, initial_state, options);

% Interpolate the dense solution onto our desired discrete time steps
t_sample = (0:dt:total_time)';
state_sample = interp1(t_dense, state_dense, t_sample);

%% 3. Post-Processing
% Discard the initial transient part of the series
transient_steps = round(transient_time / dt);
final_series = state_sample(transient_steps + 1 : end, :);

% Trim to ensure exactly the desired length
if size(final_series, 1) > series_length
    final_series = final_series(1:series_length, :);
end

% Normalize each dimension (x, y, z) independently to the range [0, 1]
lorenz_series = zeros(size(final_series));
for i = 1:3
    dim_min = min(final_series(:, i));
    dim_max = max(final_series(:, i));
    lorenz_series(:, i) = (final_series(:, i) - dim_min) / (dim_max - dim_min);
end

%% 4. Save and Visualize
save(fileName, 'lorenz_series', 'sigma', 'rho', 'beta', 'dt');
fprintf('✅ Created Lorenz series with length=%d, saved to "%s".\n', size(lorenz_series, 1), fileName);

% Plot the final, normalized attractor as a verification step
figure('Name', 'Generated Lorenz Attractor');
plot3(lorenz_series(:,1), lorenz_series(:,2), lorenz_series(:,3));
title('Normalized Lorenz Attractor');
xlabel('x'); ylabel('y'); zlabel('z');
grid on;
axis tight;
view(35, 25); % Set a nice 3D viewing angle