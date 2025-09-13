function createMGcubedseries(MG_length, predict_length, tau, beta, gamma, n, fileName)
% eg use in command line: createMGcubedseries(5000, 10, 17, 0.2, 0.1, 10, 'MGCubed_series_k10.mat');
% createMGcubedseries  Generate and save data for the Mackey-Glass CUBED task.
%
%   createMGcubedseries(MG_length, predict_length, tau, beta, gamma, n, fileName)
%       MG_length      = Number of discrete points for the INPUT series.
%       predict_length = Future steps 'k' to predict (e.g., 10).
%       tau, beta, etc.= Mackey-Glass parameters.
%       fileName       = Name of the .mat file to save the final series.
%
% This function:
%   1) Integrates the Mackey–Glass ODE to get a long, high-resolution series.
%   2) Samples at integer steps and discards an initial transient.
%   3) Normalizes the entire series to [0,1].
%   4) Creates the 'input_series' (length = MG_length).
%   5) Creates the 'target_series' by taking the future (t+k) values of the
%      original series and cubing them.
%   6) Saves both 'input_series' and 'target_series' to the specified file.

    %----------------------%
    % 1) Set up parameters
    %----------------------%
    h          = 0.01;  % smaller internal step for RK4
    initVal    = 1.2;   % initial condition x(t<=0) = initVal
    extraTrans = 200;   % extra points to skip as transient

    % We need enough points for the input series PLUS the prediction length.
    tMax = MG_length + predict_length + extraTrans + tau + 50;
    
    % Compute how many RK4 steps:
    Nsteps = round(tMax / h);
    tDense = linspace(0, tMax, Nsteps+1)'; 
    xDense = zeros(Nsteps+1, 1);
    xDense(1) = initVal;

    %---------------------------%
    % 2) RK4 with Delayed Term
    %---------------------------%
    fprintf('Integrating Mackey-Glass ODE...\n');
    for i = 1 : Nsteps
        tNow = tDense(i);
        xNow = xDense(i);

        k1 = mg_rhs(tNow,         xNow,          tDense, xDense, tau, beta, gamma, n, initVal);
        k2 = mg_rhs(tNow+0.5*h,   xNow+0.5*h*k1, tDense, xDense, tau, beta, gamma, n, initVal);
        k3 = mg_rhs(tNow+0.5*h,   xNow+0.5*h*k2, tDense, xDense, tau, beta, gamma, n, initVal);
        k4 = mg_rhs(tNow+    h,   xNow+    h*k3, tDense, xDense, tau, beta, gamma, n, initVal);

        xDense(i+1) = xNow + (h/6)*(k1 + 2*k2 + 2*k3 + k4);
    end

    %------------------------------%
    % 3) Sample, Discard Transient, Normalize
    %------------------------------%
    tSample = 0 : 1 : floor(tMax);
    xSample = interp1(tDense, xDense, tSample, 'pchip')';

    if extraTrans >= length(xSample)
        error('Not enough points after discarding transient. Increase tMax.');
    end
    xSampleFinal = xSample(extraTrans+1 : end);

    if length(xSampleFinal) < (MG_length + predict_length)
        error('Need more points. Increase tMax or decrease extraTrans.');
    end
    
    % Normalize the full available series before splitting
    mgMin = min(xSampleFinal);
    mgMax = max(xSampleFinal);
    full_normalized_series = (xSampleFinal - mgMin) / (mgMax - mgMin);

    %------------------------------%
    % 4) Create Input and Cubed Target
    %------------------------------%
    fprintf('Creating input and cubed target series...\n');
    
    % Input series is the first part of the full series
    input_series = full_normalized_series(1 : MG_length);
    
    % Target series is the future, cubed version
    target_series_future = full_normalized_series( (1+predict_length) : (MG_length + predict_length) );
    target_series = target_series_future.^3;

    %------------------------------%
    % 5) Save the result
    %------------------------------%
    save(fileName, 'input_series', 'target_series');
    fprintf('Created MG cubed task data with k=%d.\n', predict_length);
    fprintf('Input length = %d, Target length = %d.\n', length(input_series), length(target_series));
    fprintf('Saved to "%s".\n', fileName);
end

%--------------------------------------------------------------------------
% Local function: mg_rhs (No changes needed here)
%--------------------------------------------------------------------------
function dx = mg_rhs(t, x, tDense, xDense, tau, beta, gamma, n, initVal)
    tLag = t - tau;
    if tLag < 0
        xLag = initVal;
    else
        xLag = interp1(tDense, xDense, tLag, 'pchip');
    end
    dx = beta * xLag / (1 + xLag^n) - gamma * x;
end