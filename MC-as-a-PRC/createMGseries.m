function createMGseries(MG_length, tau, beta, gamma, n, fileName)
% createMGseries  Generate and save a Mackey-Glass series using RK4 integration.
%
%   createMGseries(MG_length, tau, beta, gamma, n, fileName)
%       MG_length  = Number of discrete points you ultimately want to keep
%                    (sampled at dt=1).
%       tau        = Delay in the Mackey-Glass equation (integer).
%       beta       = Mackey-Glass parameter (e.g., 0.2).
%       gamma      = Mackey-Glass parameter (e.g., 0.1).
%       n          = Nonlinearity parameter (e.g., 10).
%       fileName   = Name of the .mat file to save the final series.
%
% This function:
%   1) Integrates the Mackey–Glass ODE from t=0..tMax with a small step h=0.01 (RK4).
%   2) Samples at integer steps (0,1,2,...).
%   3) Discards an initial transient and ensures that the final discrete series 
%      has at least MG_length points.
%   4) Normalizes the series to [0,1].
%   5) Saves the result in fileName.
%
% The final saved variable 'mackey_glass_series' is a row vector of length >= MG_length.

    %----------------------%
    % 1) Set up parameters
    %----------------------%
    h          = 0.01;  % smaller internal step for RK4
    initVal    = 1.2;   % initial condition x(t<=0) = initVal
    extraTrans = 200;   % extra points to skip as transient (tweak as desired)

    % We'll integrate out to tMax. We want at least MG_length + extraTrans 
    % integer samples after t=0. So let's set:
    tMax = MG_length + extraTrans + tau + 50;  % a bit more margin
    
    % Compute how many RK4 steps:
    Nsteps = round(tMax / h);
    tDense = linspace(0, tMax, Nsteps+1)'; 
    xDense = zeros(Nsteps+1, 1);
    xDense(1) = initVal;

    %---------------------------%
    % 2) RK4 with Delayed Term
    %---------------------------%
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
    % 3) Sample at integer t = 0,1,2,...
    %------------------------------%
    tSample = 0 : 1 : floor(tMax);  % integer times
    xSample = interp1(tDense, xDense, tSample, 'pchip')';  % row vector

    % Discard an initial transient portion (including any up to tau, etc.)
    % We'll remove first 'extraTrans' points from the beginning:
    if extraTrans >= length(xSample)
        error('Not enough points after discarding transient. Increase tMax or reduce extraTrans.');
    end
    xSampleFinal = xSample(extraTrans+1 : end);

    % Ensure we have at least MG_length points:
    if length(xSampleFinal) < MG_length
        error('Need more points. Increase tMax or decrease extraTrans.');
    end

    %------------------------------%
    % 4) Normalize to [0,1]
    %------------------------------%
    mgMin = min(xSampleFinal);
    mgMax = max(xSampleFinal);
    mackey_glass_series = (xSampleFinal - mgMin) / (mgMax - mgMin);

    %------------------------------%
    % 5) Save the result
    %------------------------------%
    save(fileName, 'mackey_glass_series');
    fprintf('Created MG series with length=%d, saved to "%s".\n', length(mackey_glass_series), fileName);
end

%--------------------------------------------------------------------------
% Local function: mg_rhs
%   The Mackey-Glass derivative with delay
%--------------------------------------------------------------------------
function dx = mg_rhs(t, x, tDense, xDense, tau, beta, gamma, n, initVal)
    tLag = t - tau;
    if tLag < 0
        % If we are asking for x(t - tau) at negative time, assume constant = initVal
        xLag = initVal;
    else
        xLag = interp1(tDense, xDense, tLag, 'pchip');
    end
    dx = beta * xLag / (1 + xLag^n) - gamma * x;
end
