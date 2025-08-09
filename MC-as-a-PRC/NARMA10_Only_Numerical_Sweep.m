%% 0a. Initialize the MATLAB environment
close all;   % Close all open figure windows
clear;       % Clear all variables from the workspace

%% 0b. Analysis Setting Control
plot_in_loop = false;   % set to true to see intermediate loop figures
memorywindowlength = 1; % = 1 by default

rng(0); % IMPORTANT TO FIX THE RANDOM SEQUENCE (both for numerical and Smoldyn)

%% 1. Parameter Setup

%---------------------------------------------------------
%         Primary & Secondary Swept Parameters
%---------------------------------------------------------
param1_name   = 'k_on';      % e.g. 'k_on', 'N', 'T', 'k_off', ...
param1_values = [5e-20, 1e-19,2e-19, 5e-19, 1e-18, 2e-18, 5e-18, 1e-17, 2e-17]; % optimal range for the paper (spans 0.1-0.9 occupation ratio).


param2_name   = 'k_off';
param2_values = [0.1, 0.2, 0.5, 1, 2, 5, 10]; % optimal range for the paper (spans 0.1-0.9 occupation ratio).

% param2_name   = 'T';
% param2_values = [0.5, 0.8, 1, 1.5, 2];

% param2_name   = 'distance';
% param2_values = [1e-6, 2e-6, 5e-6, 10e-6, 20e-6, 50e-6];

% param2_name   = 'memorywindowlength';
% param2_values = [1, 2, 3, 4, 5];

% param2_name   = 'N_max';
% param2_values = [200, 500, 1000, 3000, 10000, 20000];

% param2_name   = 'D';
% param2_values = [0.5e-11, 1e-11, 2e-11, 5e-11, 1e-10, 2e-10];

%---------------------------------------------------------
%  Fixed Receptor & Binding Parameters (not swept)
%---------------------------------------------------------
N     = 500;      % total number of receptors
k_on  = 1e-18;     % binding rate constant (1/(M*s))
k_off = 1;         % unbinding rate constant (1/s)
KD    = k_off / k_on;

% Input normalization
N_min = 100;
N_max = 3000;

% Regularization & reservoir
lambda = 1e-10;
Nres_init = 100;               % Number of reservoir nodes
Nres = Nres_init * memorywindowlength;

% Communication channel parameters
distance = 10e-6;  % 10 microns
D        = 1e-11;

% Time parameters
T  = 1;   % symbol duration (seconds)

% Data lengths
washout1         = 300;
num_train_points = 2000;
wheretostarttest = 2500;
washout2         = wheretostarttest - (washout1 + num_train_points);
num_test_points  = 2000;

num_tot_points = washout1 + washout2 + num_train_points + num_test_points;

% Step size control
dt = 0.001;  % numerical solution time step
memlengthsweep = 100;
Tpeak = distance^2/(6*D);
memory_length = round((memlengthsweep * Tpeak)/T)*T; % memory of MC channel
offset = 0;

%% 2. Generate the NARMA10 Time Series (Only ONCE)

% We need (num_tot_points + 1) final data points, plus 10 warmup.
narma_len = num_tot_points + 10 + 1;   % +10 warmup, +1 so we have pairs for next-step

% Random input in [0, 0.5]
u = 0.5 * rand(narma_len, 1);

% Pre-allocate the NARMA output
q = zeros(narma_len, 1);

% Compute q(n) for NARMA10 system
for n = 10:num_tot_points-1
    q(n+1) = 0.3 * q(n) + 0.05 * q(n) * sum(q(n-9:n)) + 1.5 * u(n-9) * u(n) + 0.1;
end




%% 4. Prepare Arrays for Storing Results
nP1 = length(param1_values);
nP2 = length(param2_values);

nrmse_train_array                          = zeros(nP1, nP2);
nrmse_test_array                           = zeros(nP1, nP2);
mean_rec_occ_ratio_array                  = zeros(nP1, nP2);
tscale_char_array                         = zeros(nP1, nP2);
tscale_char_from_data_array               = zeros(nP1, nP2);
tscale_char_concentration_from_data_array = zeros(nP1, nP2);
tscale_char_from_rcsamples_array          = zeros(nP1, nP2);

%% Automatic log-scale checks
minP1   = min(param1_values);
maxP1   = max(param1_values);
ratioP1 = maxP1 / minP1;
useLogX = (minP1>0) && (ratioP1>50);

minP2   = min(param2_values);
maxP2   = max(param2_values);
ratioP2 = maxP2 / minP2;
useLogY = (minP2>0) && (ratioP2>50);



%% 5. Outer/Inner Loops over param2, param1
for i2 = 1 : nP2
    val2 = param2_values(i2);

    %--- Only set parameters that do NOT affect the data length. ---
    switch param2_name
        case 'N'
            N = val2;
        case 'k_on'
            k_on = val2;  KD = k_off / k_on;
        case 'k_off'
            k_off = val2; KD = k_off / k_on;
        case 'T'
            T = val2;
            memory_length = round((memlengthsweep * Tpeak)/T)*T;
        case 'distance'
            distance = val2;
            Tpeak = distance^2/(6*D);
            memory_length = round((memlengthsweep * Tpeak)/T)*T;
        case 'D'
            D = val2;
            Tpeak = distance^2/(6*D);
            memory_length = round((memlengthsweep * Tpeak)/T)*T;
        case 'N_min'
            N_min = val2;
        case 'N_max'
            N_max = val2;
        case 'lambda'
            lambda = val2;
        case 'Nres_init'
            Nres_init = val2;
            Nres = Nres_init * memorywindowlength;
        case 'memorywindowlength'
            memorywindowlength = val2;
            Nres = Nres_init * memorywindowlength;
        case 'memlengthsweep'
            memlengthsweep = val2;
            memory_length = round((memlengthsweep * Tpeak)/T)*T;
        otherwise
            error('Parameter2 name (%s) not recognized or not allowed to be swept.', param2_name);
    end


    for i1 = 1 : nP1
        val1 = param1_values(i1);

        tic
        %--- Only set parameters that do NOT affect the data length. ---
        switch param1_name
            case 'N'
                N = val1;
            case 'k_on'
                k_on = val1;  KD = k_off / k_on;
            case 'k_off'
                k_off = val1; KD = k_off / k_on;
            case 'T'
                T = val1;
                memory_length = round((memlengthsweep * Tpeak)/T)*T;
            case 'distance'
                distance = val1;
                Tpeak = distance^2/(6*D);
                memory_length = round((memlengthsweep * Tpeak)/T)*T;
            case 'D'
                D = val1;
                Tpeak = distance^2/(6*D);
                memory_length = round((memlengthsweep * Tpeak)/T)*T;
            case 'N_min'
                N_min = val1;
            case 'N_max'
                N_max = val1;
            case 'lambda'
                lambda = val1;
            case 'Nres_init'
                Nres_init = val1;
                Nres = Nres_init * memorywindowlength;
            case 'memorywindowlength'
                memorywindowlength = val1;
                Nres = Nres_init * memorywindowlength;
            case 'memlengthsweep'
                memlengthsweep = val1;
                memory_length = round((memlengthsweep * Tpeak)/T)*T;
            otherwise
                error('Parameter1 name (%s) not recognized or not allowed to be swept.', param1_name);
        end


        % Normalize input u(n) to obtain N_i values
        N_i = N_min + (u - min(u)) / (max(u) - min(u)) * (N_max - N_min);

        %% 7. ODE Integration for Binding

        t_total = length(N_i)*T;
        t_values = 0 : dt : t_total;
        steps_per_symbol = T/dt;

        c_values = zeros(size(t_values));
        n_values = zeros(size(t_values));
        n_values(1) = 0;

        %--- Concentration Calculation ---
        for iSym = 1 : length(N_i)
            t_symbol_start = (iSym - 1)*T;
            t_memory_end   = t_symbol_start + memory_length;
            idxRange = find(t_values > t_symbol_start & t_values <= t_memory_end);
            if isempty(idxRange), continue; end

            for j = idxRange
                tlocal = t_values(j) - t_symbol_start;
                if tlocal > 0
                    c_values(j) = c_values(j) + ...
                        (N_i(iSym) / ((4*pi*D*tlocal)^(3/2))) * ...
                        exp(-distance^2/(4*D*tlocal));
                end
            end
        end

        %--- Receptor Binding ODE (Euler) ---
        for idxT = 2 : length(t_values)
            c_t = c_values(idxT - 1);
            n_t = n_values(idxT - 1);
            dn_dt = k_on*(N - n_t)*c_t - k_off*n_t;
            n_values(idxT) = n_t + dn_dt*dt;
        end
        n_values = n_values / N;

        % Compute means and timescales
        halfIdx = round(0.5 * length(n_values));
        mean_occ_ratio = mean(n_values(halfIdx:end));
        mean_c_val     = mean(c_values(halfIdx:end));
        timescale_char = 1/(k_on*mean_c_val + k_off);

        % Autocorrelation: n_values
        n_ss = n_values(halfIdx:end) - mean(n_values(halfIdx:end));
        [acf_n, lags_n] = xcorr(n_ss, 100000, 'coeff');
        idxPos = (lags_n >= 0);
        acf_n  = acf_n(idxPos);
        lags_n = lags_n(idxPos);
        idx_e  = find(acf_n < exp(-1), 1, 'first');
        if isempty(idx_e), idx_e = length(lags_n); end
        tau_corr_n = lags_n(idx_e)*dt;

        % Autocorrelation: c_values
        c_ss = c_values(halfIdx:end) - mean(c_values(halfIdx:end));
        [acf_c, lags_c] = xcorr(c_ss, 100000, 'coeff');
        idxPos2 = (lags_c >= 0);
        acf_c   = acf_c(idxPos2);
        lags_c  = lags_c(idxPos2);
        idx_e2  = find(acf_c < exp(-1), 1, 'first');
        if isempty(idx_e2), idx_e2 = length(lags_c); end
        tau_corr_c = lags_c(idx_e2)*dt;

        if plot_in_loop
            figure;
            plot(t_values, n_values, 'LineWidth',2); grid on;
            xlabel('Time (s)'); ylabel('n(t)'); title('Receptor Occupation Fraction');
            figure;
            plot(t_values, c_values, 'LineWidth',2); grid on;
            xlabel('Time (s)'); ylabel('c(t)'); title('Ligand Concentration');
        end

        %% 8. Training Phase
        sampled_matrix = zeros(Nres, num_train_points);
        sample_indices_to_be_deleted = [];

        for iSym = (washout1+1) : (washout1 + num_train_points)
            t_symbol_start = (iSym - 1)*T + offset;
            start_idx = find(t_values >= t_symbol_start, 1);
            if isempty(start_idx)
                sample_indices_to_be_deleted(end+1) = (iSym - washout1);
                continue;
            end
            sample_indices = start_idx - (memorywindowlength-1) * steps_per_symbol + (0:(Nres-1))*(steps_per_symbol/Nres)*memorywindowlength;
            sample_indices(sample_indices>length(t_values)) = [];

            colIdx = iSym - washout1;
            if length(sample_indices) >= Nres
                sampled_matrix(:, colIdx) = n_values(sample_indices(1:Nres));
            else
                sample_indices_to_be_deleted(end+1) = colIdx;
            end
        end

        sampled_matrix(:, sample_indices_to_be_deleted) = [];
        if isempty(sampled_matrix)
            warning('No valid training samples for %s=%g, %s=%g. Skipping.', ...
                param1_name, val1, param2_name, val2);
            continue;
        end

        % Prepare target output for training (shifted q(n))
        q_train = q(washout1+2:washout1+num_train_points+1)';  % Column vector

        nCols = size(sampled_matrix, 2);
        sampled_matrix = [sampled_matrix; ones(1, nCols)]; % add bias row
        if nCols > length(q_train)
            warning('Not enough y_train data for %s=%g, %s=%g. Skipping.', ...
                param1_name, val1, param2_name, val2);
            continue;
        end
        q_train = q_train(1:nCols);
        q_train = q_train(:);

        % (#trainingSamples x #features)
        sampled_matrix = sampled_matrix';
        W_out = pinv(sampled_matrix'*sampled_matrix + lambda*eye(size(sampled_matrix,2))) ...
            * (sampled_matrix'*q_train);

        q_train_hat = sampled_matrix * W_out;

        rmse_train = sqrt(mean((q_train_hat(1:end-2) - q_train(1:end-2)).^2));

        nrmse_train = rmse_train / std(q_train(1:end-2));

        %% 9. Testing Phase
        sampled_matrix_test = zeros(Nres, num_test_points);
        sample_indices_to_be_deleted = [];


        for iSym = (washout1 + num_train_points + washout2 + 1) : num_tot_points
            t_symbol_start = (iSym - 1)*T + offset;
            start_idx = find(t_values >= t_symbol_start, 1);
            if isempty(start_idx)
                colTest = iSym - (washout1 + num_train_points + washout2);
                sample_indices_to_be_deleted(end+1) = colTest;
                continue;
            end
            sample_indices = start_idx - (memorywindowlength-1) * steps_per_symbol + (0:(Nres-1))*(steps_per_symbol/Nres)*memorywindowlength;
            sample_indices(sample_indices>length(t_values)) = [];

            colTest = iSym - (washout1 + num_train_points + washout2);
            if length(sample_indices) >= Nres
                sampled_matrix_test(:, colTest) = n_values(sample_indices(1:Nres));
            else
                sample_indices_to_be_deleted(end+1) = colTest;
            end
        end

        sampled_matrix_test(:, sample_indices_to_be_deleted) = [];
        if isempty(sampled_matrix_test)
            warning('No valid test samples for %s=%g, %s=%g. Skipping.', ...
                param1_name, val1, param2_name, val2);
            continue;
        end

        nColsTest = size(sampled_matrix_test, 2);
        sampled_matrix_test = [sampled_matrix_test; ones(1, nColsTest)];

        % Prepare target output for testing (shifted q(n))
        q_test = q(washout1 + num_train_points + washout2 +2:num_tot_points+1)';  % Column vector

        if nColsTest > length(q_test)
            warning('Not enough y_test data for %s=%g, %s=%g. Skipping.', ...
                param1_name, val1, param2_name, val2);
            continue;
        end

        q_test = q_test(1:nColsTest);
        q_test = q_test(:);

        sampled_matrix_test = sampled_matrix_test';
        q_test_hat = sampled_matrix_test * W_out;

        rmse_test = sqrt(mean((q_test_hat(1:end-2) - q_test(1:end-2)).^2));
        nrmse_test = rmse_test / std(q_test(1:end-2));

        %% 10. Timescale from ALL Reservoir Samples
        sampled_matrix_all = [];
        for iSymAll = (washout1+1) : num_tot_points
            start_idx_all = find(t_values >= (iSymAll-1)*T, 1);
            if isempty(start_idx_all), continue; end
            sample_indices_all = start_idx_all - (memorywindowlength-1) * steps_per_symbol + (0:(Nres-1))*(steps_per_symbol/Nres)*memorywindowlength;
            sample_indices_all(sample_indices_all>length(t_values)) = [];
            if length(sample_indices_all) < Nres
                continue;
            end
            newCol = n_values(sample_indices_all(1:Nres));
            sampled_matrix_all = [sampled_matrix_all, newCol]; %#ok<AGROW>
        end

        % Flatten row-by-row => single time series
        dtsample = (T / Nres) * memorywindowlength;
        res_sequence = reshape(sampled_matrix_all, 1, []);
        res_sequence_centered = res_sequence - mean(res_sequence);

        maxLag_rc = 5e4;
        [acf_rc, lags_rc] = xcorr(res_sequence_centered, maxLag_rc, 'coeff');
        idxPosRC = (lags_rc >= 0);
        acf_rc   = acf_rc(idxPosRC);
        lags_rc  = lags_rc(idxPosRC);
        idx_eRC  = find(acf_rc < exp(-1), 1, 'first');
        if isempty(idx_eRC), idx_eRC = length(lags_rc); end
        tau_corr_rc_samples = lags_rc(idx_eRC)*dtsample;

        %% 11. Save results
        nrmse_train_array(i1, i2) = nrmse_train;
        nrmse_test_array(i1, i2)  = nrmse_test;
        mean_rec_occ_ratio_array(i1, i2) = mean_occ_ratio;
        tscale_char_array(i1, i2) = timescale_char;
        tscale_char_from_data_array(i1, i2) = tau_corr_n;
        tscale_char_concentration_from_data_array(i1, i2) = tau_corr_c;
        tscale_char_from_rcsamples_array(i1, i2) = tau_corr_rc_samples;

        fprintf('%s=%g, %s=%g -> NRMSE(Tr)=%.3g, NRMSE(Te)=%.3g, Occ=%.3g, tauChar=%.3g, RCtau=%.3g\n',...
            param1_name, val1, param2_name, val2, ...
            nrmse_train, nrmse_test, mean_occ_ratio, timescale_char, tau_corr_rc_samples);


        % Plot the original q_test(n) and the estimated q_hat_test
        figure;
        plot(washout1 + num_train_points + washout2 +2 : num_tot_points+1 - 2, q_test(1:end-2), 'b-o', 'LineWidth', 2, 'MarkerSize', 6);  % Original q(n)
        hold on;
        plot(washout1 + num_train_points + washout2 +2 : num_tot_points+1 - 2, q_test_hat(1:end-2), 'r-', 'LineWidth', 2);  % Estimated q_hat_test
        xlabel('Index');
        ylabel('q\_test and estimated q\_test');
        title('Target q\_test and Estimated q\_test Values');
        legend('Original q\_test', 'Estimated q\_test');
        grid on;
        set(gca, 'XTick', []);

        toc
    end
end

%% =========== NARMA10 Autocorrelation (Example Plot) ===========
% Optional demonstration on the entire final narma_series
maxLagNarma = 200;
narma_centered = q - mean(q);
[acfNarma, lagsNarma] = xcorr(narma_centered, maxLagNarma, 'coeff');
acfNarma  = acfNarma(lagsNarma>=0);
lagsNarma = lagsNarma(lagsNarma>=0);

idx1eNarma = find(acfNarma < exp(-1), 1, 'first');
if isempty(idx1eNarma), idx1eNarma = maxLagNarma; end
tauNarma_1e = lagsNarma(idx1eNarma);
tauNarma_1e_time = tauNarma_1e * T;

figure('Name','NARMA10 Autocorrelation','Position',[300 100 600 400]);
plot(lagsNarma, acfNarma, 'LineWidth',2); hold on; grid on;
plot(tauNarma_1e, acfNarma(idx1eNarma), 'ro','MarkerSize',8,'LineWidth',2);
xlabel('Lag (samples)'); ylabel('Autocorrelation');
title('NARMA10 Autocorrelation');
legend('ACF','1/e crossing','Location','Best');
fprintf('NARMA10 correlation time ~ %d samples ~ %g seconds.\n',...
    tauNarma_1e, tauNarma_1e_time);

%% ========== 2D Heatmaps (Param1 on X, Param2 on Y) ==========
[X, Y] = meshgrid(param1_values, param2_values);
figureOpts = {'LineColor','none','LevelStepMode','auto'};

% 1) NRMSE(Train)
figure('Name','NRMSE(Train) Heatmap');
contourf(X, Y, nrmse_train_array', 20, figureOpts{:});
shading interp; colorbar;
colormap(flipud(parula));
if useLogX, set(gca,'XScale','log'); end
if useLogY, set(gca,'YScale','log'); end
xlabel(param1_name);
ylabel(param2_name);
title('NRMSE(Train) Heatmap');

% 2) NRMSE(Test)
figure('Name','NRMSE(Test) Heatmap');
contourf(X, Y, nrmse_test_array', 20, figureOpts{:});
shading interp; colorbar;
colormap(flipud(parula));
if useLogX, set(gca,'XScale','log'); end
if useLogY, set(gca,'YScale','log'); end
xlabel(param1_name);
ylabel(param2_name);
title('NRMSE(Test) Heatmap');

% 3) Mean Receptor Occupation
figure('Name','Mean Receptor Occupation Heatmap');
contourf(X, Y, mean_rec_occ_ratio_array', 20, figureOpts{:});
shading interp; colorbar;
if useLogX, set(gca,'XScale','log'); end
if useLogY, set(gca,'YScale','log'); end
xlabel(param1_name);
ylabel(param2_name);
title('Mean Receptor Occupation Ratio Heatmap');

% 4) Timescale Characteristic (Theory) => log-scale in Z
Zvals = tscale_char_array';
Zvals_log = log10(Zvals);

figure('Name','Timescale Characteristic (Theory) Heatmap');
contourf(X, Y, Zvals_log, 20, 'LineColor','none');
shading interp;
cb = colorbar;
colormap(jet);
caxis([log10(min(Zvals(:)))  log10(max(Zvals(:)))]);
tickVals = floor(log10(min(Zvals(:)))) : 1 : ceil(log10(max(Zvals(:))));
cb.Ticks = tickVals;
cb.TickLabels = arrayfun(@(x) sprintf('10^{%d}', x), tickVals, 'UniformOutput', false);

if useLogX, set(gca,'XScale','log'); end
if useLogY, set(gca,'YScale','log'); end
xlabel(param1_name);
ylabel(param2_name);
title('Timescale Characteristic (theory) Heatmap');

% 5) Timescale from n-values
figure('Name','Timescale from n-values Heatmap');
contourf(X, Y, tscale_char_from_data_array', 20, figureOpts{:});
shading interp; colorbar;
if useLogX, set(gca,'XScale','log'); end
if useLogY, set(gca,'YScale','log'); end
xlabel(param1_name);
ylabel(param2_name);
title('Timescale Characteristic (from n-values) Heatmap');

% 6) Timescale from c-values
figure('Name','Timescale from c-values Heatmap');
contourf(X, Y, tscale_char_concentration_from_data_array', 20, figureOpts{:});
shading interp; colorbar;
if useLogX, set(gca,'XScale','log'); end
if useLogY, set(gca,'YScale','log'); end
xlabel(param1_name);
ylabel(param2_name);
title('Timescale Characteristic (from c-values) Heatmap');

% % 7) Timescale from n-values (reservoir samples - Nres)
% figure('Name','Timescale from Reservoir Samples Heatmap');
% contourf(X, Y, tscale_char_from_rcsamples_array', 20, figureOpts{:});
% shading interp; colorbar;
% if useLogX, set(gca,'XScale','log'); end
% if useLogY, set(gca,'YScale','log'); end
% xlabel(param1_name);
% ylabel(param2_name);
% title('Timescale (from Reservoir Samples) Heatmap');

% disp('All done! A single NARMA10 sequence was used for all parameter sweeps.');

% %% ========== 1D Summary Plots vs. Param1 (lines for Param2) ==========
% clrMap = lines(nP2);
%
% % 1) NRMSE(Train) vs Param1
% figure('Name','NRMSE(Train) vs Param1');
% hold on; grid on;
% for i2 = 1:nP2
%     plot(param1_values, nrmse_train_array(:, i2), '-o', ...
%         'Color', clrMap(i2,:), 'LineWidth',2, 'MarkerSize',6, ...
%         'DisplayName',[param2_name,'=',num2str(param2_values(i2))]);
% end
% if useLogX, set(gca,'XScale','log'); end
% xlabel(param1_name);
% ylabel('NRMSE(Train)');
% title(['NRMSE(Train) vs ', param1_name,' (lines for ',param2_name,')']);
% legend('Location','Best');
%
% % 2) NRMSE(Test) vs Param1
% figure('Name','NRMSE(Test) vs Param1');
% hold on; grid on;
% for i2 = 1:nP2
%     plot(param1_values, nrmse_test_array(:, i2), '-o', ...
%         'Color', clrMap(i2,:), 'LineWidth',2, 'MarkerSize',6, ...
%         'DisplayName',[param2_name,'=',num2str(param2_values(i2))]);
% end
% if useLogX, set(gca,'XScale','log'); end
% xlabel(param1_name);
% ylabel('NRMSE(Test)');
% title(['NRMSE(Test) vs ', param1_name,' (lines for ',param2_name,')']);
% legend('Location','Best');
%
% % 3) Mean Receptor Occupation
% figure('Name','Mean Receptor Occupation vs Param1');
% hold on; grid on;
% for i2 = 1:nP2
%     plot(param1_values, mean_rec_occ_ratio_array(:, i2), '-s', ...
%         'Color', clrMap(i2,:), 'LineWidth',2, 'MarkerSize',6, ...
%         'DisplayName',[param2_name,'=',num2str(param2_values(i2))]);
% end
% if useLogX, set(gca,'XScale','log'); end
% xlabel(param1_name);
% ylabel('Occupation Ratio');
% title(['Mean Occupation vs ', param1_name,' (lines for ',param2_name,')']);
% legend('Location','Best');
%
% % 4) Timescale Characteristic (theory) => log-scale in Y
% figure('Name','Timescale Characteristic vs Param1');
% hold on; grid on;
% for i2 = 1:nP2
%     plot(param1_values, tscale_char_array(:, i2), '-d', ...
%         'Color', clrMap(i2,:), 'LineWidth',2, 'MarkerSize',6, ...
%         'DisplayName',[param2_name,'=',num2str(param2_values(i2))]);
% end
% if useLogX, set(gca,'XScale','log'); end
% set(gca,'YScale','log');
% xlabel(param1_name);
% ylabel('Characteristic Timescale (s)');
% title(['Timescale Characteristic vs ', param1_name,' (',param2_name,')']);
% legend('Location','Best');
%
% % 5) Timescale from data (n-values)
% figure('Name','Timescale from n-values vs Param1');
% hold on; grid on;
% for i2 = 1:nP2
%     plot(param1_values, tscale_char_from_data_array(:, i2), '-d', ...
%         'Color', clrMap(i2,:), 'LineWidth',2, 'MarkerSize',6, ...
%         'DisplayName',[param2_name,'=',num2str(param2_values(i2))]);
% end
% if useLogX, set(gca,'XScale','log'); end
% set(gca,'YScale','log');
% xlabel(param1_name);
% ylabel('Timescale (s)');
% title(['Timescale from n-values vs ', param1_name,' (',param2_name,')']);
% legend('Location','Best');
%
% % 6) Timescale from c-values
% figure('Name','Timescale from c-values vs Param1');
% hold on; grid on;
% for i2 = 1:nP2
%     plot(param1_values, tscale_char_concentration_from_data_array(:, i2), '-d', ...
%         'Color', clrMap(i2,:), 'LineWidth',2, 'MarkerSize',6, ...
%         'DisplayName',[param2_name,'=',num2str(param2_values(i2))]);
% end
% if useLogX, set(gca,'XScale','log'); end
% set(gca,'YScale','log');
% xlabel(param1_name);
% ylabel('Timescale (s)');
% title(['Timescale from c-values vs ', param1_name,' (',param2_name,')']);
% legend('Location','Best');
%
% % 7) Timescale from Reservoir Samples
% figure('Name','Timescale from Reservoir Samples vs Param1');
% hold on; grid on;
% for i2 = 1:nP2
%     plot(param1_values, tscale_char_from_rcsamples_array(:, i2), '-o', ...
%         'Color', clrMap(i2,:), 'LineWidth',2, 'MarkerSize',6, ...
%         'DisplayName',[param2_name,'=',num2str(param2_values(i2))]);
% end
% if useLogX, set(gca,'XScale','log'); end
% set(gca,'YScale','log');
% xlabel(param1_name);
% ylabel('Timescale (s)');
% title(['Timescale from Reservoir Samples vs ', param1_name]);
% legend('Location','Best');
