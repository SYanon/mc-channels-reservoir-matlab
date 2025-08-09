%% ========================================================================
%  IPC of Molecular-Communication Reservoir
%
%% ========================================================================

close all;   % Close all figures
clear;       % Clear workspace

rng(0);

%% ========================= 0. User Controls ============================
plot_in_loop       = false;         % Whether to plot intermediate signals
memorywindowlength = 1;             % Typically 1 => direct sampling

% ---------- Primary & Secondary Swept Parameters ----------------------
param1_name   = 'k_on';
%param1_values = [1e-24, 1e-23, 1e-22, 1e-21, 1e-20, 1e-19, 1e-18, 1e-17, 1e-16, 1e-15];
param1_values = [5e-20, 1e-19, 2e-19, 5e-19, 1e-18, 2e-18, 5e-18, 1e-17, 2e-17]; % optimal range for the paper (spans 0.1-0.9 occupation ratio). 


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


% ---------- Receptor & Binding Parameters -----------------------------
N     = 2000;  
k_on  = 1e-18;    
k_off = 1;       
KD    = k_off / k_on;  

% ---------- Reservoir Sampling ----------------------------------------

Nres = 50 * memorywindowlength;  

% ---------- Channel & Time -------------------------------------------
distance      = 10e-6;  
%D             = 1e-11;  
D             = 1e-11;  
T             = 1;      
memory_length = 100 * T;  

% ---------- Data Lengths ---------------------------------------------
washout_symbols     = 200;  
num_training_points = 500;  
num_tot_points      = washout_symbols + num_training_points;

% ---------- Molecule Count Normalization Range ------------------------
N_min = 100;
N_max = 3000;

%% ====================== 1. Allocate Result Arrays =====================
nP1 = length(param1_values);
nP2 = length(param2_values);

% We'll store overall total IPC for each (param1, param2).
IPC_total_array = zeros(nP1, nP2);

% We'll also store the mean receptor occupancy ratio
mean_rec_occ_ratio_array = zeros(nP1, nP2);
std_rec_occ_ratio_array = zeros(nP1, nP2);

% We'll also store the entire "ipcByOrder" and detail table for each param-sweep
ipcByOrder_sweep  = cell(nP1, nP2);
detailTable_sweep = cell(nP1, nP2);

%% ====================== 2. dsmaxPairs for IPC =========================
% Example: measure degrees up to 4 (max degree and delay in legendre polynomial functionals - check IPC papers)
dsmaxPairs = [1,20; 2,3; 3,2; 4,2; 5,2];
%dsmaxPairs = [1,10]; % it will evaluate IPC for (1,1) -> u(t-1), (1,2) -> u(t-1), ..., (1,10) -> u(t-10) 

%% ====================== 3. Parameter Sweeps ===========================
for i1 = 1:nP1
    val1 = param1_values(i1);

    for i2 = 1:nP2
        tic
        val2 = param2_values(i2);

        % 3A. Assign param1 & param2
        switch param1_name
            case 'k_on'
                k_on = val1;
                %k_off = KD*k_on;
                KD    = k_off / k_on; 
            case 'k_off'
                k_off = val1;
                KD    = k_off / k_on; 
            case 'distance'
                distance = val1;
            case 'T'
                T = val1;
                memory_length = 100 * T;
            case 'D'
                D = val1;
            case 'N_max'
                N_max = val1;
            case 'memorywindowlength'
                memorywindowlength = val1;
                Nres = 50 * memorywindowlength; 
            otherwise
                warning('Param1 "%s" not handled.', param1_name);
        end

        switch param2_name
            case 'k_on'
                k_on = val2;
                KD    = k_off / k_on; 
            case 'k_off'
                k_off = val2;
                KD    = k_off / k_on; 
            case 'distance'
                distance = val2;                
            case 'T'
                T = val2;
                memory_length = 100 * T;
            case 'D'
                D = val2;
            case 'N_max'
                N_max = val2;
            case 'memorywindowlength'
                memorywindowlength = val2;
                Nres = 50 * memorywindowlength;  
            otherwise
                warning('Param2 "%s" not handled.', param2_name);
        end

        fprintf('\n==========================================\n');
        fprintf('Sweep: %s=%.3g,   %s=%.3g   (i1=%d, i2=%d)\n', ...
            param1_name, val1, param2_name, val2, i1, i2);
        
        
        %% ------- 4. Generate i.i.d. Random Input in [-1,1] -----------
        input_series = 2*rand(1, num_tot_points) - 1;  % uniform in [-1,1]

        %% ------- 5. Convert Input -> Molecule Count -------------------
        N_i = N_min + ((input_series + 1)/2)*(N_max - N_min);

        %% ------- 6. Simulate the Channel & Receptor Binding -----------
        dt = 0.001;
        t_total  = num_tot_points * T;
        t_values = 0:dt:t_total;
        c_values = zeros(size(t_values));
        n_values = zeros(size(t_values));
        n_values(1) = 0;
        steps_per_symbol = T/dt;

        for iSym = 1:length(N_i)
            t_symbol_start = (iSym - 1)*T;
            t_memory_end   = t_symbol_start + memory_length;
            idxRange = find(t_values > t_symbol_start & t_values <= t_memory_end);
            if isempty(idxRange), continue; end

            for j = idxRange
                tlocal = t_values(j) - t_symbol_start;
                if tlocal > 0
                    c_values(j) = c_values(j) + ...
                        ( N_i(iSym) / ( (4*pi*D*tlocal)^(3/2) ) ) * ...
                        exp(-distance^2/(4*D*tlocal));
                end
            end
        end

        for idxT = 2:length(t_values)
            c_t = c_values(idxT - 1);
            n_t = n_values(idxT - 1);
            dn_dt = k_on*(N - n_t)*c_t - k_off*n_t;
            n_values(idxT) = n_t + dn_dt*dt;
        end
        n_values = n_values / N;

        mean_rec_occ_ratio_array(i1, i2) = mean(n_values(round(end/2):end));
        std_rec_occ_ratio_array(i1, i2) = std(n_values(round(end/2):end));

        % figure; plot(t_values, c_values/KD)
        % title(sprintf('c values - Parameters %s = %f, %s = %f ', param1_name, param1_values(i1), param2_name, param2_values(i2)));
        % 
        % figure; plot(t_values, n_values)
        % title(sprintf('n values - Parameters %s = %f, %s = %f ', param1_name, param1_values(i1), param2_name, param2_values(i2)));

        %% ------- 7. Build Reservoir States from n_values -------------
        sampled_matrix = zeros(Nres, num_training_points);
        sample_indices_to_be_deleted = [];

        for iSym = (washout_symbols+1) : (washout_symbols + num_training_points)
            t_symbol_start = (iSym - 1)*T;
            start_idx = find(t_values >= t_symbol_start, 1);
            if isempty(start_idx)
                sample_indices_to_be_deleted(end+1) = (iSym - washout_symbols); 
                continue; 
            end

            sample_indices = start_idx + (0:(Nres-1))*(steps_per_symbol/Nres)*memorywindowlength;
            sample_indices(sample_indices>length(t_values)) = [];

            colIdx = iSym - washout_symbols;
            if length(sample_indices) >= Nres
                sampled_matrix(:, colIdx) = n_values(sample_indices(1:Nres));
            else
                sample_indices_to_be_deleted(end+1) = colIdx;
            end
        end

        sampled_matrix(:, sample_indices_to_be_deleted) = [];
        input_train = input_series(washout_symbols+1:end);
        input_train(sample_indices_to_be_deleted) = [];

        if isempty(sampled_matrix)
            warning('No valid training samples for %s=%.3g, %s=%.3g => skip.', ...
                param1_name, val1, param2_name, val2);
            IPC_total_array(i1, i2) = NaN;
            continue;
        end

        X     = sampled_matrix.';
        u_vec = input_train(:);

        %% ------- 8. Compute IPC via "computeIPC_legendreNoZeroDup" ----
        [ipcByOrder, totalIPC, detailTable] = ...
            computeIPC_legendreNoZeroDup(X, u_vec, dsmaxPairs);

        IPC_total_array(i1, i2)       = totalIPC;
        ipcByOrder_sweep{i1, i2}     = ipcByOrder;
        detailTable_sweep{i1, i2}    = detailTable;
        
        toc
    end % for i2
end % for i1

%% ====================== 4. Decide on Log Scales =======================
minP1 = min(param1_values);
maxP1 = max(param1_values);
useLogX = (minP1>0) && (maxP1/minP1 > 20);

minP2 = min(param2_values);
maxP2 = max(param2_values);
useLogY = (minP2>0) && (maxP2/minP2 > 20);

% We'll use these figure opts for contourf:
figureOpts = {'LineColor','none','LevelStepMode','auto'};

%% ====================== 5. Always Plot Mean Occupancy Heatmap ========
figure('Name','Mean Receptor Occupancy');
[Xp1, Yp2] = meshgrid(param1_values, param2_values);
Zocc = mean_rec_occ_ratio_array';  % transpose => param1 on x, param2 on y
contourf(Xp1, Yp2, Zocc, 20, figureOpts{:});
shading interp; colorbar;
axis xy;
xlabel(param1_name);
ylabel(param2_name);
title('Heatmap of Mean Receptor Occupancy Ratio');
if useLogX
    set(gca,'XScale','log');
end
if useLogY
    set(gca,'YScale','log');
end


figure('Name','Std of Receptor Occupancy');
[Xp1, Yp2] = meshgrid(param1_values, param2_values);
Zoccstd = std_rec_occ_ratio_array';  % transpose => param1 on x, param2 on y
contourf(Xp1, Yp2, Zoccstd, 20, figureOpts{:});
shading interp; colorbar;
axis xy;
xlabel(param1_name);
ylabel(param2_name);
title('Heatmap of Std of Receptor Occupancy Ratio');
if useLogX
    set(gca,'XScale','log');
end
if useLogY
    set(gca,'YScale','log');
end



%% ====================== 6. Prompt for Plotting Options ===============
%   (1) Heatmap of total IPC
%   (2) Heatmap of partial sum for a specific polynomial degree
%   (3) Heatmap for capacity of a specified factor set
plotChoice = 1;  % user can change if desired

switch plotChoice
    case 1
        %% (i) Heatmap of total IPC
        figure('Name','Total IPC');
        Zipc = IPC_total_array';
        contourf(Xp1, Yp2, Zipc, 20, figureOpts{:});
        shading interp; colorbar;
        axis xy;
        xlabel(param1_name);
        ylabel(param2_name);
        title('Heatmap of Total IPC');
        if useLogX
            set(gca,'XScale','log');
        end
        if useLogY
            set(gca,'YScale','log');
        end

    case 2
        %% (ii) Partial sum for a specific polynomial degree
        degWanted = 5;
        partialSumMatrix = nan(nP1, nP2);

        for i1 = 1:nP1
            for i2 = 1:nP2
                bo = ipcByOrder_sweep{i1, i2};
                if isempty(bo), continue; end
                idx = find([bo.degree]==degWanted,1);
                if ~isempty(idx)
                    partialSumMatrix(i1,i2) = bo(idx).sumCapacity;
                end
            end
        end

        figure('Name',sprintf('IPC Partial Sum (Degree=%d)', degWanted));
        Zpart = partialSumMatrix';
        contourf(Xp1, Yp2, Zpart, 20, figureOpts{:});
        shading interp; colorbar;
        axis xy;
        xlabel(param1_name);
        ylabel(param2_name);
        title(sprintf('IPC Partial Sum for Degree=%d', degWanted));
        if useLogX
            set(gca,'XScale','log');
        end
        if useLogY
            set(gca,'YScale','log');
        end

    case 3
        %% (iii) Single factor-set capacity
        %factorWanted = {[2,1],[1,2]};  % e.g. { (2,1),(1,2) }
        factorWanted = {[1,2]};
        capacityMatrix = nan(nP1,nP2);

        for i1 = 1:nP1
            for i2 = 1:nP2
                dt = detailTable_sweep{i1,i2};
                if isempty(dt), continue; end
                % Compare each factor set in dt.FactorSet with factorWanted
                foundIdx = find(cellfun(@(fs) factorSetsEqual(fs, factorWanted), dt.FactorSet));
                if ~isempty(foundIdx)
                    capacityMatrix(i1,i2) = dt.Capacity(foundIdx(1));
                end
            end
        end

        figure('Name','Single Factor Set Capacity');
        Zfac = capacityMatrix';
        contourf(Xp1, Yp2, Zfac, 20, figureOpts{:});
        shading interp; colorbar;
        axis xy;
        xlabel(param1_name);
        ylabel(param2_name);

        % Build title string automatically from factorWanted
        if isempty(factorWanted)
            % If no factors at all
            strFS = '{}';
        else
            % Convert each numeric vector in factorWanted into a string like '(1,2,3)'
            factorStrings = cellfun(@(v) ...
                ['(' strjoin(arrayfun(@num2str, v, 'UniformOutput', false), ',') ')'], ...
                factorWanted, 'UniformOutput', false);

            % Join all such strings with commas and wrap in braces: '{ (..), (..), ... }'
            strFS = ['{ ' strjoin(factorStrings, ', ') ' }'];
        end

        title(['Capacity of factor set ' strFS]);

        % Optional log scaling:
        if useLogX
            set(gca,'XScale','log');
        end
        if useLogY
            set(gca,'YScale','log');
        end

    case 4
    % Example: we want to sum the capacities of these three sets:
    multipleSets = {
    {[1,1]};        
    {[1,2]}       
        };
    
    % Initialize
    capacityMatrix = nan(nP1, nP2);

    for i1 = 1:nP1
        for i2 = 1:nP2
            dt = detailTable_sweep{i1,i2};
            if isempty(dt)
                continue;
            end

            % Sum the capacities of all requested sets
            sumCap = 0;
            for msIdx = 1:numel(multipleSets)
                thisSet = multipleSets{msIdx};
                
                % Find all matches of 'thisSet' in dt.FactorSet
                foundIdx = find(cellfun(@(fs) factorSetsEqual(fs, thisSet), ...
                                        dt.FactorSet));
                
                % If enumerator lists the same factor set more than once,
                % we sum _all_ of them.  Usually there's only one match,
                % but summing handles duplicates safely.
                if ~isempty(foundIdx)
                    sumCap = sumCap + sum(dt.Capacity(foundIdx));
                end
            end
            
            capacityMatrix(i1,i2) = sumCap;
        end
    end

    figure('Name','Sum of Multiple Factor Sets');
    Zfacsum = capacityMatrix';
    contourf(Xp1, Yp2, Zfacsum, 20, figureOpts{:});
    shading interp; colorbar;
    axis xy;
    xlabel(param1_name);
    ylabel(param2_name);

    % Build a string summarizing all sets
    strAllSets = '';
    for msIdx = 1:numel(multipleSets)
        fsStr = buildFactorSetString(multipleSets{msIdx});
        if msIdx==1
            strAllSets = [strAllSets, fsStr];
        else
            strAllSets = [strAllSets, ', ', fsStr];
        end
    end
    title(['Sum of factor sets: ', strAllSets]);

    if useLogX, set(gca,'XScale','log'); end
    if useLogY, set(gca,'YScale','log'); end


    otherwise
        disp('No additional IPC plotting chosen. (Mean Occupancy was still plotted.)');
end


