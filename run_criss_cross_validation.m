% =========================================================================
% run_criss_cross_validation.m (v2 - Automated)
%
% DESCRIPTION:
%   A fully automated script that first finds the optimal hyperparameters
%   for three distinct tasks via Bayesian Optimization, and then immediately
%   uses those parameters to perform a criss-cross validation.
%
%   PHASE 1: Run 50 Bayesian trials for each task to find the best params.
%   PHASE 2: Run 9 simulations to test each best param set on all tasks.
%   PHASE 3: Generate a 3x3 plot and a summary heatmap of the results.
%
% =========================================================================

%% 0. Initialize Environment
close all;
clear;
rng('default');
fprintf('Starting Fully Automated Optimization and Criss-Cross Validation...\n\n');

%% ========================================================================
%   PHASE 1: DYNAMICALLY FIND BEST HYPERPARAMETERS
% ========================================================================

% --- 1a. Define Tasks and Optimization Settings ---
tasks = {
    struct('name', 'Forecasting (MG)', 'objective_func', @objectiveFunction_mg, 'config_func', @get_default_config_mg, 'loader_func', @load_mg_data);
    struct('name', 'Transformation (Sine-Sq)', 'objective_func', @objectiveFunction_sine, 'config_func', @get_default_config_sine, 'loader_func', @load_sine_data);
    struct('name', 'Hybrid (MG-Cubed)', 'objective_func', @objectiveFunction_mg_cubed, 'config_func', @get_default_config_mg_cubed, 'loader_func', @load_mg_cubed_data);
};

num_bayes_trials = 50;
best_param_sets = cell(3, 1);

vars = [
    optimizableVariable('k_on', [5e-20, 2e-17], 'Transform', 'log');
    optimizableVariable('k_off', [0.1, 10]);
    optimizableVariable('T', [0.5, 2]);
    optimizableVariable('distance', [1e-6, 50e-6], 'Transform', 'log');
    optimizableVariable('memorywindowlength', [1, 5], 'Type', 'integer');
    optimizableVariable('N_max', [200, 20000]);
    optimizableVariable('D', [0.5e-11, 2e-10], 'Transform', 'log');
];

% --- 1b. Run Optimization Sequentially for Each Task ---
for i = 1:length(tasks)
    task = tasks{i};
    fprintf('===== PHASE 1: Optimizing for Task: %s =====\n', task.name);
    
    config = task.config_func();
    data = task.loader_func(config);
    objFun = @(params) task.objective_func(params, data, config);
    
    results = bayesopt(objFun, vars, ...
        'MaxObjectiveEvaluations', num_bayes_trials, ...
        'IsObjectiveDeterministic', true, ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'Verbose', 1);
        
    best_param_sets{i} = results.XAtMinObjective;
    fprintf('Best parameters found for %s.\n', task.name);
    disp(best_param_sets{i});
    fprintf('\n');
end

fprintf('===== PHASE 1 COMPLETE: All optimal parameter sets found. =====\n\n');

%% ========================================================================
%   PHASE 2: AUTOMATED CRISS-CROSS VALIDATION
% ========================================================================
fprintf('===== PHASE 2: Starting Automated Criss-Cross Validation =====\n');

param_sets_for_validation = {
    struct('name', 'MG Params', 'params', table2struct(best_param_sets{1}));
    struct('name', 'Sine-Sq Params', 'params', table2struct(best_param_sets{2}));
    struct('name', 'MG-Cubed Params', 'params', table2struct(best_param_sets{3}));
};

results_nrmse = zeros(3, 3);
results_predictions = cell(3, 3);
ground_truths = cell(3, 1);

for i_task = 1:3
    task_info = tasks{i_task};
    config = task_info.config_func();
    data = task_info.loader_func(config);
    ground_truths{i_task} = data.test_target;
    
    fprintf('--- Running Simulations for Task: %s ---\n', task_info.name);
    
    for j_params = 1:3
        param_info = param_sets_for_validation{j_params};
        fprintf('   with %s...\n', param_info.name);
        
        run_config = config;
        fields = fieldnames(param_info.params);
        for k = 1:length(fields)
            run_config.(fields{k}) = param_info.params.(fields{k});
        end
        run_config.Nres = config.Nres * run_config.memorywindowlength;
        
        [~, r_states] = run_channel_simulation(data.input_series, run_config);
        W_out = train_readout(r_states, data.train_target, run_config);
        [~, nrmse, y_pred] = test_readout(W_out, r_states, data, run_config);
        
        results_nrmse(i_task, j_params) = nrmse;
        results_predictions{i_task, j_params} = y_pred;
    end
end
fprintf('\n===== PHASE 2 COMPLETE: All 9 validation simulations finished. =====\n\n');

%% ========================================================================
%   PHASE 3: VISUALIZATION
% ========================================================================
fprintf('===== PHASE 3: Generating final plots... =====\n');

% --- 3a. Generate 3x3 Criss-Cross Plot ---
figure('Name', 'Criss-Cross Validation Results', 'Position', [50, 50, 1200, 900]);
sgtitle('Criss-Cross Validation: Performance of Optimized Parameters on All Tasks', 'FontSize', 16, 'FontWeight', 'bold');

for i_task = 1:3
    for j_params = 1:3
        plot_idx = (i_task - 1) * 3 + j_params;
        subplot(3, 3, plot_idx);
        
        plot(ground_truths{i_task}, 'k-', 'LineWidth', 2, 'DisplayName', 'Ground Truth');
        hold on;
        plot(results_predictions{i_task, j_params}, 'r-', 'LineWidth', 1, 'DisplayName', 'Prediction');
        hold off;
        grid on;
        
        title(sprintf('NRMSE = %.4f', results_nrmse(i_task, j_params)), 'FontSize', 12);
        if i_task == 1
            xlabel(param_sets_for_validation{j_params}.name, 'FontWeight', 'bold');
        end
        if j_params == 1
            ylabel(tasks{i_task}.name, 'FontWeight', 'bold');
        end
        
        if i_task == j_params
            ax = gca;
            ax.XColor = [0 0.5 0]; ax.YColor = [0 0.5 0];
            ax.LineWidth = 1.5;
        end
    end
end

% --- 3b. Generate Summary NRMSE Heatmap ---
figure('Name', 'NRMSE Summary Heatmap');
task_labels = cellfun(@(c) c.name, tasks, 'UniformOutput', false);
param_labels = cellfun(@(c) c.name, param_sets_for_validation, 'UniformOutput', false);

h = heatmap(param_labels, task_labels, results_nrmse, 'Colormap', flipud(parula));
h.Title = 'NRMSE Performance: Task vs. Parameter Set';
h.XLabel = 'Parameter Set Used';
h.YLabel = 'Task Performed';
h.CellLabelFormat = '%.3f';

fprintf('\nValidation complete. Plots generated.\n');


% =========================================================================
%                    --- LOCAL HELPER FUNCTIONS ---
% =========================================================================
% --- Functions for Mackey-Glass Forecasting Task ---
function nrmse_test = objectiveFunction_mg(params, data, config); run_config=config; run_config.k_on=params.k_on; run_config.k_off=params.k_off; run_config.T=params.T; run_config.distance=params.distance; run_config.memorywindowlength=params.memorywindowlength; run_config.N_max=params.N_max; run_config.D=params.D; run_config.Nres=config.Nres*run_config.memorywindowlength; [~,r_states]=run_channel_simulation(data.input_series,run_config); W_out=train_readout(r_states,data.train_target,run_config); [~,nrmse_test]=test_readout(W_out,r_states,data,run_config); end
function config = get_default_config_mg(); config.task_name='mg_forecasting'; config.N=500; config.N_min=100; config.lambda=1e-6; config.Nres=50; config.predictlength=6; config.washout1=500; config.num_train_points=500; config.wheretostarttest=1200; config.num_test_points=500; config.washout2=config.wheretostarttest-(config.washout1+config.num_train_points); config.num_tot_points=config.washout1+config.washout2+config.num_train_points+config.num_test_points; config.dt=0.001; config.memlengthsweep=100; config.MG_FILENAME='MGseries_RK4_tau17_beta0.20_gamma0.1_n10_len5000_dt1.0.mat'; end
function data = load_mg_data(config); loaded=load(config.MG_FILENAME,'mackey_glass_series'); series_raw=loaded.mackey_glass_series; seg=series_raw(1:config.num_tot_points+config.predictlength); norm_seg=(seg-min(seg))/(max(seg)-min(seg)); data.input_series=norm_seg(1:end-config.predictlength); data.target_series=norm_seg((config.predictlength+1):end); data.train_target=data.target_series(config.washout1+1:config.washout1+config.num_train_points); data.test_target=data.target_series(config.wheretostarttest+1:config.wheretostarttest+config.num_test_points); end
% --- Functions for Sine-to-Square Transformation Task ---
function nrmse_test = objectiveFunction_sine(params, data, config); run_config=config; run_config.k_on=params.k_on; run_config.k_off=params.k_off; run_config.T=params.T; run_config.distance=params.distance; run_config.memorywindowlength=params.memorywindowlength; run_config.N_max=params.N_max; run_config.D=params.D; run_config.Nres=config.Nres*run_config.memorywindowlength; [~,r_states]=run_channel_simulation(data.input_series,run_config); W_out=train_readout(r_states,data.train_target,run_config); [~,nrmse_test]=test_readout(W_out,r_states,data,run_config); end
function config = get_default_config_sine(); config.task_name='sine_transform'; config.N=500; config.N_min=100; config.lambda=1e-6; config.Nres=100; config.washout1=500; config.num_train_points=2000; config.wheretostarttest=3000; config.num_test_points=1000; config.washout2=config.wheretostarttest-(config.washout1+config.num_train_points); config.num_tot_points=config.washout1+config.washout2+config.num_train_points+config.num_test_points; config.dt=0.001; config.memlengthsweep=100; config.SINE_FILENAME='SINEseries_len5000_p25.mat'; end
function data = load_sine_data(config); loaded=load(config.SINE_FILENAME,'input_sine_series','target_square_series'); data.input_series=loaded.input_sine_series(1:config.num_tot_points); data.target_series=loaded.target_square_series(1:config.num_tot_points); data.train_target=data.target_series(config.washout1+1:config.washout1+config.num_train_points); data.test_target=data.target_series(config.wheretostarttest+1:config.wheretostarttest+config.num_test_points); end
% --- Functions for Mackey-Glass Cubed Hybrid Task ---
function nrmse_test = objectiveFunction_mg_cubed(params, data, config); run_config=config; run_config.k_on=params.k_on; run_config.k_off=params.k_off; run_config.T=params.T; run_config.distance=params.distance; run_config.memorywindowlength=params.memorywindowlength; run_config.N_max=params.N_max; run_config.D=params.D; run_config.Nres=config.Nres*run_config.memorywindowlength; [~,r_states]=run_channel_simulation(data.input_series,run_config); W_out=train_readout(r_states,data.train_target,run_config); [~,nrmse_test]=test_readout(W_out,r_states,data,run_config); end
function config = get_default_config_mg_cubed(); config.task_name='mg_cubed_hybrid'; config.N=500; config.N_min=100; config.lambda=1e-6; config.Nres=50; config.washout1=500; config.num_train_points=500; config.wheretostarttest=1200; config.num_test_points=500; config.washout2=config.wheretostarttest-(config.washout1+config.num_train_points); config.num_tot_points=config.washout1+config.washout2+config.num_train_points+config.num_test_points; config.dt=0.001; config.memlengthsweep=100; config.MG_CUBED_FILENAME='MGCubed_series_k10.mat'; end
function data = load_mg_cubed_data(config); loaded=load(config.MG_CUBED_FILENAME,'input_series','target_series'); data.input_series=loaded.input_series(1:config.num_tot_points); data.target_series=loaded.target_series(1:config.num_tot_points); data.train_target=data.target_series(config.washout1+1:config.washout1+config.num_train_points); data.test_target=data.target_series(config.wheretostarttest+1:config.wheretostarttest+config.num_test_points); end
% --- COMMON SIMULATION AND LEARNING FUNCTIONS ---
function [sim, reservoir_states] = run_channel_simulation(input_series, config); inp_min=min(input_series); inp_max=max(input_series); N_i=config.N_min+(input_series-inp_min)/(inp_max-inp_min)*(config.N_max-config.N_min); t_total=length(N_i)*config.T; sim.time=0:config.dt:t_total; sim.concentration=zeros(size(sim.time)); Tpeak=config.distance^2/(6*config.D); memory_length=round((config.memlengthsweep*Tpeak)/config.T)*config.T; for iSym=1:length(N_i); t_symbol_start=(iSym-1)*config.T; t_memory_end=t_symbol_start+memory_length; idxRange=find(sim.time>t_symbol_start & sim.time<=t_memory_end); if isempty(idxRange), continue; end; t_local=sim.time(idxRange)-t_symbol_start; pulse=(N_i(iSym)./((4*pi*config.D*t_local).^(3/2))).*exp(-config.distance^2./(4*config.D*t_local)); sim.concentration(idxRange)=sim.concentration(idxRange)+pulse; end; sim.occupation=zeros(size(sim.time)); for idxT=2:length(sim.time); c_t=sim.concentration(idxT-1); n_t=sim.occupation(idxT-1); dn_dt=config.k_on*(config.N-n_t*config.N)*c_t-config.k_off*n_t*config.N; sim.occupation(idxT)=sim.occupation(idxT-1)+(dn_dt/config.N)*config.dt; end; steps_per_symbol=config.T/config.dt; num_symbols=config.num_tot_points; reservoir_states=zeros(config.Nres,num_symbols); for iSym=1:num_symbols; t_symbol_start=(iSym-1)*config.T; start_idx=round(t_symbol_start/config.dt)+1; sample_indices_float=start_idx-(config.memorywindowlength-1)*steps_per_symbol+(0:(config.Nres-1))*(steps_per_symbol/config.Nres)*config.memorywindowlength; sample_indices=round(sample_indices_float); if all(sample_indices>0 & sample_indices<=length(sim.occupation)); reservoir_states(:,iSym)=sim.occupation(sample_indices); end; end; end
function W_out = train_readout(reservoir_states, train_target, config); train_states=reservoir_states(:,config.washout1+1:config.washout1+config.num_train_points); X_train=[train_states;ones(1,config.num_train_points)]; y_train=train_target(:); W_out=pinv(X_train*X_train'+config.lambda*eye(size(X_train,1)))*(X_train*y_train); end
function [nrmse_train, nrmse_test, y_test_hat] = test_readout(W_out, reservoir_states, data, config); train_states=reservoir_states(:,config.washout1+1:config.washout1+config.num_train_points); X_train=[train_states;ones(1,config.num_train_points)]; y_train_hat=X_train'*W_out; nrmse_train=sqrt(mean((y_train_hat-data.train_target(:)).^2))/std(data.train_target); test_indices=config.wheretostarttest+1:config.wheretostarttest+config.num_test_points; test_states=reservoir_states(:,test_indices); X_test=[test_states;ones(1,config.num_test_points)]; y_test_hat=X_test'*W_out; nrmse_test=sqrt(mean((y_test_hat-data.test_target(:)).^2))/std(data.test_target); end