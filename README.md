# Molecular Communication Channel as a Physical Reservoir Computer 🔬

This repository contains the MATLAB and Smoldyn code for the paper "A Molecular Communication Channel as a Task-Adaptive Physical Reservoir Computer." The project reconceptualizes the intrinsic memory and nonlinear dynamics of a molecular communication (MC) channel as computational resources.

We frame a canonical point-to-point MC channel, featuring ligand diffusion and reversible receptor binding, as a physical reservoir computer (PRC). The system's performance is evaluated on its ability to perform complex temporal processing on standard chaotic time-series benchmarks and nonlinear mapping tasks.

The workflows are implemented using two complementary simulation approaches:
* **Deterministic Mean-Field Modeling** in MATLAB for rapid parameter analysis.
* **Particle-Based Spatial Stochastic Simulations** using the [Smoldyn](http://www.smoldyn.org/) simulator to account for intrinsic molecular noise.

---
## What's New?

* **Task-Adaptability Framework:** The project now includes workflows for forecasting, nonlinear transformation, and combined tasks to demonstrate the reservoir's reconfigurable nature.
* **Expanded Benchmarks:** Added Lorenz-63 (forecasting) and Wiener-Hammerstein (combined) tasks to the analysis.
* **Bayesian Optimization:** Scripts for efficient hyperparameter optimization using Bayesian methods have been added for all major tasks.
* **Code Refinements:** The codebase has been cleaned and organized for better readability and efficiency.

---
## Required Dependencies

* **MATLAB:** The primary environment for all analysis and orchestration scripts. The Bayesian optimization scripts require the **Statistics and Machine Learning Toolbox**.
* **Smoldyn:** An external, open-source particle simulator. It must be installed and accessible from the system's command line for the stochastic simulation scripts to function.

---
## Overall Workflow

The project is divided into workflows for three distinct classes of computational tasks—**Forecasting**, **Nonlinear Mapping**, and **Combined Tasks**—followed by general analysis and optimization workflows.

### Forecasting Tasks
These workflows evaluate the reservoir's ability to predict the future values of chaotic time-series, a task that heavily relies on the system's *fading memory*.

#### Workflow 1: Mackey-Glass Prediction Task 📈
This workflow evaluates the reservoir's ability to predict the future values of the chaotic Mackey-Glass time series.

**Scripts & Execution Order:**
1.  **`createMGseries.m`**
    * **Purpose:** Generates the Mackey-Glass time-series data.
    * **Note:** Run this first to create the `.mat` data file.
2.  **`plot_mg_numerical_analysis.m` or `mg_numerical_analysis.m`**
    * **Purpose:** Sweeps parameters using the fast deterministic model to analyze performance. The `plot_` version includes in-loop visualizations.
3.  **(Optional) `NewMG_Generate_Smoldyn_Data.m`**
    * **Purpose:** Runs a full Smoldyn simulation to generate high-fidelity reservoir states for stochastic analysis.

---
#### Workflow 2: NARMA10 Prediction Task 🧠
This workflow evaluates the reservoir on the NARMA10 benchmark, which requires capturing longer-term dependencies.

**Scripts & Execution Order:**
1.  **`plot_narma_numerical_analysis.m`**
    * **Purpose:** Generates NARMA10 data and sweeps parameters using the deterministic model.
2.  **(Optional) `NARMA10_Generate_Smoldyn_Data.m`**
    * **Purpose:** Runs a full Smoldyn simulation for the NARMA10 task.

---
#### Workflow 3: Lorenz-63 Prediction Task 🦋
This workflow evaluates the reservoir's ability to predict the future states of the 3D Lorenz attractor, a challenging chaotic system.

**Scripts & Execution Order:**
1.  **`createLorenzseries.m`**
    * **Purpose:** Generates the Lorenz-63 time-series data (X, Y, and Z coordinates).
    * **Note:** Run this first to create the `lorenz_series_...mat` data file.
2.  **`plot_lorenz_numerical_analysis.m`**
    * **Purpose:** Performs a parameter sweep using the deterministic model for the Lorenz prediction task.
3.  **`optimize_lorenz_hyperparams_bayesopt.m`**
    * **Purpose:** Uses Bayesian optimization to find the optimal biophysical parameters for Lorenz forecasting.
4.  **(Optional) `lorenz_generate_smoldyn_data.m`**
    * **Purpose:** Runs a full Smoldyn simulation to generate high-fidelity reservoir states for stochastic analysis of the Lorenz task.

---
### Nonlinear Mapping Tasks
These workflows test the reservoir's capacity to learn complex, static transformations, a task that primarily tests the system's *nonlinearity*.

#### Workflow 4: Sine-to-Square Transformation Task 〰️🔲
This workflow tests the reservoir's ability to perform a complex nonlinear mapping from a sine wave input to a square wave target.

**Scripts & Execution Order:**
1.  **`createSINEseries.m`**
    * **Purpose:** Generates the sine wave input and square wave target data.
    * **Note:** Run this first to create the `SINEseries_...mat` data file.
2.  **`plot_sine_numerical_analysis.m`**
    * **Purpose:** Performs a parameter sweep using the deterministic model.
3.  **`optimize_sine_hyperparams_bayesopt.m`**
    * **Purpose:** Uses Bayesian optimization to efficiently find the optimal biophysical parameters for this task.

---
#### Workflow 5: Sine-to-Sawtooth Transformation Task 〰️📈
This workflow evaluates the performance on converting a sine wave to a sawtooth wave.

**Scripts & Execution Order:**
1.  **`createSAWseries.m`**
    * **Purpose:** Generates the sine wave input and sawtooth wave target data.
    * **Note:** Run this first to create the `SAWseries_...mat` data file.
2.  **`optimize_saw_hyperparams_bayesopt.m`**
    * **Purpose:** Uses Bayesian optimization to find the best-performing hyperparameters for this task.

---
### Combined Forecasting & Nonlinear Transformation Tasks

#### Workflow 6: Wiener-Hammerstein (WH) Prediction Task ⚙️
This workflow evaluates the reservoir on a benchmark that requires a combination of both memory and nonlinear transformation capabilities.

**Scripts & Execution Order:**
1.  **`createWHseries.m`**
    * **Purpose:** Generates the Wiener-Hammerstein time-series data.
    * **Note:** Run this first to create the `WHseries_...mat` data file.
2.  **`plot_wh_numerical_analysis.m`**
    * **Purpose:** Performs a parameter sweep using the deterministic model for the WH task.
3.  **`optimize_wh_hyperparams_bayesopt.m`**
    * **Purpose:** Uses Bayesian optimization to find the optimal biophysical parameters for this combined task.

---
### General Analysis & Optimization

#### Workflow 7: Information Processing Capacity (IPC) Analysis 📊
This workflow provides a task-independent characterization of the reservoir's computational power.

**Scripts & Execution Order:**
* **`IPC_sweep.m` (Main Script)**
    * **Purpose:** Performs the complete IPC analysis.
    * **Core Functionality:** Calls `computeIPC_legendreNoZeroDup.m`.
    * **Output:** Heatmap plots showing the Total IPC.

---
#### Workflow 8: Bayesian Hyperparameter Optimization 🎯
This workflow uses Bayesian optimization to find the optimal hyperparameters for any given task.

**Scripts:**
* `optimize_mg_hyperparams_bayesopt.m`: For Mackey-Glass.
* `optimize_narma_hyperparams_bayesopt.m`: For NARMA10.
* `optimize_lorenz_hyperparams_bayesopt.m`: For Lorenz-63.
* `optimize_wh_hyperparams_bayesopt.m`: For Wiener-Hammerstein.
* *(See Nonlinear Mapping workflows for other task-specific optimization scripts).*

---
#### Workflow 9: Kathy Ludge Post-Processing Analysis 🛠️
This workflow applies advanced post-processing methods to the reservoir's output for improved performance, particularly on noisy stochastic data.

**Scripts:**
* **`kl_mg_numerical_analysis.m`** and **`kl_mg_stochastic_sweep.m`**: For the Mackey-Glass task.
* **`kl_narma_numerical_analysis.m`** and **`kl_narma_stochastic_sweep.m`**: For the NARMA10 task.
* **`optimize_kl_mg_hyperparams.m`** and **`optimize_kl_narma_hyperparams.m`**: For optimizing the hyperparameters of the post-processing methods.
