import csv
import subprocess
import json
import os

# doses <- c(1,1.5,2,2.5,3)
# observation_time <- 365
# normalization <- 1

#### Simualtion

# n_patients = [50, 100, 200]
# doses = [1, 1.5, 2, 2.5, 3]
# observation_time = 365
# normalization = 1 


simulation_variables = {

    # The name of the expriment
    "experiment_name"      : "test_of_the_process_4",

    # Variables for the simualtion fo the Xmat
    "doses"                : [1, 1.5, 2, 2.5, 3],
    "observation_time"     : [365],

    # Variables for simualtion of the WCEmat
    "n_patients"           : [50,100],
    "normalization"        : [1],
    "weight_function_list" : ["exponential_weight"], #,"bi_linear_weight","constant_weight","early_peak_weight","inverted_u_weight","late_effect_weight"]

    # Variables for the different experiment
    "n_bootstraps_list"    : [1000],#,1000],
    "nknots_list"          : [1,2,3],
    "cutoff_list"          : [180],
    "constraint"           : ["Right"]#[None, "Right"]

}


path_simulation_results = "Simulation_results/" + simulation_variables["experiment_name"]
if not os.path.exists(path_simulation_results):
    os.mkdir(path_simulation_results)


path_model_results = path_simulation_results + "/models"
if not os.path.exists(path_model_results):
    os.mkdir(path_model_results)
    
path_Xmat = path_simulation_results + "/Xmat"
if not os.path.exists(path_Xmat):
    os.mkdir(path_Xmat)

path_WCEmat = path_simulation_results + "/WCEmat"
if not os.path.exists(path_WCEmat):
    os.mkdir(path_WCEmat)


simulation_variables_str = json.dumps(simulation_variables)

command = ['Rscript', 'simulate_WCEmat.r', simulation_variables_str]
print(command)
# subprocess.run(command)
output_lists = subprocess.run(command, capture_output=True, text=True)
print("#######################")
print(output_lists)

# Command to run the second Python script
command = ['python', 'analysis.py', simulation_variables_str]

# Execute the command
subprocess.run(command)

# Command to run the second Python script
command = ['python', 'post_clep_analysis_error_based.py', simulation_variables_str]

# Execute the command
subprocess.run(command)

