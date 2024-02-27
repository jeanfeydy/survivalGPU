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


simulation_parameters = {

    # The name of the expriment
    "experiment_name"      : "multiple weights : 100 - 10000",

    # Parameters for the simualtion fo the Xmat
    "doses"                : [1, 1.5, 2, 2.5, 3],
    "observation_time"     : [365],

    # Parameters for simualtion of the WCEmat
    "n_patients"           : [100,1000,10000], #,30000,40000,50000],
    "normalization"        : [1],
    "weight_function_list" : ["bi_linear_weight","early_peak_weight","early_peak_weight","inverted_u_weight","null_weight"], #"exponential_weight",#"constant_weight""inverted_u_weight","late_effect_weight"]

    # Parameters for the different experiment
    "n_bootstraps_list"    : [1000],
    "nknots_list"          : [1,2,3],
    "cutoff_list"          : [180],
    "constraint"           : ["Right","None"]

}


path_simulation_results = "Simulation_results/" + simulation_parameters["experiment_name"]
if not os.path.exists(path_simulation_results):
    os.mkdir(path_simulation_results)


path_model_results = path_simulation_results + "/models"
if not os.path.exists(path_model_results):
    os.mkdir(path_model_results)
    
# path_Xmat = path_simulation_results + "/Xmat"
# if not os.path.exists(path_Xmat):
#     os.mkdir(path_Xmat)

# path_WCEmat = path_simulation_results + "/WCEmat"
# if not os.path.exists(path_WCEmat):
#     os.mkdir(path_WCEmat)


simulation_parameters_json = json.dumps(simulation_parameters)


with open("Simulation_results/simulation_parameters.json", "w") as outfile:
    outfile.write(simulation_parameters_json)

# command = ['Rscript', 'simulate_WCEmat.r', simulation_variables_str]
# print(command)
# # subprocess.run(command)
# output_lists = subprocess.run(command, capture_output=True, text=True)
# print("#######################")
# print(output_lists)

# # Command to run the second Python script
# command = ['python', 'analysis.py', simulation_variables_str]

# # Execute the command
# subprocess.run(command)

# # Command to run the second Python script
# command = ['python', 'post_clep_analysis_error_based.py', simulation_variables_str]

# # Execute the command
# subprocess.run(command)

