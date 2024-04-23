library(WCE)
library(boot)
library(jsonlite)
library(devtools)
library(dplyr)

options(scipen = 999)

devtools::load_all("../../../survivalGPU/R")

source("functions_data_simulation.R")
source("weight_functions.R")
source("experiment_parameters.R")
source("simulation_and_analysis.R")



n_patients_list = c(100,100,200,500,1000,2000,5000,10000,20000,50000,100000)

max_time = 365
doses = c(1,1.5,2,2.5,3)
HR_target = 1.5
constraint = "R"
scenario = "exponential_scenario"

n_bootstraps = 1000
cutoff = 180
n_knots = 1

experiment_name <- "2024-04-22 : benchmark CPU no bootstraps"

computation_type <- "CPU"

result_dict_path = file.path("../Simulation_results",experiment_name)

if (dir.exists(result_dict_path)){
    stop("this experiment already exist")
}else{
    dir.create(result_dict_path)
}

result_df_path = file.path("../Simulation_results",experiment_name)
file_result_name = paste0(experiment_name,".csv")
file_result_path = file.path(result_df_path,file_result_name)


number_conditions = length(n_patients_list)

df_results = data.frame(n_patients = integer(number_conditions),
                       computation_type = integer(number_conditions),
                       simulation_time = integer(number_conditions),
                       computation_time_no_bootstraps = integer(number_conditions)
                    #    computation_time_1000_bootstraps =integer(number_conditions)
                       )

number_analyzed_conditions = 0


for (n_patients in n_patients_list){

    number_analyzed_conditions = number_analyzed_conditions + 1
    df_results$computation_type[number_analyzed_conditions] = computation_type
    df_results$n_patients[number_analyzed_conditions] = n_patients


    start_simulation_time = Sys.time()
    dataset = simulate_dataset(max_time, n_patients, doses, scenario, HR_target, constraint)
    end_simulation_time = Sys.time()
    elapsed_simulation_time <- as.numeric(difftime(end_simulation_time, start_simulation_time, units = "secs"))

    df_results$simulation_time[number_analyzed_conditions] = elapsed_simulation_time
    
    start_no_bootstraps_time = Sys.time()
    wce_model_without_bootstraps = modelize_dataset(max_time, n_patients, cutoff, n_bootstraps =1, n_knots =n_knots,dataset =dataset, constraint = constraint)
    end_no_bootstraps_time = Sys.time()
    elapsed_no_bootstraps_time <- as.numeric(difftime(end_no_bootstraps_time, start_no_bootstraps_time, units = "secs"))
    df_results$computation_time_no_bootstraps[number_analyzed_conditions] = elapsed_no_bootstraps_time

    # start_1000_bootstraps_time = Sys.time()
    # wce_model_1000_bootstraps = modelize_dataset(max_time, n_patients, cutoff, n_bootstraps, n_knots,dataset, constraint)
    # end_1000_bootstraps_time = Sys.time()
    # elapsed_1000_bootstraps_time <- as.numeric(difftime(end_1000_bootstraps_time, start_1000_bootstraps_time, units = "secs"))
    # df_results$computation_time_1000_bootstraps[number_analyzed_conditions] = elapsed_1000_bootstraps_time

    write.csv(df_results, file_result_path)


}

print(df_results)