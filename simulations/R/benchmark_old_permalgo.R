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



n_patients_list = c(100,100,200,5000,1000,2000,5000,10000,20000,50000,100000)

max_time = 365
doses = c(1,1.5,2,2.5,3)
HR_target = 1.5
constraint = "R"
scenario = "exponential_weight"

n_bootstraps = 1000
cutoff = 180
n_knots = 1

experiment_name <- "2024-04-22 : benchmark CPU old permalgo"

computation_type <- "CPU old permalgo"

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
                       simulation_time = integer(number_conditions)
                       )

number_analyzed_conditions = 0


for (n_patients in n_patients_list){

    number_analyzed_conditions = number_analyzed_conditions + 1
    df_results$computation_type[number_analyzed_conditions] = computation_type
    df_results$n_patients[number_analyzed_conditions] = n_patients


    start_simulation_time = Sys.time()
    dataset  <- generate_dataset_batch(
            doses = doses,
            observation_time = max_time,
            n_patients = n_patients,
            scenario = scenario,
            cutoff = cutoff,
            HR_target = HR_target,
            batchsize = 10000000     
        )
    end_simulation_time = Sys.time()
    elapsed_simulation_time = end_simulation_time - start_simulation_time
    df_results$simulation_time[number_analyzed_conditions] = elapsed_simulation_time
    
    write.csv(df_results, file_result_path)


}

print(df_results)