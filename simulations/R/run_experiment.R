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

########################################## EXPERIMENTS DATA ##########################################

#Generate parameters
experiment_parameters = generate_parameters()


# get experiment paramters
experiment_name <- experiment_parameters$experiment_name
number_of_simulations <- experiment_parameters$number_of_simulations
save_models <- experiment_parameters$save_models




# Get simulation parameters 
HR_target_list= experiment_parameters$simulation$HR_target_list
scenario_functions_list= experiment_parameters$simulation$scenario_functions_list
n_patients_list= experiment_parameters$simulation$n_patients_list
max_time_list= experiment_parameters$simulation$max_time_list
doses_list = experiment_parameters$simulation$doses_list
binarization_dose_list = experiment_parameters$simulation$binarization_dose_list


# Get analysis parameters 
n_knots_list= experiment_parameters$analysis$n_knots_list
constraint_list = experiment_parameters$analysis$constraint_list
cutoff_list= experiment_parameters$analysis$cutoff_list
n_bootstraps_list = experiment_parameters$analysis$n_bootstraps_list


########################################## EXPERIMENTS SCRIPT ########################################

print("#######")


simulation_id = list()

HR_target_result = list()
scenario_functions_result = list()
n_patients_result = list()
max_time_result = list()
doses_result = list()
binarization_dose_result = list()
n_knots_result = list() 
constraint_result = list() 
cutoff_result = list() 
n_bootstraps_result = list() 

BIC_results = list()
HR_results = list()
lower_IC_results = list()
higher_IC_results = list()



combinations_simulations_parameters <- expand.grid(HR_target = HR_target_list,
                                                   scenario_function = scenario_functions_list, 
                                                   max_time = max_time_list,
                                                   n_patients = n_patients_list,
                                                   doses = doses_list,
                                                   binarization_dose_list = binarization_dose_list)

combinations_analysis_parameters <- expand.grid(n_knots_list = n_knots_list,
                                                n_bootstraps = n_bootstraps_list,
                                                constraint_list = constraint_list, 
                                                cutoff = cutoff_list, 
                                                n_bootstraps = n_bootstraps_list)



number_line_df = nrow(combinations_simulations_parameters) * nrow(combinations_analysis_parameters)
print(number_line_df)



results_df = data.frame(simulation_id = integer(number_line_df),
                        HR_target = integer(number_line_df),
                        scenario_functions = character(number_line_df),
                        n_patients = integer(number_line_df),
                        max_time = integer(number_line_df),
                        doses = integer(number_line_df),
                        binarization_dose = character(number_line_df),
                        n_knots = integer(number_line_df),
                        constraint = character(number_line_df),
                        cutoff = integer(number_line_df),
                        n_bootstraps = integer(number_line_df),
                        BIC = integer(number_line_df),
                        HR = integer(number_line_df),
                        lower_IC = integer(number_line_df),
                        higher_IC = integer(number_line_df))

if (save_models == TRUE){
    results_df$path_model <- character(number_line_df)
    print(results_df["path_model"])

}

print(combinations_simulations_parameters)

print(combinations_analysis_parameters)




result_dict_path = file.path("../Simulation_results",experiment_name)

if (dir.exists(result_dict_path)){
    stop("this experiment already exist")
}else{
    dir.create(result_dict_path)
}

result_df_path = file.path("../Simulation_results",experiment_name)
file_result_name = paste0("analyzed_",experiment_name,".csv")
file_result_path = file.path(result_df_path,file_result_name)

condition_simulation_id <- 0
number_of_analyzed_models <- 0

for (i in 1:nrow(combinations_simulations_parameters)){

    condition_simulation_id <- condition_simulation_id + 1

    for(simulation_number in 1:number_of_simulations )

        HR_target <- combinations_simulations_parameters$HR_target[i]
        scenario_function <- combinations_simulations_parameters$scenario_function[i]
        n_patients <- combinations_simulations_parameters$n_patients[i]
        max_time <- combinations_simulations_parameters$max_time[i]
        doses <- combinations_simulations_parameters$doses[i]
        binarization_dose <- combinations_simulations_parameters$binarization_dose[i]


        simulated_dataset <- simulate_dataset(max_time, n_patients, doses, scenario_function, HR_target)

        simulation_id <- paste0(experiment_name,"_condition-",as.character(condition_simulation_id),"_",as.character(simulation_number))

 
        for(j in 1:nrow(combinations_analysis_parameters)){

            number_of_analyzed_models <- number_of_analyzed_models + 1
            print("##################")
            print(number_of_analyzed_models)

            n_knots <- combinations_analysis_parameters$n_knots_list[j]
            n_bootstraps <- combinations_analysis_parameters$n_bootstraps[j]
            constraint <- combinations_analysis_parameters$constraint_list[j]
            cutoff <- combinations_analysis_parameters$cutoff[j]
            n_bootstraps <- combinations_analysis_parameters$n_bootstraps[j]


            wce_model <- modelize_dataset(max_time, n_patients, cutoff, n_bootstraps, n_knots,simulated_dataset, constraint)
            HR <- analyze_model(wce_model, n_patients, cutoff)


            BIC <- mean(wce_model$info.criterion)




            results_df$simulation_id[number_of_analyzed_models] <- simulation_id
            results_df$HR_target[number_of_analyzed_models] <- HR_target
            results_df$scenario_functions[number_of_analyzed_models] <- as.character(scenario_function)
            results_df$n_patients[number_of_analyzed_models] <- n_patients
            results_df$max_time[number_of_analyzed_models] <- max_time
            results_df$doses[number_of_analyzed_models] <- paste(doses, collapse ='')
            results_df$binarization_dose[number_of_analyzed_models] <- binarization_dose
            results_df$n_knots[number_of_analyzed_models] <- n_knots
            results_df$constraint[number_of_analyzed_models] <- as.character(constraint)
            results_df$cutoff[number_of_analyzed_models] <- cutoff
            results_df$n_bootstraps[number_of_analyzed_models] <- n_bootstraps
            results_df$BIC[number_of_analyzed_models] <- BIC
            results_df$HR[number_of_analyzed_models] <- HR[1]
            results_df$lower_IC[number_of_analyzed_models] <- HR[2]
            results_df$higher_IC[number_of_analyzed_models] <- HR[3]

            if (save_models == TRUE){
                model_name = paste0("saved_model_",as.character(number_of_analyzed_models))
                results_df$path_model = model_name


                model_result_name = paste0(model_name,".rds")
                model_result_path = file.path(result_df_path,model_result_name)
                saveRDS(wce_model, model_result_path)
                
            }

 


            write.csv(results_df, file_result_path)
        }

}

print(HR_results)
print(HR_target_result)



print(results_df)





quit()







