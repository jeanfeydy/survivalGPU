library(jsonlite)

# imports 

source("weight_functions.R")
source("simulation_parameters.R")
source("functions_data_simulation.R")

options(scipen = 999)


########  Defintion of the max batch size for generation of dataset greater than 10000 patients, necessary for computation time 
BATCHSIZE = 10000 
########

####### SCRIPT

simulation_parameters <- generate_simulation_parameters()
print(simulation_parameters)

number_of_simulations <- simulation_parameters$number_of_simulations

doses_list <- simulation_parameters$doses_list
observation_time_list <- simulation_parameters$observation_time_list
n_patients_list <- simulation_parameters$n_patients_list
scenario_list <- simulation_parameters$scenario_list
cutoff_list <- simulation_parameters$cutoff_list
HR_target_list <- simulation_parameters$HR_target_list


combinaisons_parameters <- expand.grid( 
                                        # doses = doses_list,
                                        observation_time = observation_time_list,
                                        n_patients = n_patients_list,
                                        scenario = scenario_list,
                                        cutoff = cutoff_list,
                                        HR_target = HR_target_list)
print(combinaisons_parameters)


for(i in 1:nrow(combinaisons_parameters)){

    # doses <- combinaisons_parameters$doses[i]
    doses <- doses_list
    observation_time <- combinaisons_parameters$observation_time[i]
    n_patients <- combinaisons_parameters$n_patients[i]
    scenario <- combinaisons_parameters$scenario[i]
    cutoff <- combinaisons_parameters$cutoff[i]
    HR_target <- combinaisons_parameters$HR_target[i]


    for (iteration_simulation in 1:number_of_simulations){

        # print("generation dataset")

        df_wce <- generate_dataset_batch(
            doses = doses,
            observation_time = observation_time,
            n_patients = n_patients,
            scenario = scenario,
            cutoff = cutoff,
            HR_target = HR_target,
            batchsize = BATCHSIZE     
        )

        print(head(df_wce))

        folder_path = file.path("../simulated_datasets", scenario,paste0("HR-",as.character(HR_target)),paste0("n_",as.character(n_patients)))

            dir.create(folder_path)

            if (!dir.exists(folder_path)){
                dir.create(folder_path, recursive = TRUE)
            } 

            

            number_existing_datasets <- length(list.files(folder_path))

            file_name <- paste0("dataset_",as.character(number_existing_datasets),".csv")

            export_path <- file.path(folder_path,file_name)

            print(export_path)
            write.csv(df_wce, export_path, row.names=FALSE)  
    }
}


