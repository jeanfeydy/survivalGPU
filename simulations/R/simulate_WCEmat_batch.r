library(jsonlite)

# imports 

source("../src/data_simulation.r")
source("../src/weight_functions.r")
source("simulation_parameters.R")

options(scipen = 999)


########  Defintion of the max batch size for generation of dataset greater than 10000 patients, necessary for computation time 
BATCHSIZE = 10000 
########

###########"GENERATE dataset function"

# A function that generate a dataset with the batch method (if necessary)
generate_dataset_batch <- function(doses,observation_time, n_patients,scenario,cutoff,HR_target,batchsize){ 
    
    # Generation of the dose amtrix for the number of desired patients
    Xmat <- generate_Xmat(observation_time,n_patients,doses)

    scenario_time_list <- list()
    
    # Translating the wieght function form the scenario as a string
    scenario_function <- scenario_translator(scenario)


    # TODO Will need to change the nomraliszation factor calcul in order to take into account the 
    normalization_factor <- calculate_normalization_factor(scenario_function,HR_target,cutoff)



    # Generation of the WCEmat that calculate the WCE weight of the patient at every time        
    wce_mat <- do.call("rbind", lapply(1:cutoff, wce_vector, scenario = scenario_function, Xmat = Xmat,normalization_factor = normalization_factor)) 

    # Look if HR is 1
    if(HR_target == 1){
        is_null_weight <- TRUE
    }else{
        is_null_weight <- FALSE
    }

    # simple generation of the data if n_patients <= BATCHSIZE
    if (n_patients <= batchsize){
        df_wce <- get_dataset(Xmat = Xmat, wce_mat = wce_mat,is_null_weight)
    }
    # Batch generation of the data if n_patients <= BATCHSIZE
    else{
        current_n <- 1
        print(current_n)
        max_n <- current_n + batchsize -1
        wce_mat_batch <- wce_mat[,current_n:max_n]
        Xmat_batch <- Xmat[,current_n:max_n]
        current_n <- current_n + batchsize
        df_wce_batch <- get_dataset(Xmat = Xmat_batch, wce_mat = wce_mat_batch,is_null_weight)
        df_wce_batch <- df_wce_batch[order(df_wce_batch$patient),]
        df_wce <- df_wce_batch

        while(current_n < n_patients){

               max_n <- min(current_n + batchsize -1,n_patients)
               wce_mat_batch <- wce_mat[,current_n:max_n]
               Xmat_batch <- Xmat[,current_n:max_n]
               df_wce_batch <- get_dataset(Xmat = Xmat_batch, wce_mat = wce_mat_batch,is_null_weight)
               df_wce_batch <- df_wce_batch[order(df_wce_batch$patient),]
               df_wce_batch["patient"] <- df_wce_batch["patient"] + current_n -1
               df_wce <- rbind(df_wce,df_wce_batch)
               current_n <- current_n + batchsize
           }
    }    

}



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


