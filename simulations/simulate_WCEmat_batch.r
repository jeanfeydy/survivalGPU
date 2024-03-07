library(jsonlite)

# imports 

source("src/data_simulation.r")
source("src/weight_functions.r")

options(scipen = 999)



# simulation_parameters <- fromJSON("Simulation_results/simulation_parameters.json")

# doses <- simulation_parameters$doses
# observation_time <- simulation_parameters$observation_time[1]
# normalization <- simulation_parameters$normalization[1]
# n_patients_list <- simulation_parameters$n_patients
# scenario_list <- simulation_parameters$weight_function_list

### Simualtion parameters 

doses <- c(1,1.5,2,2.5,3)
observation_time <- c(365)
n_patients_list <- c(100,200)#c(100,1000,10000)
scenario_list =  c("bi_linear_weight")
cutoff = 180
HR_target = 2.8


if(HR_target == 1){
    
    print("HR target = 1, only scenario is null_weight")
    scenario_list = c("null_weight")
}



# simulation_times_list <- list()

for (n_patients in n_patients_list){


     

    Xmat <- generate_Xmat(observation_time,n_patients,doses)
    write.csv(Xmat, paste0("Xmat/", n_patients,".csv  "))

    
     
    scenario_time_list <- list()


   
    for (scenario in scenario_list){


        scenario_function <- scenario_translator(scenario)

        batchsize <- 10000

        start_time <- Sys.time()

        print(n_patients)

        normalization_factor <- calculate_normalization_factor(scenario_function,HR_target,cutoff)

        print(paste0("normalization_factor: ",normalization_factor))

        wce_mat <- do.call("rbind", lapply(1:cutoff, wce_vector, scenario = scenario_function, Xmat = Xmat,normalization_factor = normalization_factor)) 

        if(HR_target == 1){
            is_null_weight <- TRUE
        }else{
            is_null_weight <- FALSE
        }

        if (n_patients > batchsize){

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
                print(current_n)
                max_n <- min(current_n + batchsize -1,n_patients)
                print(max_n)
                wce_mat_batch <- wce_mat[,current_n:max_n]
                Xmat_batch <- Xmat[,current_n:max_n]

                

                df_wce_batch <- get_dataset(Xmat = Xmat_batch, wce_mat = wce_mat_batch,is_null_weight)
                df_wce_batch <- df_wce_batch[order(df_wce_batch$patient),]

                df_wce_batch["patient"] <- df_wce_batch["patient"] + current_n -1
                df_wce <- rbind(df_wce,df_wce_batch)
                current_n <- current_n + batchsize

            }
            print(current_n)



        }
        else{
            
            df_wce <- get_dataset(Xmat = Xmat, wce_mat = wce_mat,is_null_weight)

            
            # print("#### scenario ####")

            # print(scenario_time)
            # print(scenario)
            # print("###list##")
            
            # print(scenario_time_list)
            # print("#########")

        }

        file_name = paste0(scenario,"_",as.character(n_patients),".csv")
        
        HR_folder_name = paste0("HR-",as.character(HR_target))


        folder_path <- file.path("WCEmat",HR_folder_name)

        if (!dir.exists(folder_path)){
        dir.create(folder_path)
        } 

        export_path <- file.path(folder_path,file_name)


        print(export_path)
        write.csv(df_wce, export_path, row.names=FALSE) 
        end_time <- Sys.time()

        simulation_times <- end_time - start_time
        scenario_time = as.numeric(simulation_times, units = "secs")
        scenario_time_list[[scenario]] <-  scenario_time

        }


    # print("######## n_patient")
    # # print(scenario_time_list)
    # # simulation_times_list[[as.character(n_patients)]] <- scenario_time_list
    # # print(simulation_times_list)


    # print("##############")


 }
# print("####final")
# # print(simulation_times_list)

# # json_file_name <-paste0(Xmat,"/simulation_times.json")
# # # print(simulation_times_list)
# # write(toJSON(simulation_times_list), file = json_file_name)



