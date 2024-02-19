library(jsonlite)

# imports 

source("src/data_simulation.r")
source("src/weight_functions.r")


# simulation_parameters <- fromJSON("Simulation_results/simulation_parameters.json")

# doses <- simulation_parameters$doses
# observation_time <- simulation_parameters$observation_time[1]
# normalization <- simulation_parameters$normalization[1]
# n_patients_list <- simulation_parameters$n_patients
# scenario_list <- simulation_parameters$weight_function_list

### Simualtion parameters 

doses <- c(1,1.5,2,2.5,3)
observation_time <- c(365)
n_patients_list <- c(100,500,1000,5000,10000,50000,100000)
normalization = c(1)
scenario_list = c("exponential_weight")


scenario_translator <- function(scenario_name){

    scenario_list <- list(
    list(name ="exponential_weight", weights = exponential_weight),
    list(name ="bi_linear_weight", weights = bi_linear_weight),
    list(name ="early_peak_weight", weights = early_peak_weight),
    list(name ="inverted_u_weight", weights = inverted_u_weight),
    list(name ="constant_weight", weights = constant_weight),
    list(name ="late_effect_weight", weights = late_effect_weight)
    
    )

    for(scenario in scenario_list) {
        if(scenario$name == scenario_name) {
            return(scenario$weights)
        }
    } 

}

simulation_times_list <- list()

for (n_patients in n_patients_list){

    path_experiement_result <- paste0("Simulation_results/",simulation_parameters$experiment_name)
     

    Xmat <- generate_Xmat(observation_time,n_patients,doses)
    write.csv(Xmat, paste0("Xmat/", n_patients,".csv  "))
    
    
    scenario_time_list <- list()
    for (scenario in scenario_list){

        batchsize <- 10000

        start_time <- Sys.time()

        print(n_patients)

        wce_mat <- do.call("rbind", lapply(1:dim(Xmat)[1], wce_vector, scenario = scenario, Xmat = Xmat,normalization = normalization))

        


        if (n_patients > batchsize){

            current_n <- 1

            print(current_n)
            max_n <- current_n + batchsize -1
            wce_mat_batch <- wce_mat[,current_n:max_n]
            Xmat_batch <- Xmat[,current_n:max_n]

            current_n <- current_n + batchsize

            df_wce_batch <- get_dataset(Xmat = Xmat_batch, wce_mat = wce_mat_batch)
            df_wce_batch <- df_wce_batch[order(df_wce_batch$patient),]

            df_wce <- df_wce_batch

            while(current_n < n_patients){
                print(current_n)
                max_n <- min(current_n + batchsize -1,n_patients)
                print(max_n)
                wce_mat_batch <- wce_mat[,current_n:max_n]
                Xmat_batch <- Xmat[,current_n:max_n]

                

                df_wce_batch <- get_dataset(Xmat = Xmat_batch, wce_mat = wce_mat_batch)
                df_wce_batch <- df_wce_batch[order(df_wce_batch$patient),]

                df_wce_batch["patient"] <- df_wce_batch["patient"] + current_n -1
                df_wce <- rbind(df_wce,df_wce_batch)
                current_n <- current_n + batchsize

            }
            print(current_n)



        }
        else{
            
            df_wce <- get_dataset(Xmat = Xmat, wce_mat = wce_mat)

            
            # print("#### scenario ####")

            # print(scenario_time)
            # print(scenario)
            # print("###list##")
            
            # print(scenario_time_list)
            # print("#########")

        }

        export_path <- paste0("WCEmat/", scenario,"_",n_patients,".csv")
        print(export_path)
        write.csv(df_wce, export_path, row.names=FALSE) 
        end_time <- Sys.time()

        simulation_times <- end_time - start_time
        scenario_time = as.numeric(simulation_times, units = "secs")
        scenario_time_list[[scenario]] <-  scenario_time

        }
    print("######## n_patient")
    # print(scenario_time_list)
    simulation_times_list[[as.character(n_patients)]] <- scenario_time_list
    print(simulation_times_list)


    print("##############")


 }
print("####final")
print(simulation_times_list)

json_file_name <-paste0(path_experiement_result,"/simulation_times.json")
# print(simulation_times_list)
write(toJSON(simulation_times_list), file = json_file_name)



