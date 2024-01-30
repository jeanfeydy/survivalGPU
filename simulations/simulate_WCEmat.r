library(jsonlite)

# imports 

source("src/data_simulation.r")
source("src/weight_functions.r")


simulation_parameters <- fromJSON("Simulation_results/simulation_parameters.json")

doses <- simulation_parameters$doses
observation_time <- simulation_parameters$observation_time[1]
normalization <- simulation_parameters$normalization[1]
n_patients_list <- simulation_parameters$n_patients
scenario_list <- simulation_parameters$weight_function_list



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

for (n_patients in n_patients_list){

    path_experiement_result <- paste0("Simulation_results/",simulation_parameters$experiment_name)
    

    Xmat <- generate_Xmat(observation_time,n_patients,doses)
    write.csv(Xmat, paste0(path_experiement_result,"/Xmat/", n_patients,".csv  "))
    simulation_time_list <- list()
    
    scenario_time_list <- list()
    for (scenario in scenario_list){
        
        start_time <- Sys.time()
        wce_mat <- do.call("rbind", lapply(1:dim(Xmat)[1], wce_vector, scenario = scenario_translator(scenario), Xmat = Xmat,normalization = normalization))
        df_wce <- get_dataset(Xmat = Xmat, wce_mat = wce_mat)
        export_path <- paste0(path_experiement_result,"/WCEmat/", scenario,"_",n_patients,".csv")
        print(export_path)
        write.csv(df_wce, export_path, row.names=FALSE) 
        end_time <- Sys.time()
        # print("#### scenario ####")
        simulation_time <- end_time - start_time
        scenario_time = as.numeric(simulation_time, units = "secs")
        # print(scenario_time)
        # print(scenario)
        # print("###list##")
        # scenario_time_list[[scenario]] <-  scenario_time
        # print(scenario_time_list)
        # print("#########")
        }
    print("######## n_patient")
    # print(scenario_time_list)
    simulation_time_list[[as.character(n_patients)]] <- scenario_time_list
    print(simulation_time_list)


    print("##############")


 }
print("####final")
print(simulation_time_list)

json_file_name <-paste0(path_experiement_result,"/simulation_time.json")
# print(simulation_time_list)
write(toJSON(simulation_time_list), file = json_file_name)



