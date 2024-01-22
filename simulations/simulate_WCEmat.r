library(jsonlite)

# imports 

source("src/data_simulation.r")
source("src/weight_functions.r")

# library(WCE)

# Simualtion charateristics

file_name <- "test_data"

args <- commandArgs(trailingOnly = TRUE)
simulation_variables <- fromJSON(args[1])

doses <- simulation_variables$doses
observation_time <- simulation_variables$observation_time[1]
normalization <- simulation_variables$normalization[1]






scenario_list <- list(
    list(name ="exponential_weight", weights = exponential_weight)#,
    # list(name ="bi_linear_weight", weights = bi_linear_weight)#,
    # list(name ="early_peak_weight", weights = early_peak_weight),
    # list(name ="inverted_u_weight", weights = inverted_u_weight),
    # list(name ="constant_weight", weights = constant_weight),
    # list(name ="late_effect_weight", weights = late_effect_weight)
    
)

# n_patients_list = c(20, 30, 40, 70, 100, 500, 1000, 2500, 5000)
n_patients_list = c(30,50)


for (n_patients in n_patients_list){
    

    Xmat <- generate_Xmat(observation_time,n_patients,doses)
    write.csv(Xmat, paste0("Xmat_data/", n_patients,".csv  "))

    for (scenario in scenario_list){
        print(paste("#### Generating scenario : ",scenario$name))
        print(paste("##n_patients = ",n_patients))
        wce_mat <- do.call("rbind", lapply(1:dim(Xmat)[1], wce_vector, scenario = scenario$weights, Xmat = Xmat,normalization = normalization))
        df_wce <- get_dataset(Xmat = Xmat, wce_mat = wce_mat)
        export_path <- paste0("WCEmat_data/", scenario$name,"_",n_patients,".csv")
        print(export_path)
        write.csv(df_wce, export_path, row.names=FALSE)
        }
 }


