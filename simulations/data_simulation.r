# imports 

source("~/dev/survivalGPU/simulations/src/data_simulation.r")
source("~/dev/survivalGPU/simulations/src/weight_functions.r")

library(WCE)

# Simualtion charateristics

file_name <- "test_data"

doses <- c(1,1.5,2,2.5,3)
observation_time <- 365
n_patients <- 10000
scenario <- exponential_weight
normalization <- 1

# Simulation of Xmat

Xmat<- generate_Xmat(observation_time,n_patients,doses)

scenario_list <- list(
    list(name ="exponential_weight", weights = exponential_weight),
    list(name ="bi_linear_weight", weights = bi_linear_weight),
    list(name ="early_peak_weight", weights = early_peak_weight),
    list(name ="inverted_u_weight", weights = inverted_u_weight),
    list(name ="constant_weight", weights = constant_weight),
    list(name ="late_effect_weight", weights = late_effect_weight)
    
)






for (scenario in scenario_list){


    print(paste("#### Generating scenario : ",scenario$name))

    wce_mat <- do.call("rbind", lapply(1:dim(Xmat)[1], wce_vector, scenario = scenario$weights, Xmat = Xmat,normalization = normalization))
    df_wce <- get_dataset(Xmat = Xmat, wce_mat = wce_mat)
    export_path <- paste0("~/dev/survivalGPU/simulations/WCEmat_data/", scenario$name)
    export_path <- paste0(export_path, ".csv")
    write.csv(df_wce, export_path, row.names=FALSE)
 }


