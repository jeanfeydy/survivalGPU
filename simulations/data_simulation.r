# imports 

source("~/dev/survivalGPU/simulations/src/data_simulation.r")
source("~/dev/survivalGPU/simulations/src/weight_functions.r")

library(WCE)

# Simualtion charateristics

file_name <- "test_data"

doses <- c(1,1.5,2,2.5,3)
observation_time <- 365
n_patients <- 500
scenario <- exponential_weight
normalization <- 1

# Simulation of Xmat

Xmat<- generate_Xmat(observation_time = 365,n_patients = 500,doses = doses)


function_name <- deparse(substitute(exponential_weight))


scenario_list <- list(
    list(name ="exponential_weight", weights = exponential_weight),
    list(name ="bi_linear_weight", weights = bi_linear_weight)
)






for (scenario in scenario_list){


    print(paste("#### Generating scenario : ",scenario$name))

    wce_mat <- do.call("rbind", lapply(1:dim(Xmat)[1], wce_vector, scenario = scenario$weights, Xmat = Xmat,normalization = normalization))
    df_wce <- get_dataset(Xmat = Xmat, wce_mat = wce_mat)
    export_path <- paste0("~/dev/survivalGPU/simulations/WCEmat_data/", scenario$name)
    export_path <- paste0(export_path, ".csv")
    write.csv(df_wce, export_path, row.names=FALSE)
 }


