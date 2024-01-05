# imports 

source("src/data_simulation.r")
library(WCE)

# Simualtion charateristics

file_name <- "test_data"

doses <- c(1,1.5,2,2.5,3)
observation_time <- 365
n_patients <- 500
scenario <- exponential_weight

# Simulation of Xmat

Xmat<- generate_Xmat(observation_time = 365,n_patients = 500,doses = doses)


# export matrice

export_path <- paste("Xmat_data/", file_name)
write.csv(Xmat, export_path, row.names=FALSE)
