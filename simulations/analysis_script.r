# import

library(dplyr)
library(WCE)
library(purrr)

source("src/data_simulation.r")
source("src/weight_functions.r")
source("../R/R/wceGPU.R")


# variable definition

Xmat_file_name <- "test_data"
scenario <- exponential_weight
cutoff <- 180
normalization <- normalize_function(scenario = scenario, 1, cutoff/365)

# script


Xmat <- read.table(paste("Xmat_data/",Xmat_file_name), sep =",",dec = ".",)



