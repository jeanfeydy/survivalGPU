# import

library(dplyr)
library(WCE)
library(purrr)

source("src/data_simulation.r")
source("src/weight_functions.r")
source("../R/R/wceGPU.R")


# variable definition

Xmat_file_name <- "exponential_weight.csv"
scenario <- exponential_weight
cutoff <- 180
normalization <- normalize_function(scenario = scenario, 1, cutoff/365)

# script


dataset <- read.table(paste0("WCEmat_data/",Xmat_file_name), sep =",",dec = ".",)
sorted_dataset <- dataset[order(dataset[, 1]), ]

print(head(sorted_dataset))

