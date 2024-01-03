library(dplyr)
library(purrr)
library(devtools)
library(WCE)

load_all("/home/dev/survivalGPU/R")
source("simulations/src/data_simulation_GPU.r")
source("simulations/src/weight_functions.r")



doses = c(1,1.5,2,2.5,3)


n_patients <- 500
observation_time <- 365

Xmat<- generate_Xmat(observation_time = observation_time,n_patients = n_patients,doses = doses)

scenario <- exponential_weight
cutoff <- 180
normalization <- normalize_function(scenario = scenario, 1, cutoff/365)
n_bootstraps <- 10
batchsize <- 50

df_wce = simulate_dataset(exponential_weight,Xmat,normalization)

print(head(df_wce))

print("#### Simulation package WCE ####")

wce_right_constrained <- WCE(df_wce, "cox", 1, cutoff, constrained = "right",
                             id = "patient", event = "event", start = "start",
                             stop = "stop", expos = "dose")

print("OK")

print("#### Simulation package wceGPU ####")


wce_right_constrained <- wceGPU(df_wce, 1, cutoff, constrained = "right",
                             id = "patient", event = "event", start = "start",
                             stop = "stop", expos = "dose")

