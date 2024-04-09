library(reticulate)
library(devtools)
library(WCE)
source("weight_functions.R")
source("functions_data_simulation.R")
# py_config()
# py_exe()
# devtools::load_all("../../../survivalGPU")


# path <- system.file("python", package = <package>)

# test <- import_from_path(module = "simulation",path="../python/taichi_simulations/simualtion_script.py")

devtools::load_all("../../../survivalGPU/R")

survivalgpu = use_survivalGPU()
simulate_dataset = survivalgpu$simulate_dataset
py_generate_wce_mat = survivalgpu$generate_wce_mat
py_get_dataset = survivalgpu$get_dataset
py_matching_algo = survivalgpu$matching_algo

HR_target = 2.8
max_time = 365

doses = c(1,1.5,2,2.5,3)


n_patients = 1000
print(class(n_patients))

cutoff = 180

print(max_time)
scenario = "exponential_weight"


Xmat <- generate_Xmat(max_time,n_patients,doses)


n_knots = 1
n_bootstraps = 1
batchsize = 0


observation_time = max_time


########## Rprocess


Xmat <- generate_Xmat(observation_time,n_patients,doses)
scenario_function <- scenario_translator(scenario)
normalization_factor <- calculate_normalization_factor(scenario_function,HR_target,observation_time)

wce_mat <- do.call("rbind", lapply(1:observation_time, wce_vector, scenario = scenario_function, Xmat = Xmat,normalization_factor = normalization_factor)) 
print("R WCE MAT")
print(head(wce_mat))
events_generation <- event_censor_generation(observation_time,n_patients)
matching_result <- matching_algo(wce_mat,events_generation)
df_R <- get_dataset(Xmat = Xmat, wce_mat = wce_mat_batch,matching_result = matching_result)

######### Python process 


######## Validation WCE_mat

# Xmat is OK

df_python = simulate_dataset(max_time, n_patients, doses, "exponential_scenario", cutoff, HR_target, Xmat)



wce_model_GPU_bootstraps <- wceGPU(df_python, n_knots, cutoff, constrained = "R",
               id = "patient", event = "event", start = "start",
               stop = "stop", expos = "dose",nbootstraps = 1,batchsize = 0, verbosity=0)
               

wce_model_GPU_bootstraps_R <- wceGPU(df_R, n_knots, cutoff, constrained = "R",
               id = "patient", event = "event", start = "start",
               stop = "stop", expos = "dose",nbootstraps = 1,batchsize = 0, verbosity=0)





exposed   <- rep(1, cutoff)
non_exposed   <- rep(0, cutoff)

HR_result_GPU_bootstraps = HR(wce_model_GPU_bootstraps,vecnum = exposed, vecdenom= non_exposed)  
HR_result_GPU_bootstraps_R = HR(wce_model_GPU_bootstraps_R,vecnum = exposed, vecdenom= non_exposed)  

print("python")
print(HR_result_GPU_bootstraps)
print("R")
print(HR_result_GPU_bootstraps_R)





# wce_model_GPU_bootstraps <- wceGPU(df_R, n_knots, cutoff, constrained = "R",
#                id = "patient", event = "event", start = "start",
#                stop = "stop", expos = "dose",nbootstraps = n_bootstraps,batchsize = batchsize, verbosity=0)


# exposed   <- rep(1, cutoff)
# non_exposed   <- rep(0, cutoff)

# HR_result_GPU_bootstraps = HR(wce_model_GPU_bootstraps,vecnum = exposed, vecdenom= non_exposed)


# print(HR_result_GPU_bootstraps)