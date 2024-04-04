library(reticulate)
library(devtools)

# py_config()
# py_exe()
# devtools::load_all("../../../survivalGPU")


# path <- system.file("python", package = <package>)

# test <- import_from_path(module = "simulation",path="../python/taichi_simulations/simualtion_script.py")

devtools::load_all("../../../survivalGPU/R")

survivalgpu = use_survivalGPU()
simulate_dataset = survivalgpu$simulate_dataset


n_patients = 10000
print(class(n_patients))

cutoff = 180

df_wce <- simulate_dataset(max_time = 365, n_patients = n_patients, doses = c(1,2,3), scenario = "exponential_scenario", cutoff = cutoff)

n_knots = 1
n_bootstraps = 100
batchsize = 100



wce_model_GPU_bootstraps <- wceGPU(df_wce, n_knots, cutoff, constrained = "R",
               id = "patient", event = "event", start = "start",
               stop = "stop", expos = "dose",nbootstraps = n_bootstraps,batchsize = batchsize, verbosity=0)


exposed   <- rep(1, cutoff)
non_exposed   <- rep(0, cutoff)

HR_result_GPU_bootstraps = HR(wce_model_GPU_bootstraps,vecnum = exposed, vecdenom= non_exposed)


print(HR_result_GPU_bootstraps)