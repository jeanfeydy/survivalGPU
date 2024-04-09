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
scenario = "exponential_scenario"


Xmat <- generate_Xmat(max_time,n_patients,doses)

py_wce_mat = py_generate_wce_mat(scenario, Xmat,max_time) 
print("OK")
print("NOW R : ")

scenario_function <- scenario_translator("exponential_weight")
print(scenario_function)
normalization_factor = calculate_normalization_factor(scenario_function, HR_target,cutoff)
print(normalization_factor)
R_wce_mat  <-  do.call("rbind", lapply(1:max_time, wce_vector, scenario = scenario_function, Xmat = Xmat,normalization_factor = normalization_factor)) 
print(R_wce_mat)

print("#######################")

print(dim(R_wce_mat))
print(dim(py_wce_mat))




df_wce <- simulate_dataset(max_time, n_patients, doses, scenario , cutoff, HR_target, Xmat)
print(head(df_wce))
# # simulate_dataset(max_time, n_patients, doses, scenario, cutoff, HR_target)
n_knots = 1
n_bootstraps = 1
batchsize = 0


# # result = py_matching_algo(wce_mat = R_wce_mat, max_time = max_time, n_patients = n_patients, HR_target = HR_target)

# # print(result)

# # quit()


# print("OK")

# is_null_weight = FALSE


# restults <- get_dataset(Xmat = Xmat, wce_mat = R_wce_mat,is_null_weight)

# df_wce = restults$df_wce
# matching_result = restults$matching_result

# df_wce = restults$df_wce

# wce_id_indexes = matching_result$patient_order
# fu = matching_result$df_event["FUP_Ti"]
# event = matching_result$df_event["event"]

# df_event =  matching_result$df_event

# print(df_event)
# print(wce_id_indexes)

# df_wce = df_wce[order(df_wce$patient),]
# print("df_wce")
# print(df_wce)



# test = get_dataset(Xmat, R_wce_mat, HR_target)

# df_wce_python = py_get_dataset(Xmat, R_wce_mat, HR_target,fu, event, wce_id_indexes)

# print("df_wce_python")
# print(head(df_wce_python))

# print("####################")
# print(head(df_wce_python))
# print(head(df_wce))

# print("Xmat")
# # print(Xmat)



# df_R = generate_dataset_batch(doses,observation_time, n_patients,scenario,cutoff,HR_target,batchsize)


# quit()

# wce_model_GPU_bootstraps <- wceGPU(df_wce_python, n_knots, cutoff, constrained = "R",
#                id = "patient", event = "event", start = "start",
#                stop = "stop", expos = "dose",nbootstraps = 1,batchsize = 0, verbosity=0)
               

wce_model_GPU_bootstraps_R <- wceGPU(df_wce, n_knots, cutoff, constrained = "R",
               id = "patient", event = "event", start = "start",
               stop = "stop", expos = "dose",nbootstraps = 1,batchsize = 0, verbosity=0)





exposed   <- rep(1, cutoff)
non_exposed   <- rep(0, cutoff)

# HR_result_GPU_bootstraps = HR(wce_model_GPU_bootstraps,vecnum = exposed, vecdenom= non_exposed)  
HR_result_GPU_bootstraps_R = HR(wce_model_GPU_bootstraps_R,vecnum = exposed, vecdenom= non_exposed)  

# print(HR_result_GPU_bootstraps)
print(HR_result_GPU_bootstraps_R)

print(head(df_wce))




# wce_model_GPU_bootstraps <- wceGPU(df_R, n_knots, cutoff, constrained = "R",
#                id = "patient", event = "event", start = "start",
#                stop = "stop", expos = "dose",nbootstraps = n_bootstraps,batchsize = batchsize, verbosity=0)


# exposed   <- rep(1, cutoff)
# non_exposed   <- rep(0, cutoff)

# HR_result_GPU_bootstraps = HR(wce_model_GPU_bootstraps,vecnum = exposed, vecdenom= non_exposed)


# print(HR_result_GPU_bootstraps)