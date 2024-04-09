library(reticulate)
library(devtools)
library(WCE)
source("weight_functions.R")
source("functions_data_simulation.R")


devtools::load_all("../../../survivalGPU/R")

survivalgpu = use_survivalGPU()
py_simulate_dataset = survivalgpu$simulate_dataset
py_generate_wce_mat = survivalgpu$generate_wce_mat
py_get_dataset = survivalgpu$get_dataset
py_matching_algo = survivalgpu$matching_algo

HR_target = 2.8
max_time = 365

doses = c(1,1.5,2,2.5,3)


n_patients = 50
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
events_generation <- event_censor_generation(observation_time,n_patients)
matching_result <- matching_algo(wce_mat,events_generation)
df_R <- get_dataset(Xmat = Xmat, wce_mat = wce_mat_batch,matching_result = matching_result)

######### Python process 


######## Validation WCE_mat

# Xmat is OK

wce_mat_python = py_generate_wce_mat(scenario_name= "exponential_scenario", Xmat = Xmat, max_time= max_time)
# print(wce_mat)
# print(wce_mat_python)


fu_list = c()
fu_df = matching_result$df_event["FUP_Ti"]
for (fu in fu_df){

    fu_list = append(fu_list,fu)
}



event_list = c()
event_df = matching_result$df_event["event"]
for (event in event_df){

    event_list = append(event_list,event)
}


df_python_pure = py_simulate_dataset(max_time, n_patients, doses, "exponential_scenario", cutoff, HR_target, Xmat,wce_mat,event_list,fu_list)
wce_id_indexes = py_matching_algo(wce_mat, max_time,n_patients, HR_target,event_list, fu_list)
df_python = py_get_dataset(Xmat, max_time,n_patients, HR_target, fu_list,event_list,wce_id_indexes)


wce_model_python_pure <- wceGPU(df_python_pure, n_knots, cutoff, constrained = "R",
               id = "patient", event = "event", start = "start",
               stop = "stop", expos = "dose",nbootstraps = 1,batchsize = 0, verbosity=0)

wce_model_python <- wceGPU(df_python, n_knots, cutoff, constrained = "R",
               id = "patient", event = "event", start = "start",
               stop = "stop", expos = "dose",nbootstraps = 1,batchsize = 0, verbosity=0)

wce_model_R <- wceGPU(df_R, n_knots, cutoff, constrained = "R",
               id = "patient", event = "event", start = "start",
               stop = "stop", expos = "dose",nbootstraps = 1,batchsize = 0, verbosity=0)


exposed   <- rep(1, cutoff)
non_exposed   <- rep(0, cutoff)

HR_result_python_pure = HR(wce_model_python_pure,vecnum = exposed, vecdenom= non_exposed)  
HR_result_python = HR(wce_model_python,vecnum = exposed, vecdenom= non_exposed)  
HR_result_R = HR(wce_model_R,vecnum = exposed, vecdenom= non_exposed)  


print("python_pure")
print(HR_result_python_pure)
print("python")
print(HR_result_python)
print("R")
print(HR_result_R)

