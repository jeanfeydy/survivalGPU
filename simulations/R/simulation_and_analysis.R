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




simulate_dataset <- function(max_time, n_patients, doses, scenario, HR_target, constraint){

    survivalgpu = use_survivalGPU()

    gpu_simulate_dataset = survivalgpu$simulate_dataset

    dataset = gpu_simulate_dataset(max_time, n_patients, doses, scenario,HR_target)

    return(dataset)

}

modelize_dataset <- function(max_time, n_patients, cutoff, n_bootstraps, n_knots,dataset, constraint){



    if (n_bootstraps == 1){
        batchsize = 0
    }else if (n_patients < 100){
        batchsize = n_patients
    }
    else if (n_patients <= 10000){
        batchsize = 100

    }else{
        batchsize = 10
    }

    wce_model_list = list()


    wce_model <- wceGPU(dataset, n_knots, cutoff, constrained = constraint,
                        id = "patient", event = "event", start = "start",
                        stop = "stop", expos = "dose",nbootstraps = n_bootstraps,batchsize = , verbosity=0)



   return(wce_model)

}



analyze_model <- function(wce_model, n_patients, cutoff){
    # BIC = mean(wce_model_GPU_bootstraps$info.criterion)

    exposed   <- rep(1, cutoff)
    non_exposed   <- rep(0, cutoff)
    HR = HR(wce_model,vecnum = exposed, vecdenom= non_exposed)


    return(HR)

}

# analysis_for_all_nknots <- function(dataset, cutoff, constraint,n_bootstraps, n_knots_list){


#     print(n_knots_list)
#     analysis_results_list <- list()

#     lowest_nknot <- FALSE
#     lowest_BIC <- FALSE


#     for(n_knots in n_knots_list){

        

#         wce_model <- modelize_dataset(max_time, n_patients, cutoff, n_bootstraps, n_knots[[1]],dataset)
#         BIC <- mean(wce_model$info.criterion)
#         HR <- analyze_model(wce_model, n_patients, cutoff)

#         print(HR) 

#         if(lowest_BIC == FALSE){
#             lowest_BIC <- BIC
#             lowest_nknot <- HR
#         }



#         analysis_results_list[[n_knots]] = list(
#             "HR" = HR[1],
#             "lower_CI" = HR[2],
#             "higher_CI" = HR[3],
#             "BIC" = BIC
#         )

#         print(analysis_results_list)
#     }



#     return(analysis_results_list)
# }

# get_better_bic_nknots <- function(analysis_results_list,n_knots_list){

#     first_knots  <- min(n_knots_list)

#     print(analysis_results_list)

#     print("OK1")

#     lowest_n_knots <- first_knots
#     lowest_BIC <- analysis_results_list[[first_knots]]$BIC

#     for (n_knots in n_knots_list){
#         BIC <- analysis_results_list[[n_knots]]$BIC
#         if (BIC < lowest_BIC){
#             lowest_BIC <- BIC
#             lowest_n_knots = n_knots
#             HR <- analysis_results_list[[n_knots]]$HR
#         }

#     }

#     results = list(
#         "n_knots" = lowest_n_knots,
#         "BIC" = lowest_BIC,
#         "HR" = HR)

#     quit()

#     return(results
#     )
# }


