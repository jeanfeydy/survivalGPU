library(WCE)
library(boot)
library(jsonlite)
library(devtools)
library(dplyr)

# options(scipen = 999)

devtools::load_all("../../../survivalGPU/R")

library(survivalGPU)


use_model <- function(
    dataset,
    cutoff,
    nknots,
    constraint,
    mode)
{


    if(mode == "WCE"){

        time_start = Sys.time()
        model = WCE::WCE(
            data = dataset,
            analysis = "Cox",
            nknots=c(nknots),
            cutoff=cutoff,
            id="patients",
            event = "events",
            start = "start",
            stop = "stop",
            expos = "dose",
            constrained = "Right"
        )
        time_stop = Sys.time()


        

    } else if(mode == "GPU"){
        time_start = Sys.time()
        model = wceGPU(
            data = dataset,
            nknots = nknots, 
            cutoff = cutoff, 
            id = "patients",
            event = "events",
            start = "start",
            stop = "stop",
            expos = "dose",
            constrained = "Right",
            verbosity = 0
        )
        time_stop = Sys.time()
    }
        else{
            print("Invalid mode")
        }

    time = time_stop - time_start


    return(list(model = model, time = time))
}


analyze_model_WCE <- function(model,cutoff){

    HR = HR.WCE(model, rep(1, cutoff), rep(0, cutoff))


    WCEmat = model$WCEmat
    BIC = model$info.criterion







    # beta <- model$coefficients
    # HR <- model$HR
    # BIC <- model$BIC
    return(list(WCEmat = WCEmat,
                HR = HR[1], 
                BIC = BIC[1]))
}


analyze_model_GPU <- function(model,cutoff){

    HR = HR(model, rep(1, cutoff), rep(0, cutoff))[[1]]

    BIC = model$info.criterion

    WCEmat = model$WCEmat[1, ]
    

    return(list(WCEmat = WCEmat,
                HR = HR, 
                BIC = BIC))

}


best_model <- function(model_list){

    best_model = model_list[[1]]

    
    for (model in model_list) {
        if (model$BIC < best_model$BIC) {
            best_model = model
        }
    }

    return(best_model)
}


analyze_model <- function(model, cutoff, mode){

    if(mode == "WCE"){
        return(analyze_model_WCE(model, cutoff))
    } else if(mode == "GPU"){
        return(analyze_model_GPU(model, cutoff))
    } else{
        print("Invalid mode")
    }
}


analyze_3_knots <- function(
    dataset,
    cutoff,
    constraint,
    mode){

    model_1 = use_model(dataset, cutoff, 1, constraint, mode)
    model_2 = use_model(dataset, cutoff, 2, constraint, mode)
    model_3 = use_model(dataset, cutoff, 3, constraint, mode)


    result_1 = analyze_model(model_1$model, cutoff, mode)
    result_2 = analyze_model(model_2$model, cutoff, mode)
    result_3 = analyze_model(model_3$model, cutoff, mode )

    model_list = list(
        result_1,
        result_2,
        result_3
    )



    total_time = as.numeric(model_1$time + model_2$time + model_3$time, units = "secs")

    best_model_result = best_model(model_list)

    return(list(best_model_result = best_model_result,
                total_time = total_time))
    }





result_iteration <- function(
    n_patients,
    max_time,
    scenario_name,
    HR_target,
    cutoff,
    constraint,
    mode
){

    print("inside")

    dataset = simulation_paper(
        n_patients = n_patients, 
        max_time = max_time, 
        scenario_name = scenario_name,
        HR_target = HR_target
    )

    
    result_cpu = analyze_3_knots(
        dataset,
        cutoff,
        constraint,
        mode
    )

    result_gpu = analyze_3_knots(
        dataset,
        cutoff,
        constraint,
        mode
    )

    return(list(result_cpu = result_cpu,
                result_gpu = result_gpu))


}


result_to_row <- function(
    iteration,
    library,
    scenario_name,
    HR_target,
    result,
    cutoff
){

    final_result = list(
        iteration = iteration,
        library = library,
        scenario_name = scenario_name,
        HR_target = HR_target,
        computation_time = result$total_time,
        simulated_HR = result$best_model_result$HR,
        beta = log(result$best_model_result$HR),
        biais_beta = (log(result$best_model_result$HR) - log(HR_target))/log(HR_target) * 100,
        BIC = result$best_model_result$BIC
    )


    

    return(final_result)
}


launch_experiment <- function(
    n_experiment,
    n_patients,
    max_time,
    scenario_name,
    HR_target,
    cutoff,
    constraint,
    mode
){


    iteration_list = c()
    library_list = c()
    scenario_name_list = c()
    HR_target_list = c()
    computation_time_list = c()
    simulated_HR_list = c()
    beta_list = c()
    biais_beta_list = c()
    BIC_list = c()


    # matrix to store WCE values of both cpu and gpu

    WCE_matrix = matrix(0, nrow = n_experiment*2, ncol = cutoff)





    for(i_iteration in 1:n_experiment){


        dataset = simulation_paper(
            n_patients = n_patients, 
            max_time = max_time, 
            scenario_name = scenario_name,
            HR_target = HR_target
        )

        result = result_iteration(
            n_patients = n_patients,
            max_time = max_time,
            scenario_name = scenario_name,
            HR_target = HR_target,
            cutoff = cutoff,
            constraint = constraint,
            mode = mode
        )


        result_cpu <- result$result_cpu
        result_gpu <- result$result_gpu

        row_cpu = result_to_row(
            iteration = i_iteration,
            library = "WCE",
            scenario_name = scenario_name,
            HR_target = 4,
            result = result_cpu,
            cutoff = 180
            )

        iteration_list = c(iteration_list, i_iteration)
        library_list = c(library_list, "WCE")
        scenario_name_list = c(scenario_name_list, scenario_name)
        HR_target_list = c(HR_target_list, 4)
        computation_time_list = c(computation_time_list, row_cpu$computation_time)
        simulated_HR_list = c(simulated_HR_list, row_cpu$simulated_HR)
        beta_list = c(beta_list, row_cpu$beta)
        biais_beta_list = c(biais_beta_list, row_cpu$biais_beta)
        BIC_list = c(BIC_list, row_cpu$BIC)


        cpu_WCE = result_cpu$best_model_result$WCEmat


        WCE_matrix[(i_iteration-1 )*2 +1, 1:cutoff] = cpu_WCE



        

        row_gpu = result_to_row(
            iteration = i_iteration,
            library = "GPU",
            scenario_name = "exponential_scenario",
            HR_target = 4,
            result = result_gpu,
            cutoff = 180
            )

        

        iteration_list = c(iteration_list, i_iteration)
        library_list = c(library_list, "survivalGPU")
        scenario_name_list = c(scenario_name_list, scenario_name)
        HR_target_list = c(HR_target_list, 4)
        computation_time_list = c(computation_time_list, row_gpu$computation_time)
        simulated_HR_list = c(simulated_HR_list, row_gpu$simulated_HR)
        beta_list = c(beta_list, row_gpu$beta)
        biais_beta_list = c(biais_beta_list, row_gpu$biais_beta)
        BIC_list = c(BIC_list, row_gpu$BIC)


        gpu_WCE = result_gpu$best_model_result$WCEmat

        WCE_matrix[i_iteration*2, 1:cutoff] = gpu_WCE


        

    }



    print(result_cpu$WCEmat)


    result_df = data.frame(
    iteration = iteration_list,
    library = library_list,
    scenario_name = scenario_name_list,
    HR_target = HR_target_list,
    computation_time = computation_time_list,
    simulated_HR = simulated_HR_list,
    beta = beta_list,
    biais_beta = biais_beta_list,
    BIC = BIC_list
    )


    for (i in 1:cutoff){
        result_df[paste0("t", i)] = WCE_matrix[, i]
    }



    return(result_df)




}

################### SIMULATIONS 

print("start")


n_patients = 500
max_time = 365
scenario_name = "exponential_scenario"
HR_target = 4
cutoff = 180


# df = launch_experiment(
#     n_experiment = 10,
#     n_patients = n_patients,
#     max_time = max_time,
#     scenario_name = scenario_name,
#     HR_target = HR_target,
#     cutoff = cutoff,
#     constraint = "Right",
#     mode = "GPU"
# )


scenario_list =  c(
    "exponential_scenario", 
     "bi_linear_scenario",
    "early_peak_scenario",
    "inverted_u_scenario"
    )


columns = c("iteration",
                "library",
                "scenario_name", 
                "HR_target",
                "computation_time",
                "simulated_HR",
                "beta",
                "biais_beta",
                "BIC",
                paste0("t", 1:cutoff))

df = data.frame(matrix(ncol = length(columns), nrow = 0))
colnames(df) <- columns



for (scenario in scenario_list){
    scenario_df = launch_experiment(
        n_experiment = 10,
        n_patients = n_patients,
        max_time = max_time,
        scenario_name = scenario,
        HR_target = HR_target,
        cutoff = cutoff,
        constraint = "Right",
        mode = "GPU"
    )

    df = rbind(df, scenario_df)
}



print(df)


# save as csv

write.csv(df, "result_experiment_500.csv")





# dataset = simulation_paper(
#     n_patients = n_patients, 
#     max_time = max_time, 
#     scenario_name = scenario_name,
#     HR_target = HR_target
# )




# result = result_iteration(
#     n_patients = n_patients,
#     max_time = max_time,
#     scenario_name = scenario_name,
#     HR_target = HR_target,
#     cutoff = cutoff,
#     constraint = "Right",
#     mode = "GPU"
# )



# columns = c("iteration",
#                 "library",
#                 "scenario_name", 
#                 "HR_target",
#                 "computation_time",
#                 "simulated_HR",
#                 "beta",
#                 "biais_beta",
#                 "BIC",
#                 paste0("t", 1:cutoff))


# columns = c("iteration",
#                 "library",
#                 "scenario_name", 
#                 "HR_target",
#                 "computation_time",
#                 "simulated_HR",
#                 "beta",
#                 "biais_beta",
#                 "BIC")


# result_df <- data.frame(matrix(ncol = length(columns), nrow = 0))
# colnames(result_df) <- columns



# result = result_iteration(
#     n_patients = 500,
#     max_time = 365,
#     scenario_name = "exponential_scenario",
#     HR_target = 4,
#     cutoff = 180,
#     constraint = "Right",
#     mode = "GPU"
# )






# result_cpu <- result$result_cpu
# result_gpu <- result$result_gpu

# row_cpu = result_to_row(
#     iteration = 1,
#     library = "WCE",
#     scenario_name = "exponential_scenario",
#     HR_target = 4,
#     result = result_cpu,
#     cutoff = 180
# )

# print(row_cpu)


# columns = c("iteration",
#                 "library",
#                 "scenario_name", 
#                 "HR_target",
#                 "computation_time",
#                 "simulated_HR",
#                 "beta",
#                 "biais_beta",
#                 "BIC")


# iteration_list = c(),
# library_list = c(),
# scenario_name_list = c(),
# HR_target_list = c(),
# computation_time_list = c(),
# simulated_HR_list = c(),
# beta_list = c(),
# biais_beta_list = c(),
# BIC_list = c()


# result_df = data.frame(
#     iteration = iteration_list,
#     library = library_list,
#     scenario_name = scenario_name_list,
#     HR_target = HR_target_list,
#     computation_time = computation_time_list,
#     simulated_HR = simulated_HR_list,
#     beta = beta_list,
#     biais_beta = biais_beta_list,
#     BIC = BIC_list
# )

















# print(names(model))

# print(model$info.criterion)

# result_model = analyze_model_WCE(model, cutoff)

# print(result_model)


# model = result$model

# WCEmat = model$WCEmat
# BIC = model$info.criterion
# analyzed_result = analyze_model_survivalgpu(model, 180)


# # print(result$WCEmat)


# i = 1


# final_result = list(
#     iteration = i,
#     library = "WCE",
#     scenario_name = scenario_name,
#     HR_target = HR_target,
#     computation_time = result$time,
#     simulated_HR = analyzed_result$HR,
#     beta = log(analyzed_result$HR),
#     biais_beta = (log(analyzed_result$HR) - log(HR_target))/log(HR_target) * 100,
#     BIC = analyzed_result$BIC#,
#     WCEmet = result$WCEmat
# )

# print(final_result)


# model_1 = use_model(dataset, cutoff, 1, "Right", mode = "WCE")
# model_2 = use_model(dataset, cutoff, 2, "Right", mode = "WCE")
# model_3 = use_model(dataset, cutoff, 3, "Right", mode = "WCE")



# result_1 = analyze_model_WCE(model_1$model, cutoff)
# result_2 = analyze_model_WCE(model_2$model, cutoff)
# result_3 = analyze_model_WCE(model_3$model, cutoff)




# model_list = list(
#     result_1,
#     result_2,
#     result_3
# )

# for(model in model_list){

#     print(model$BIC)
#     print(model$HR)
# }
# print(best_model(model_list)$BIC)
# print(best_model(model_list)$HR)


# model_1 = use_model(dataset, cutoff, 1, "Right", mode = "GPU")
# model_2 = use_model(dataset, cutoff, 2, "Right", mode = "GPU")
# model_3 = use_model(dataset, cutoff, 3, "Right", mode = "GPU")



# result_1 = analyze_model_GPU(model_1$model, cutoff)
# result_2 = analyze_model_GPU(model_2$model, cutoff)
# result_3 = analyze_model_GPU(model_3$model, cutoff)




# model_list = list(
#     result_1,
#     result_2,
#     result_3
# )

# for(model in model_list){

#     print(model$BIC)
#     print(model$HR)
# }
# print(best_model(model_list)$BIC)
# print(best_model(model_list)$HR)









# cutoff = 180

# columns = c("iteration",
#             "library",
#             "scenario_name", 
#             "HR_target",
#             "computation_time",
#             "simulated_HR",
#             "beta",
#             "biais_beta",
#             "BIC"
#             paste0("t", 1:cutoff))
# wce_df <- data.frame(matrix(ncol = length(columns), nrow = 0))
# colnames(wce_df) <- columns


# print(wce_df)