library(WCE)
library(boot)
library(jsonlite)
library(devtools)
library(dplyr)

# options(scipen = 999)

devtools::load_all("../../../survivalGPU/R")

library(survivalGPU)


simulation_iteration <- function(
    n_patients,
    max_time, 
    scenario_name,
    HR_target
){
    survivalgpu <- use_survivalGPU()

    simulate_for_experiment = survivalgpu$simulate_for_experiment

    dataset = simulate_for_experiment(
        n_patients = n_patients, 
        max_time = max_time, 
        scenario_name = scenario_name,
        HR_target = HR_target
    )

    model_cpu = WCE::WCE(
        data = dataset,
        analysis = "Cox",
        nknots=c(3),
        cutoff=180,
        id="patients",
        event = "events",
        start = "start",
        stop = "stop",
        expos = "dose",
    )

    exposed   <- rep(1, 180)
    unexposed <- rep(0, 180)

    HR_cpu = WCE::HR.WCE(model_cpu, exposed, unexposed)

    model_gpu = wceGPU(
        data = dataset,
        nknots = 3, 
        cutoff = 180, 
        id = "patients",
        event = "events",
        start = "start",
        stop = "stop",
        expos = "dose",
        constrained = "r"
    )

    HR_gpu = HR(model_gpu, exposed, unexposed)

    return(list(HR_cpu = HR_cpu, HR_gpu = HR_gpu))
}


# HRs = simulation_iteration(
#     n_patients = 500, 
#     max_time = 365,
#     HR_target = 4,
#     scenario_name = "exponential_scenario"
# )




multiple_simulation <- function(
    n_simualtions,
    n_patients,
    max_time, 
    scenario_name,
    HR_target
){
    results_cpu= c()
    results_gpu= c()

    for(i in 1:n_simualtions){
        HRs = simulation_iteration(
            n_patients = n_patients, 
            max_time = max_time,
            HR_target = HR_target,
            scenario_name = scenario_name
        )

        results_cpu = c(results_cpu, HRs$HR_cpu)
        results_gpu = c(results_gpu, HRs$HR_gpu)
        
    }


    analysis_CPU = result_analysis(results_cpu, HR_target)
    analysis_GPU = result_analysis(results_gpu, HR_target)

    return(list(
        analysis_CPU = analysis_CPU,
        analysis_GPU = analysis_GPU
    ))




}


result_analysis <- function(result_list, HR_target){

    true_beta <- log(HR_target)
    mean_beta <- mean(log(result_list))
    sd_beta <- sd(log(result_list))
    biais <- (true_beta - mean_beta)/true_beta * 100


    return(list(
        true_beta = true_beta,
        mean_beta = mean_beta, 
        sd_beta = sd_beta, 
        biais = biais))
}






# multiple_simulation(n_simualtions = 5,
#     n_patients = 500, 
#     max_time = 365,
#     HR_target = 4,
#     scenario_name = "exponential_scenario"
# )


# for(scenario_name in c(
#     "exponential_scenario", 
#     "bi_linear_scenario", 
#     "early_peak_scenario",
#     "inverted_u_scenario")){
#     print(scenario_name)
#     result = multiple_simulation(n_simualtions = 2,
#         n_patients = 500, 
#         max_time = 365,
#         HR_target = 4,
#         scenario_name = scenario_name
#     )

#     print(result)


# }

results <- list()

for(scenario_name in c(
    "exponential_scenario", 
    "bi_linear_scenario", 
    "early_peak_scenario",
    "inverted_u_scenario")){
    print(scenario_name)
    result = multiple_simulation(n_simualtions = 2,
        n_patients = 500, 
        max_time = 365,
        HR_target = 4,
        scenario_name = scenario_name
    )

    results[[scenario_name]] <- result
}

results_df <- do.call(rbind, lapply(names(results), function(scenario) {
    data.frame(
        scenario = scenario,
        true_beta_CPU = results[[scenario]]$analysis_CPU$true_beta,
        mean_beta_CPU = results[[scenario]]$analysis_CPU$mean_beta,
        sd_beta_CPU = results[[scenario]]$analysis_CPU$sd_beta,
        biais_CPU = results[[scenario]]$analysis_CPU$biais,
        true_beta_GPU = results[[scenario]]$analysis_GPU$true_beta,
        mean_beta_GPU = results[[scenario]]$analysis_GPU$mean_beta,
        sd_beta_GPU = results[[scenario]]$analysis_GPU$sd_beta,
        biais_GPU = results[[scenario]]$analysis_GPU$biais,
        biais_diff = results[[scenario]]$analysis_CPU$biais - results[[scenario]]$analysis_GPU$biais,
        sted_diff = results[[scenario]]$analysis_CPU$sd_beta - results[[scenario]]$analysis_GPU$sd_beta
    )

    
}))

write.csv(results_df, "simulation_results.csv", row.names = FALSE)







