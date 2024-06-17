generate_parameters <- function(){

    #################### Parameters Selection



    # Experiment parameters
    experiment_name = "Test reproducibility"
    number_of_simulations = 10
    save_models = TRUE


    # Simulation parameters 

    HR_target_list = c(1.5)#c(1,1.25,1.5,2,2.8)#c(1.25,1.5,2,2.8)#c(1)#c(2.8)#c(1)#c(1.25,1.5,2,2.8)#c(1)#c(2.8)
    scenario_functions_list = c("exponential_scenario")#c("early_peak_scenario")#c("bi_linear_scenario") #, early_peak_scenario, inverted_u_scenario") #c("exponential_scenario")
    n_patients_list = c(50)#c(100,200,500,1000,2000,5000,10000,20000,50000,100000)#c(100,200,500,1000,2000,5000,10000) #,20000) #,50000,100000)#c(100,1000,10000,100000)##c(100,1000,10000,100000)#c(100,200,500,1000,2000,5000,10000,20000,50000,100000)#,10000)
    max_time_list = c(365)
    binarization_dose_list = c(FALSE) #,TRUE) #FALSE
    doses_list = list(c(1,1.5,2,2.5,3))
    

    # analysis parameters

    n_knots_list = c(1)#c(1,2,3)
    n_bootstraps_list = c(50)
    cutoff_list = c(180)
    constraint_list = ("R")



    simulation_parameters <- list(
        "HR_target_list" = HR_target_list,
        "scenario_functions_list" = scenario_functions_list,
        "n_patients_list" = n_patients_list,
        "max_time_list" = max_time_list,
        "binarization_dose_list" = binarization_dose_list,
        "doses_list" = doses_list)

    analysis_parameters <- list(
        "n_knots_list" = n_knots_list,
        "n_bootstraps_list" = n_bootstraps_list,
        "cutoff_list" = cutoff_list,
        "constraint_list" = constraint_list
    )

    experiment_parameters <- list(
        "experiment_name" = experiment_name,
        "save_models" = save_models,
        "number_of_simulations" = number_of_simulations,
        "simulation" = simulation_parameters,
        "analysis" = analysis_parameters

    )



    return(experiment_parameters)
}           
