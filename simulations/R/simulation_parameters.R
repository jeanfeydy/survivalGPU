



generate_experiment_parameters <- function(){

    experiment_name <- "test simulation "

    # number of simulations 
    number_of_simulations <- 1

    # Xmat parameters 
    doses_list <- c(1,1.5,2,2.5,3)
    observation_time_list <- c(365)
    n_patients_list <- c(500)#c(200,500,2000,5000,20000,50000,100000)#c(100,200,500,1000,2000,5000,10000,20000,50000,100000)#c(100,1000,10000)

    # Defintion of the target impact
    scenario_list =  c("exponential_weight") #c("inverted_u_weight","early_peak_weight")#
    cutoff_list = c(180)
    HR_target_list = c(1.5)


    simulation_parameters <- list(
        "number_of_simulations" = number_of_simulations,
        "doses_list" = doses_list,
        "observation_time_list" = observation_time_list,
        "n_patients_list" = n_patients_list,
        "scenario_list" = scenario_list,
        "cutoff_list" = cutoff_list,
        "HR_target_list" = HR_target_list

    )

    return(simulation_parameters)


}