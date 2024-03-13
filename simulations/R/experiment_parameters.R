generate_parameters <- function(){

    #################### Parameters Selection

    experiment_name = "test binarization"

    # static parameters 
    n_bootstraps = 1000
    cutoff = 180

    # variable paramters 
    HR_target_list = c(2.8)#c(1.25,1.5,2,2.8)#c(1)#c(2.8)
    weight_functions_list = c("exponential_weight") #c("bi_linear_weight, early_peak_weight, inverted_u_weight") #c("exponential_weight")
    n_patients_list = c(100,1000)#c(100,1000,10000,100000)#c(100,200,500,1000,2000,5000,10000,20000,50000,100000)#,10000)
    n_knots_list = c(1.25)#c(1,2,3)
    binarization_dose_list = c(TRUE,FALSE) #FALSE


    experiment_parameters <- list(
        "experiment_name" = experiment_name,
        "n_bootstraps" = n_bootstraps,
        "cutoff" = cutoff,
        "HR_target_list" = HR_target_list,
        "weight_functions_list" = weight_functions_list,
        "n_patients_list" = n_patients_list,
        "n_knots_list" = n_knots_list,
        "binarization_dose_list" = binarization_dose_list
    )

    return(experiment_parameters)
}           
