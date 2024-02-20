# imports 

source("src/data_simulation.r")
source("src/weight_functions.r")

scenario_list <- list(
    list(name ="exponential_weight", weights = exponential_weight),
    list(name ="bi_linear_weight", weights = bi_linear_weight),
    list(name ="early_peak_weight", weights = early_peak_weight),
    list(name ="inverted_u_weight", weights = inverted_u_weight),
    list(name ="constant_weight", weights = constant_weight),
    list(name ="late_effect_weight", weights = late_effect_weight)

    
)

normalize_shape <- function(scenario,cutoff,normalization){
    expo_list <- lapply((1:365)/365, scenario)
    expo <- do.call("rbind", expo_list)/365
    print(expo)

    integral <- integrate(scenario, lower = 1/365, upper = 1)
    normalization_factor =  normalization/integral$value


    return(expo*normalization_factor)
}
normalization = 1

for (scenario in scenario_list){

    correct_shape <- normalize_shape(scenario$weights,cutoff,normalization)
    export_path <- paste0("weight_functions_shapes/",scenario$name,"_",normalization,".csv")
    write.csv(correct_shape, export_path, row.names=FALSE)
}

# null weight 

expo_list <- lapply((1:365)/365, null_weight) 
expo <- do.call("rbind",expo_list)
export_path <- paste0("weight_functions_shapes/null_weigh_",normalization,".csv")
write.csv(correct_shape, export_path, row.names=FALSE)
