library(WCE)
library(boot)
library(jsonlite)
library(devtools)

devtools::load_all("../../../survivalGPU/R")

##################### EXPERIMENTS DATA #####################

expriment_name = "simple_test"

# static parameters 
n_bootstraps = 10
cutoff = 180

# variable paramters 
HR_cible_list = c(2.8)
weight_functions_list = c("exponential_weight")
n_patients_list = c(100,1000)
n_knots_list = c(1,2)



###########################################################



analyze_data <- function(file_path, cutoff, n_bootstraps,nknots) {

    data = read.csv(file_path)

    model <- wceGPU(data, nknots, cutoff, constrained = "right",
               id = "patient", event = "event", start = "start",
               stop = "stop", expos = "dose",nbootstraps = n_bootstraps,batchsize = 10, verbosity=0)

    exposed   <- rep(1, cutoff)
    non_exposed   <- rep(0, cutoff)
    Hazard_ratio = HR(model,vecnum = exposed, vecdenom= non_exposed)
    return(Hazard_ratio)
}


file_name_translator <- function(n_patients, HR_cible,weight_function){

    HR_cible = 1 ## WILL CHANGE WHEN SIMULATION IMPLEMENT HR 
    file_name = paste0(weight_function,"_",as.character(HR_cible),"_",as.character(n_patients),".csv")
    file_path = file.path("../WCEmat",file_name)


    return(file_path)
}



######## Script


weight_function_results = c()
n_patients_results = c()
n_bootstraps_results = c()
cutoff_results = c()
HR_cible_results = c()
HR_calculated_results = c()
HR_calculated_2_5_results = c()
HR_calculated_97_5_results = c()

combinaisons_parameters <- expand.grid(HR = HR_cible_list,weight_function = weight_functions_list, n_patients = n_patients_list,n_knots= n_knots_list)
print(combinaisons_parameters)

for (i in 1:nrow(combinaisons_parameters)){

    HR_cible <- combinaisons_parameters$HR[i]
    weight_function <- combinaisons_parameters$weight_function[i]
    n_patients <- combinaisons_parameters$n_patients[i]
    nknots <- combinaisons_parameters$n_knots[i]


    file_path = file_name_translator(n_patients,HR_cible,weight_function)
    print(file_path)
    HR_result = analyze_data(file_path, cutoff,n_bootstraps,nknots)
    print(HR_result)

    ### here logic is false, see https://datatofish.com/export-dataframe-to-csv-in-r/ for better
    ## need separate each oine in different list, not like python

    weight_function_results = c(weight_function_results,weight_function)  
    n_patients_results = c(n_patients_results,n_patients)  
    n_bootstraps_results = c(n_bootstraps_results,n_bootstraps)  
    cutoff_results = c(cutoff_results,cutoff)  
    HR_cible_results = c(HR_cible_results,HR_cible)       
    HR_calculated_results = c(HR_calculated_results,HR_result[1])  
    HR_calculated_2_5_results = c(HR_calculated_2_5_results,HR_result[2]) 
    HR_calculated_97_5_results = c(HR_calculated_97_5_results,HR_result[3])  


result_dict = list("weight_function"= weight_function_results,
                   "n_patients"= n_patients_results,
                   "n_bootstraps"= n_bootstraps_results,
                   "cutoff"= cutoff_results,
                   "HR_cible"= HR_cible_results,
                   "HR_calculated"= HR_calculated_results,
                   "HR_calculated_2_5"= HR_calculated_2_5_results,
                   "HR_calculated_97_5"= HR_calculated_97_5_results
                   )

}

result_dict_path = file.path("../Simulation_results",expriment_name)


if (!dir.exists(result_dict_path)){
dir.create(result_dict_path)
} 

file_result_name = paste0("analyzed_",expriment_name,".csv")
file_result_path = file.path(result_dict_path,file_result_name)

write.csv(result_dict, file_result_path)





