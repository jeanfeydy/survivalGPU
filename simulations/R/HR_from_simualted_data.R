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
HR_cible = c(2.8)
weight_function = c("exponential_weight")
n_patients_list = c(100,1000)


###########################################################



analyze_data <- function(file_path, cutoff, n_bootstraps) {

    data = read.csv(file_path)

    model <- wceGPU(data, 1, cutoff, constrained = "right",
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

for (n_patients in n_patients_list){
    file_path = file_name_translator(n_patients,HR_cible,weight_function)
    print(file_path)
    HR_result = analyze_data(file_path, cutoff,n_bootstraps)
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

print(weight_function_results)
write.csv(result_dict, "test_2.csv")

}







