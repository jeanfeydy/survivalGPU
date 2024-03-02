library(WCE)
library(boot)
library(jsonlite)
library(devtools)

options(scipen = 999)

devtools::load_all("../../../survivalGPU/R")



source("../src/data_simulation.r")
source("../src/weight_functions.r")

########################################## EXPERIMENTS DATA ##########################################

expriment_name = "HR : 100 - 100 000"

# static parameters 
n_bootstraps = 1000
cutoff = 180

# variable paramters 
HR_target_list = c(2)
weight_functions_list = c("exponential_weight") #c("exponential_weight")
n_patients_list = c(1000)#,10000)
n_knots_list = c(1)


######################################################################################################








######## Script


weight_function_results = c()
n_patients_results = c()
n_bootstraps_results = c()
cutoff_results = c()
HR_target_results = c()
HR_calculated_results = c()
HR_calculated_2_5_results = c()
HR_calculated_97_5_results = c()

combinaisons_parameters <- expand.grid(HR_target = HR_target_list,weight_function = weight_functions_list, n_patients = n_patients_list,n_knots= n_knots_list)
print(combinaisons_parameters)

for (i in 1:nrow(combinaisons_parameters)){

    HR_target <- combinaisons_parameters$HR_target[i]
    weight_function <- combinaisons_parameters$weight_function[i]
    n_patients <- combinaisons_parameters$n_patients[i]
    n_knots <- combinaisons_parameters$n_knots[i]

    print(paste0("Starting computation for :" ,as.character(n_patients),"patients - HR_target = ",HR_target," - weight function = ",weight_function,"- ",as.character(n_knots)," knots"))


    file_name = paste0(weight_function,"_",as.character(n_patients),".csv")
    file_path = file.path("../WCEmat",paste0("HR-",as.character(HR_target)),file_name)

    print(file_path)


    if (n_patients < 10000){
        batchsize = 100
    }else{
        batchsize = 10
    }



    data = read.csv(file_path)

    wce_model <- wceGPU(data, n_knots, cutoff, constrained = "right",
               id = "patient", event = "event", start = "start",
               stop = "stop", expos = "dose",nbootstraps = n_bootstraps,batchsize = batchsize, verbosity=0)

    wce_model_normal <- WCE(data,analysis = "cox", nknots = n_knots, cutoff = cutoff, constrained = "R",
           id = "patient", event = "event", start = "start",
           stop = "stop", expos = "dose")



    exposed   <- rep(1, cutoff)
    non_exposed   <- rep(0, cutoff)
    
    HR_result_normal = HR.WCE(wce_model_normal,vecnum = exposed, vecdenom= non_exposed)

    HR_result = HR(wce_model,vecnum = exposed, vecdenom= non_exposed)

    print(HR_result_normal)
    print(HR_result)

    # print(exp(sum(wce_model$WCEmat[1,])))

    # print(wce_model$WCEmat[,1])

    t = 1:cutoff
    # distrib = (lapply.mean(wce_model$WCEmat[,1]))

    print("########")
    scenario_function = scenario_translator(weight_function)
    expo = calcul_exposition(scenario_function,HR_target,cutoff)
    print(exp(sum(expo)))

    result = c()

    # print(wce_model$WCEmat)

    for (t in 1:cutoff){
        result = c(result, mean(wce_model$WCEmat[,t]))
    }

    # print(result)



    plot(1:cutoff,result,type="l",col="red")
    lines(1:cutoff,expo,col="green")


    print(HR_result)

    weight_function_results = c(weight_function_results,weight_function)  
    n_patients_results = c(n_patients_results,n_patients)  
    n_bootstraps_results = c(n_bootstraps_results,n_bootstraps)  
    cutoff_results = c(cutoff_results,cutoff)  
    HR_target_results = c(HR_target_results,HR_target)       
    HR_calculated_results = c(HR_calculated_results,HR_result[1])  
    HR_calculated_2_5_results = c(HR_calculated_2_5_results,HR_result[2]) 
    HR_calculated_97_5_results = c(HR_calculated_97_5_results,HR_result[3])  


result_dict = list("weight_function"= weight_function_results,
                   "n_patients"= n_patients_results,
                   "n_bootstraps"= n_bootstraps_results,
                   "cutoff"= cutoff_results,
                   "HR_target"= HR_target_results,
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





