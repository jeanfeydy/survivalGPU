library(WCE)
library(boot)
library(jsonlite)
library(devtools)

options(scipen = 999)

devtools::load_all("../../../survivalGPU/R")



source("../src/data_simulation.r")
source("../src/weight_functions.r")

########################################## EXPERIMENTS DATA ##########################################

expriment_name = "06-03-2024 : all functions, 100-10000"

# static parameters 
n_bootstraps = 100
cutoff = 180

# variable paramters 
HR_target_list = c(1.25,1.5,2,2.8)
weight_functions_list = c("bi_linear_weight") #c("exponential_weight")
n_patients_list = c()#,10000)
n_knots_list = c(1,2,3)



######################################################################################################








######## Script


weight_function_results = c()
n_patients_results = c()
n_bootstraps_results = c()
cutoff_results = c()
HR_target_results = c()
HR_GPU_bootstraps_results = c()
HR_GPU_bootstraps_2_5_results = c()
HR_GPU_bootstraps_97_5_results = c()
HR_GPU_results = c()
HR_CPU_resutls = c()

print(HR_target_list)
print(weight_functions_list)
print(n_patients_list)
print(n_knots_list)

combinaisons_parameters <- expand.grid(HR_target = HR_target_list,weight_function = weight_functions_list, n_patients = n_patients_list,n_knots= n_knots_list)
print(combinaisons_parameters)

result_dict_path = file.path("../Simulation_results",expriment_name)

if (!dir.exists(result_dict_path)){
dir.create(result_dict_path)
} 


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


            
    # wce_model_GPU <- wceGPU(data, n_knots, cutoff, constrained = "R",
    #            id = "patient", event = "event", start = "start",
    #            stop = "stop", expos = "dose", verbosity=0)


    # wce_model_CPU <- WCE(data,analysis = "cox", nknots = n_knots, cutoff = cutoff, constrained = "R",
    #        id = "patient", event = "event", start = "start",
    #        stop = "stop", expos = "dose")

    wce_model_GPU_bootstraps <- wceGPU(data, n_knots, cutoff, constrained = "R",
               id = "patient", event = "event", start = "start",
               stop = "stop", expos = "dose",nbootstraps = n_bootstraps,batchsize = batchsize, verbosity=0)



    exposed   <- rep(1, cutoff)
    non_exposed   <- rep(0, cutoff)
    
    # HR_result_CPU = HR.WCE(wce_model_CPU,vecnum = exposed, vecdenom= non_exposed)

    # HR_result_GPU = HR(wce_model_GPU,vecnum = exposed, vecdenom= non_exposed)

    HR_result_GPU_bootstraps = HR(wce_model_GPU_bootstraps,vecnum = exposed, vecdenom= non_exposed)

    # print(paste0("HR CPU :",HR_result_CPU))
    # print(paste0("HR GPU :",HR_result_GPU))
    print(paste0("HR GPU_bootstraps :",HR_result_GPU_bootstraps))

    # print(exp(sum(wce_model$WCEmat[1,])))

    # print(wce_model$WCEmat[,1])

    t = 1:cutoff
    # distrib = (lapply.mean(wce_model$WCEmat[,1]))

    print("########")
    scenario_function = scenario_translator(weight_function)
    target_WCE_function = calcul_exposition(scenario_function,HR_target,cutoff)
    print(exp(sum(target_WCE_function)))

    # print(wce_model$WCEmat)

    mean_WCE_function_result_GPU_bootstraps = c()
    lower_WCE_function_result_GPU_bootstraps = c()
    higher_WCE_function_result_GPU_bootstraps = c()

    # GPU_result = c()
    # CPU_result = c()


    

    for (t in 1:cutoff){

        quantiles = quantile(wce_model_GPU_bootstraps$WCEmat[,t],probs=c(0.025,0.975))
        
        mean_WCE_function_result_GPU_bootstraps = c(mean_WCE_function_result_GPU_bootstraps, mean(wce_model_GPU_bootstraps$WCEmat[,t]))
        lower_WCE_function_result_GPU_bootstraps = c(lower_WCE_function_result_GPU_bootstraps, quantiles[1])
        higher_WCE_function_result_GPU_bootstraps = c(higher_WCE_function_result_GPU_bootstraps, quantiles[2])


        # GPU_result = c(GPU_result, wce_model_GPU$WCEmat[,t])
        # CPU_result = c(CPU_result, wce_model_CPU$WCEmat[,t])


    }

    # maximum  = max(max(target_WCE_function),max(GPU_result))
    # minimum = min(0,GPU_result)


    # plot(1:cutoff,
    # target_WCE_function,
    # type="l",
    # col="red",
    # ylim = c(minimum - 0.2*minimum,maximum+0.2*maximum),
    # xlab = "temps (j)",
    # ylab = "WCE weight",
    # main = paste0("GPU ",weight_function," patients : ",n_patients," HR : ",HR_target,"nknots : ",n_knots)
    # )
    # lines(1:cutoff,GPU_result,col="black")

    # maximum  = max(max(target_WCE_function),max(CPU_result))
    # minimum = min(0,CPU_result)

    # plot(1:cutoff,
    # target_WCE_function,
    # type="l",
    # col="red",
    # ylim = c(minimum - 0.2*minimum,maximum+0.2*maximum),
    # xlab = "temps (j)",
    # ylab = "WCE weight",
    # main = paste0("CPU ",weight_function," patients : ",n_patients," HR : ",HR_target,"nknots : ",n_knots)
    # )
    # lines(1:cutoff,CPU_result,col="black")


    maximum  = max(max(higher_WCE_function_result_GPU_bootstraps),max(target_WCE_function))
    minimum = min(0,lower_WCE_function_result_GPU_bootstraps)

    file_name = paste0(weight_function," patients : ",n_patients," HR : ",HR_target," nknots : ",n_knots,".png")

    file_path = file.path("../Simulation_results",expriment_name,file_name)

    png(file_path, width = 800, height = 600)

    plot(1:cutoff,
    target_WCE_function,
    type="l",
    col="red",
    ylim = c(minimum - 0.2*minimum,maximum+0.2*maximum),
    xlab = "time (days)",
    ylab = "WCE weight",
    main = paste0(weight_function," patients : ",n_patients," HR : ",HR_target," nknots : ",n_knots)
    )
    lines(1:cutoff,mean_WCE_function_result_GPU_bootstraps,col="black")
    lines(1:cutoff,lower_WCE_function_result_GPU_bootstraps,col ="black", lty ="dashed" )    
    lines(1:cutoff,higher_WCE_function_result_GPU_bootstraps,col ="black", lty ="dashed" )

    dev.off()



    weight_function_results = append(weight_function_results,weight_function) #c(weight_function_results,weight_function)  
    n_patients_results = c(n_patients_results,n_patients)  
    n_bootstraps_results = c(n_bootstraps_results,n_bootstraps)  
    cutoff_results = c(cutoff_results,cutoff)  
    HR_target_results = c(HR_target_results,HR_target)       
    HR_GPU_bootstraps_results = c(HR_GPU_bootstraps_results,HR_result_GPU_bootstraps[1])  
    HR_GPU_bootstraps_2_5_results = c(HR_GPU_bootstraps_2_5_results,HR_result_GPU_bootstraps[2]) 
    HR_GPU_bootstraps_97_5_results = c(HR_GPU_bootstraps_97_5_results,HR_result_GPU_bootstraps[3])  
    # HR_GPU_results = c(HR_GPU_results,HR_result_GPU)
    # HR_CPU_resutls = c(HR_CPU_resutls,HR_result_CPU)

print("############ Debug Weight function resutls")

result_dict = list("weight_function"= weight_function_results,
                   "n_patients"= n_patients_results,
                   "n_bootstraps"= n_bootstraps_results,
                   "cutoff"= cutoff_results,
                   "HR_target"= HR_target_results,
                   "HR_calculated_GPU_bootstraps"= HR_GPU_bootstraps_results,
                   "HR_calculated_GPU_bootstraps_2_5"= HR_GPU_bootstraps_2_5_results,
                   "HR_calculated_GPU_bootstraps_97_5"= HR_GPU_bootstraps_97_5_results
                #    "HR_GPU" = HR_GPU_results,
                #    "HR_CPU" = HR_CPU_resutls
                   )

}

result_dict_path = file.path("../Simulation_results",expriment_name)

file_result_name = paste0("analyzed_",expriment_name,".csv")
file_result_path = file.path(result_dict_path,file_result_name)

write.csv(result_dict, file_result_path)





