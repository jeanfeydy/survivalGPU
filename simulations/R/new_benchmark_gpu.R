library(WCE)
library(boot)
library(jsonlite)
library(devtools)
library(dplyr)

# options(scipen = 999)

#devtools::load_all("../../../survivalGPU/R")
devtools::load_all("../../../survivalGPU/R")

library(survivalGPU)




one_benchmark <- function(
    dataset
){


    time_start = Sys.time()

    model_gpu = wceGPU(
        data = dataset,
        nknots = 3, 
        cutoff = 180, 
        id = "patients",
        event = "events",
        start = "start",
        stop = "stop",
        expos = "dose",
        constrained = "r",
        verbosity = 0
    )



    time_stop = Sys.time()

    time_cpu = time_stop - time_start

    return(list(n_patients = n_patients,
        time_cpu = time_cpu)
    )

}





time_list = c()
patient_list = c()

for (n_patients in c(
    500,
    500,
    1000,
    5000,
    10000,
    50000
)){


    dataset_path =paste0("../benchmark_datasets/", n_patients, ".csv")


    dataset = read.csv(dataset_path)

    


    
    print(paste("n_patients: ", n_patients))

    results = one_benchmark(
        dataset
    )

    time = as.numeric(results$time_cpu, units = "secs")

    time_list = c(time_list, time)


    
    patient_list = c(patient_list, results$n_patients)

    print(time_list)



    
    

}

results_df <- data.frame(
    n_patients = patient_list,
    time_cpu = time_list
)






time_list = c()rite.csv(results_df, "benchmark_survivalgpu_gpu.csv", row.names = FALSE)







# [1]  5.4241710  0.4261839  0.7527981  3.5013433  6.8908169 37.0276775
