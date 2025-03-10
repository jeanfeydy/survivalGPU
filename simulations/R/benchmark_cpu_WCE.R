library(WCE)
library(boot)
library(jsonlite)
library(devtools)
library(dplyr)

# options(scipen = 999)

devtools::load_all("../../../survivalGPU/R")

library(survivalGPU)




one_benchmarl <- function(
    n_patients,
    max_time,
    HR_target,
    scenario_name
){
    dataset = simulate_for_experiment(
        n_patients = n_patients, 
        max_time = max_time, 
        scenario_name = scenario_name,
        HR_target = HR_target
    )

    time_start = Sys.time()

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

    time_stop = Sys.time()

    time_cpu = time_stop - time_start

    return(list(n_patients = n_patients,
        time_cpu = time_cpu)
    )

}


time_list = c()
patient_list = c()

for (n_patients in c(
    100,
    500#,
    # 1000
    # 5000,
    # 10000
)){

    survivalgpu <- use_survivalGPU()

    simulate_for_experiment = survivalgpu$simulate_for_experiment

    
    print(paste("n_patients: ", n_patients))

    results = one_benchmarl(
        n_patients = n_patients, 
        max_time = 365,
        HR_target = 4,
        scenario_name = "exponential_scenario"
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



write.csv(results_df, "benchmark.csv", row.names = FALSE)







