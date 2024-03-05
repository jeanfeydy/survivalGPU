library(WCE)
library(boot)
library(jsonlite)
library(devtools)

devtools::load_all("../../survivalGPU/R")


# first test the bootstraps method
# Do it with no bootstraps and 1000 bootsraps 



# wce_right_constrained <- WCE(data, "Cox", 1, cutoff, constrained = "right",
#                              id = "patient", event = "event", start = "start",
#                              stop = "stop", expos = "dose")


# wce_right_constrained
# # system.time()

# wce_statistic <- function(data, indices) {
#   # Subset the data based on bootstrap indices
#   boot_data <- data[indices, ]
#   # Fit the WCE model on the bootstrap sample
#   wce_model <- wce::WCE(data=boot_data, "Cox", 1, 180, constrained = "right",
#                              id = "patient", event = "event", start = "start",
#                              stop = "stop", expos = "dose"
#   # Extract the statistic of interest, e.g., the effect estimate
#   statistic <- summary(wce_model)$coef[...]
#   return(statistic)
# }

options(scipen = 999)


run_cpu <- function(data, n_bootstraps){


    start_time = Sys.time()

    ID <- unique(data$Id)
    boot.WCE <- matrix(NA, ncol = 180, nrow=n_bootstraps)
    boot.HR <- rep(NA, n_bootstraps)


    # Sample IDs with replacement
    for (i in 1:n_bootstraps){
        ID.resamp <- sort(sample(ID, replace=T))
        datab <- data[data$Id %in% ID.resamp,] # select obs. but duplicated Id are ignored
        # deal with duplicated Id and assign them new Id
        step <- 1
        repeat {
        # select duplicated Id in ID.resamp
        ID.resamp <- ID.resamp[duplicated(ID.resamp)==TRUE]
        if (length(ID.resamp)==0) break # stop when no more duplicated Id to deal with
            # select obs. but remaining duplicated Id are ignored
            subset.dup <- data[data$Id %in% ID.resamp,]
            # assign new Id to duplicates
            subset.dup$Id <- subset.dup$Id + step * 10^ceiling(log10(max(data$Id)))
            # 10^ceiling(log10(max(data$Id)) is the power of 10
            # above the maximum Id from original data
            datab <- rbind(datab, subset.dup)
            step <- step+1
        }
        mod <- WCE(data, "Cox", 1, cutoff, constrained = "right",
                   id = "patient", event = "event", start = "start",
                   stop = "stop", expos = "dose")
        # return best WCE estimates and corresponding HR
        best <- which.min(mod$info.criterion)
        boot.WCE[i,] <- mod$WCEmat[best,]
        boot.HR[i] <- HR.WCE(mod, rep(1, 180), rep(0, 180))
        }
        # Summarize bootstrap results using percentile method
        apply(boot.WCE, 2, quantile, p = c(0.05, 0.95))
        quantile(boot.HR, p = c(0.05, 0.95))

    end_time <- Sys.time()
    time_difference <- end_time - start_time
    computation_time <- as.numeric(time_difference, units = "secs")

    result = list(model = boot, computation_time = computation_time)
    return(computation_time)
}

run_gpu <- function(data,n_bootstraps,n_patients){
    start_time = Sys.time()
    
    if(n_patients < 10000){
        batchsize = 100
    }  
    else{
        batchsize = 10
    } 
    print("#############BATCHSIZE")
    
    print(batchsize)

    model <- wceGPU(data, 1, cutoff, constrained = "right",
                   id = "patient", event = "event", start = "start",
                   stop = "stop", expos = "dose",nbootstraps = n_bootstraps,batchsize=batchsize)


    end_time <- Sys.time()
    time_difference <- end_time - start_time   
    print("######### DEBUG")
    print(time_difference)
    print("######### DEBUG")

    computation_time <- as.numeric(time_difference, units = "secs")

    return(computation_time)

}

run_gpu_no_bootstraps <- function(data){
    start_time = Sys.time()

    model <- wceGPU(data, 1, cutoff, constrained = "right",
                   id = "patient", event = "event", start = "start",
                   stop = "stop", expos = "dose")


    end_time <- Sys.time()
    time_difference <- end_time - start_time   
    print("######### DEBUG")
    print(time_difference)
    print("######### DEBUG")

    computation_time <- as.numeric(time_difference, units = "secs")

    return(computation_time)

}

run_cpu_no_bootstraps <- function(data){
    start_time = Sys.time()

    model <- WCE(data, "Cox", 1, cutoff, constrained = "right",
                   id = "patient", event = "event", start = "start",
                   stop = "stop", expos = "dose")

    end_time <- Sys.time()
    time_difference <- end_time - start_time   
    print("######### DEBUG")
    print(time_difference)
    print("######### DEBUG")

    computation_time <- as.numeric(time_difference, units = "secs")

    return(computation_time)

}


#############" MAIN"

normalization <- 1
cutoff = 180
n_bootstraps = 1000
weight_function = "exponential_weight"

gpu_no_bootstraps_computation_times_list <- list()
gpu_with_bootstraps_computation_times_list <- list()

# n_patients_list = c(100,1000,10000,100000)
n_patients_list = c(100,1000,10000,100000)#,10000)#,10000,100000)



experiment_name <- "GPU with and without bootstraps : 100 - 100 000"


#launch first to test theory first survivalGPU

n_patients = 100
file_name <- paste0("WCEmat/", weight_function,"_",as.character(normalization), "_",as.character(n_patients),".csv")
data = read.csv(file_name)
print("dummy first one")
gpu_no_bootstraps_computation_time = run_gpu_no_bootstraps(data)
print(paste0("Computation for CPU took : ",as.character(gpu_no_bootstraps_computation_time),"s"))


for (n_patients in n_patients_list){
    print(paste0("Start computation for : ",as.character(n_patients)," patients"))
    file_name <- paste0("WCEmat/", weight_function,"_",as.character(normalization), "_",as.character(n_patients),".csv")
    data = read.csv(file_name)


    gpu_no_bootstraps_computation_time = run_gpu_no_bootstraps(data)
    # gpu_no_bootstraps_computation_time = run_cpu_no_bootstraps(data)


    gpu_no_bootstraps_computation_times_list[[as.character(n_patients)]] <- gpu_no_bootstraps_computation_time
    
    print(paste0("Computation for GPU no bootstraps took : ",as.character(gpu_no_bootstraps_computation_time),"s"))

    
    gpu_with_bootstraps_computation_time <- run_gpu(data,n_bootstraps,n_patients)
    # gpu_with_bootstraps_computation_time <- run_gpu_no_bootstraps(data)

    gpu_with_bootstraps_computation_times_list[[as.character(n_patients)]] <- gpu_with_bootstraps_computation_time
    
    
    print(paste0("Computation for GPU with bootstraps took : ",as.character(gpu_with_bootstraps_computation_time),"s"))


    computation_time_output <- list("GPU_no_bootstraps_computaiton_time" = gpu_no_bootstraps_computation_times_list,
                                    "GPU_with_bootstraps_computation_time" = gpu_with_bootstraps_computation_times_list)

    write(toJSON(computation_time_output), file = paste0("Simulation_results/Computation_time_comprison/",experiment_name,".json"))


}






