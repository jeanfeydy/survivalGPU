library(WCE)
library(boot)
library(jsonlite)

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


run_with_bootstraps <- function(data, n_bootstraps){


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
    return(result)
}


#############" MAIN"

normalization <- 1
cutoff = 180
n_bootstraps = 1
weight_function = "exponential_weight"

computation_times_list <- list()

n_patients_list = c(100,1000,10000) #,100000)




for (n_patients in n_patients_list){
    print(paste0("Start computation for : ",as.character(n_patients)," patients"))
    file_name <- paste0("WCEmat/", weight_function,"_",as.character(normalization), "_",as.character(n_patients),".csv")
    data = read.csv(file_name)
    result = run_with_bootstraps(data, n_bootstraps)


    computation_time = result$computation_time
    computation_times_list[[as.character(n_patients)]] <- computation_time
    print(paste0("Computation took : ",as.character(computation_time),"s"))
    write(toJSON(computation_times_list), file = "Simulation_results/computation_time_Rsurvival.json")


}

print(computation_times_list)





