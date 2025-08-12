library(WCE)
library(boot)
library(jsonlite)
library(devtools)
library(dplyr)

# options(scipen = 999)

#devtools::load_all("../../../survivalGPU/R")
devtools::load_all("../../../survivalGPU/R")

library(survivalGPU)


bootstrap_WCE <- function(data, model_function, id_var = "Id", cutoff = 180, B = 10) {
  
  ID <- unique(data[[id_var]])
  boot.WCE <- matrix(NA, ncol = cutoff, nrow = B)
  boot.HR <- rep(NA, B)
  max_id <- max(data[[id_var]])
  id_shift <- 10^ceiling(log10(max_id))
  
  for (i in 1:B) {
    ID.resamp <- sort(sample(ID, replace = TRUE))
    datab <- data[data[[id_var]] %in% ID.resamp, ]
    
    step <- 1
    repeat {
      ID.resamp <- ID.resamp[duplicated(ID.resamp)]
      if (length(ID.resamp) == 0) break
      
      subset.dup <- data[data[[id_var]] %in% ID.resamp, ]
      subset.dup[[id_var]] <- subset.dup[[id_var]] + step * id_shift
      datab <- rbind(datab, subset.dup)
      step <- step + 1
    }
    
    mod <- model_function(datab)
    
    best <- which.min(mod$info.criterion)
    boot.WCE[i, ] <- mod$WCEmat[best, ]
    boot.HR[i] <- HR.WCE(mod, rep(1, cutoff), rep(0, cutoff))
  }
  
  list(
    boot.WCE = boot.WCE,
    boot.HR = boot.HR,
    WCE_CI = apply(boot.WCE, 2, quantile, p = c(0.05, 0.95)),
    HR_CI = quantile(boot.HR, p = c(0.05, 0.95))
  )
}




n_patients = 10000

dataset_path =paste0("../benchmark_datasets/", n_patients, ".csv")
dataset = read.csv(dataset_path)

B = 1000


model_gpu <- function(data) {
  wceGPU(
    data = data,
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
}

model_cpu <- function(data) {
  WCE::WCE(
        data = data,
        analysis = "Cox",
        nknots=c(3),
        cutoff=180,
        id="patients",
        event = "events",
        start = "start",
        stop = "stop",
        expos = "dose",
    )
}


time_start <- Sys.time()

# results <- bootstrap_WCE(data = dataset, model_function = model_cpu, id_var = "patients", B = B)

time_stop <- Sys.time()
time_bootstrap_manual <- time_stop - time_start



time_start <- Sys.time()
model_gpu <-  wceGPU(
    data = dataset,
    nknots = 3, 
    cutoff = 180, 
    id = "patients",
    event = "events",
    start = "start",
    stop = "stop",
    expos = "dose",
    constrained = "r",
    verbosity = 0,
    nbootstraps = B,
    batchsize = 10
  )


time_stop <- Sys.time()

time_bootstrap_survivalgpu <- time_stop - time_start

print(time_bootstrap_manual)

print(time_bootstrap_survivalgpu)




# # Extract results
# results$WCE_CI
# results$HR_CI