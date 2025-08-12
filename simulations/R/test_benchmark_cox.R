library(WCE)
library(boot)
library(jsonlite)
library(devtools)
library(dplyr)
library(bootstrap)

# options(scipen = 999)

#devtools::load_all("../../../survivalGPU/R")
devtools::load_all("../../../survivalGPU/R")

library(survivalGPU)
library(survival)

n_patients = 10000

dataset_path =paste0("../benchmark_datasets/", n_patients, ".csv")
dataset = read.csv(dataset_path)

print(head(dataset))


time_start = Sys.time()
coxphGPU_breslow <- coxphGPU(Surv(start, stop, events) ~ dose,
                             dataset,
                             ties = "breslow"
)
time_stop = Sys.time()
time_gpu = time_stop - time_start


time_start = Sys.time()
coxph_survival <- coxph(Surv(start, stop, events) ~ dose, data = dataset, ties = "breslow")
summary(coxph_survival)
time_stop = Sys.time()

time_cpu = time_stop - time_start



summary(coxphGPU_breslow)


library(survival)

# # Function to fit Cox model and return coefficients
# fit_cox_model <- function(data) {
#   model <- coxph(Surv(start, stop, events) ~ dose, data = data, ties = "breslow")
#   return(coef(model))
# }

# # Manual bootstrap function
# bootstrap_cox <- function(data, B = 1000, seed = 123) {
#   set.seed(seed)
#   n <- nrow(data)
  
#   # Initial fit to get number and names of coefficients
#   init_fit <- tryCatch(fit_cox_model(data), error = function(e) rep(NA, 1))
#   p <- length(init_fit)
#   boot_coefs <- matrix(NA, nrow = B, ncol = p)

#   for (b in 1:B) {
#     boot_idx <- sample(1:n, replace = TRUE)
#     boot_data <- data[boot_idx, ]
    
#     # Fit and store results
#     fit <- tryCatch({
#       fit_cox_model(boot_data)
#     }, error = function(e) rep(NA, p))
    
#     boot_coefs[b, ] <- fit
#   }

#   colnames(boot_coefs) <- names(init_fit)
#   return(boot_coefs)
# }

# # Timing the bootstrap run
# time_start <- Sys.time()
# boot_results <- bootstrap_cox(dataset, B = 10, seed = 123)
# time_stop <- Sys.time()
# time_bootstrap <- time_stop - time_start

# # Show runtime and results summary
# print(time_bootstrap)
# print(apply(boot_results, 2, mean, na.rm = TRUE))
# print(apply(boot_results, 2, sd, na.rm = TRUE))

# # print("time results")

# print("time_gpu:")
# print(time_gpu)
# print("time_cpu:")
# print(time_cpu)
# print("Bootstrap time:")
# print(time_bootstrap)


one_benchmark <- function(
    dataset
){


    time_start = Sys.time()

    model <- coxph(Surv(start, stop, events) ~ dose, data = dataset, ties = "breslow")
    
    model_cox = 


    time_stop = Sys.time()

    time_cpu = time_stop - time_start

    return(list(n_patients = n_patients,
        time_cpu = time_cpu)
    )

}

results_list = list()


for(i in c(
    500,
    1000,
    5000,
    10000,
    50000
)){

    dataset_path =paste0("../benchmark_datasets/", i, ".csv")

    dataset = read.csv(dataset_path)

    print(paste("n_patients: ", i))

    results = one_benchmark(
        dataset
    )

    print(results)

    results_list[[as.character(i)]] = results

}



# for(i in c(
#     500,
#     1000,
#     5000,
#     10000,
#     50000
# )){

#     dataset_path =paste0("../benchmark_datasets/", i, ".csv")

#     dataset = read.csv(dataset_path)

#     print(paste("n_patients: ", i))

#     results = one_benchmark(
#         dataset
#     )

#     print(results)

#     results_list[[as.character(i)]] = results

# }




# print(results_list)


