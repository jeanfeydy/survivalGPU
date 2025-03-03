library(WCE)
library(boot)
library(jsonlite)
library(devtools)
library(dplyr)

# options(scipen = 999)

devtools::load_all("../../../survivalGPU/R")


# dataset = WCE::drugdata

# print(head(dataset))



library(survivalGPU)


    # list_covariates = 


#Xmat = WCE::drugdata$Xmat




# dataset = simulation_paper(
#     n_patients = 500, 
#     max_time = 365,
#     HR_target = 4,
#     scenario_name = "exponential_scenario"
# )

# print(head(dataset))



# model_cpu = WCE::WCE(
#     data = dataset,
#     analysis = "Cox",
#     nknots=c(3),
#     cutoff=180,
#     id="patients",
#     event = "events",
#     start = "start",
#     stop = "stop",
#     expos = "dose",
# )
# exposed   <- rep(1, 180)
# unexposed <- rep(0, 180)

# print(WCE::HR.WCE(model_cpu, exposed, unexposed))

# model_gpu = wceGPU(
#     data = dataset,
#     nknots = 3, 
#     cutoff = 180, 
#     id = "patients",
#     event = "events", 
#     start = "start", 
#     stop = "stop",
#     expos = "dose",
#     constrained = 'r', 
#     nbootstraps = 1, 
#     batchsize = 0
# )
# HR_cpu = WCE::HR.WCE(model_cpu, exposed, unexposed)
# beta_cpu = log(HR_cpu[1])
# print(beta_cpu)

# biais_beta_cpu = (log(4) - beta_cpu )/log(4) 



# print(biais_beta_cpu)

# HR_gpu = HR(model_gpu, exposed, unexposed)
# beta_gpu = log(HR_gpu[1])
# print(beta_gpu)

# biais_beta_gpu = (log(4) - beta_gpu )/log(4)

# print(biais_beta_gpu)

# print("Biais HR")

# biais_HR_cpu = (4 - HR_cpu[1])/4
# print(biais_HR_cpu)

# biais_HR_gpu = (4 - HR_gpu[1])/4
# print(biais_HR_gpu)







# print(Xmat)
# print(WCE::drugdata$Event)

# set.seed(123)  # For reproducibility
# Xmat <- matrix(sample(0:1, 365*500*3, replace = TRUE), ncol = 3)


# permalgoWCE(
#     n_patients = 500, 
#     max_time = 365,
#     Xmat = Xmat,
#     betas = c(2,1.5,1.7),
#     names = c("WCE"),
#     wce_status = c(TRUE,FALSE,FALSE),
#     scenario = "exponential_scenario",
#     eventRandom = NULL,
#     censorRandom = NULL)




