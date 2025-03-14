library(WCE)
library(boot)
library(jsonlite)
library(devtools)
library(dplyr)

# options(scipen = 999)

devtools::load_all("../../../survivalGPU/R")

library(survivalGPU)

drugdata <- WCE::drugdata

print(head(drugdata))


model_cpu = WCE::WCE(
        data = drugdata,
        analysis = "Cox",
        nknots=c(3),
        cutoff=180,
        id="Id",
        event = "Event",
        start = "Start",
        stop = "Stop",
        expos = "dose",
        constrained = "Right"
    )

exposed   <- rep(1, 180)
unexposed <- rep(0, 180)

HR_cpu = WCE::HR.WCE(model_cpu, exposed, unexposed)

model_gpu = wceGPU(
    data = drugdata,
    nknots = 3, 
    cutoff = 180, 
    id="Id",
    event = "Event",
    start = "Start",
    stop = "Stop",
    expos = "dose",
    constrained = "Right",
    verbosity = 0
)



HR_gpu = HR(model_gpu, exposed, unexposed)

print(paste("HR_cpu : "))
print(HR_cpu)
print(paste("HR_gpu : "))
print(HR_gpu)
