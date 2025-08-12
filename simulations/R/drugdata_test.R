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
        constrained = "Right",
        covariates = c("age", "sex")
    )



# HR_cpu = WCE::HR.WCE(model_cpu, exposed, unexposed)

# BIC = model_cpu$info.criterion


# for (nknot in c(1,2,3)){
#     for (cutoff in c(90, 180, 270)){
#         model_cpu = WCE::WCE(
#             data = drugdata,
#             analysis = "Cox",
#             nknots=c(nknot),
#             cutoff=cutoff,
#             id="Id",
#             event = "Event",
#             start = "Start",
#             stop = "Stop",
#             expos = "dose",
#             constrained = "Right",
#             covariates = c("age", "sex")
#         )
#         BIC = c(BIC, model_cpu$info.criterion)
#         HR = WCE::HR.WCE(model_cpu, exposed, unexposed)
    
#     }
# }



# HR_cpu_list = c()


model_gpu = wceGPU(
    data = drugdata,
    nknots = 1, 
    cutoff = 180, 
    id="Id",
    event = "Event",
    start = "Start",
    stop = "Stop",
    expos = "dose",
    constrained = "Right",
    covariates = c("age","sex"),
    verbosity = 0,
    nbootstraps = 100
)


print("############## GPU")


print(summary(model_gpu))





exposed   <- rep(1, 180)
unexposed <- rep(0, 180)
HR_gpu = HR(model_gpu, exposed, unexposed)

plot(model_gpu)


print(HR_gpu)
print(paste0("HR : ", HR_cpu))

# print(paste0("coef :",model_gpu$coef))
# print(paste0("SE :",model_gpu$SE))

# print("GPU coef")
# print(model_gpu$coef)

# print("GPU SE")
# print(model_gpu$SE)


print(summary(model_gpu))



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
            constrained = "Right",
            covariates = c("age","sex"))


print("############## CPU")

HR_cpu = WCE::HR.WCE(model_cpu, exposed, unexposed)

print(paste0("WCE.HR : ", HR_cpu))
# print(names(model_cpu))
# print(paste0("covariates :",model_cpu$covariates))
# print(paste0("se.covariates :",model_cpu$se.covariates))

plot(model_cpu)


print(summary(model_cpu))







quit()
nknots_list = c()
beta_cpu_wce_list = c()
beta_age_cpu_list = c()
beta_sex_cpu_list = c()
sd_age_cpu_list = c()
beta_gpu_wce_list = c()
beta_age_gpu_list = c()
beta_sex_gpu_list = c()
sd_age_gpu_list = c()
BIC_cpu_list = c()



for (nknot in c(1,2,3)){
    
        model_cpu = WCE::WCE(
            data = drugdata,
            analysis = "Cox",
            nknots=c(nknot),
            cutoff=180,
            id="Id",
            event = "Event",
            start = "Start",
            stop = "Stop",
            expos = "dose",
            constrained = "Right",
            covariates = c("age","sex"))

        print(names(model_cpu))
        

        BIC_cpu = model_cpu$info.criterion
        exposed   <- rep(1, 180)
        unexposed <- rep(0, 180)
        HR_cpu = WCE::HR.WCE(model_cpu, exposed, unexposed)
        beta_cpu_wce = log(HR_cpu)
        beta_age_cpu = model_cpu$covariates[1]
        sd_age_cpu = model_cpu$se.covariates[1]
        beta_sex_cpu = model_cpu$covariates[2]
        sd_sex_cpu = model_cpu$se.covariates[2]


        nknots_list = c(nknots_list, nknot)
        beta_cpu_wce_list = c(beta_cpu_wce_list, beta_cpu_wce)
        beta_age_cpu_list = c(beta_age_cpu_list, beta_age_cpu)
        sd_age_cpu_list = c(sd_age_cpu_list, sd_age_cpu)
        beta_sex_cpu_list = c(beta_sex_cpu_list,beta_sex_cpu)
        BIC_cpu_list = c(BIC_cpu_list, BIC_cpu)

        plot(model_cpu)

        
        


        model_gpu = wceGPU(
            data = drugdata,
            nknots = nknot, 
            cutoff = 180, 
            id="Id",
            event = "Event",
            start = "Start",
            stop = "Stop",
            expos = "dose",
            constrained = "Right",
            verbosity = 0,
            covariates = c("age","sex")
        )

        BIC_gpu =model_gpu$info.criterion
        HR_gpu = HR(model_gpu, exposed, unexposed)[1]
        beta_gpu_wce = log(HR_gpu)
        beta_age_gpu = beta_gpu_wce[1]
        beta_sex_gpu = beta_gpu_wce[2]


        beta_gpu_wce_list = c(beta_gpu_wce_list, beta_gpu_wce)
        beta_age_gpu_list = c(beta_age_gpu_list, beta_age_gpu)
        beta_sex_gpu_list = c(beta_sex_gpu_list,beta_sex_gpu)

        print(beta_sex_gpu_list)




}


df_results = data.frame(
    nknots = nknots_list,
    beta_cpu_wce = beta_cpu_wce_list,
    beta_age_cpu = beta_age_cpu_list,
    beta_sex_cpu = beta_sex_cpu_list,
    beta_gpu_wce = beta_gpu_wce_list,
    beta_age_gpu = beta_age_gpu_list,
    beta_sex_gpu = beta_sex_gpu_list
)


print(df_results)
print()




        



# model_gpu = wceGPU(
#     data = drugdata,
#     nknots = 3, 
#     cutoff = 180, 
#     id="Id",
#     event = "Event",
#     start = "Start",
#     stop = "Stop",
#     expos = "dose",
#     constrained = "Right",
#     verbosity = 0,
#     covariates = c("age","sex")
# )



# HR_gpu = HR(model_gpu, exposed, unexposed)

# print(paste("HR_cpu : "))
# print(HR_cpu)
# print(paste("HR_gpu : "))
# print(HR_gpu)
