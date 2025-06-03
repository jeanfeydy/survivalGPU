library(WCE)
library(boot)
library(jsonlite)
library(devtools)
library(dplyr)

# options(scipen = 999)

devtools::load_all("../../../survivalGPU/R")

library(survivalGPU)
library(WCE)

drugdata <- WCE::drugdata


result = list()


# Create a data frame of all combinations
params_grid <- expand.grid(
  nknots = c(1, 2, 3),
  constraint = c("Right", FALSE),
  cutoff = c(60, 90, 180, 360),
 stringsAsFactors = FALSE


)

for (i in 1:nrow(params_grid)) {
    model = wceGPU(data = drugdata,
                nknots = params_grid$nknots[i], 
                cutoff = params_grid$cutoff[i], 
                id="Id",
                event = "Event",
                start = "Start",
                stop = "Stop",
                expos = "dose",
                constrained = params_grid$constraint[i],
                covariates = c("age","sex"),
                verbosity = 0)
    
      result[[i]] <- list(nknots = params_grid$nknots[i], 
                          constraint = params_grid$constraint[i], 
                          cutoff = params_grid$cutoff[i], 
                          BIC = model$info.criterion)
}


result_df <- do.call(rbind, lapply(result, as.data.frame))
result_df <- as.data.frame(do.call(rbind, result))

print(result_df)




stop()



# model_1 = wceGPU(
#     data = drugdata,
#     nknots = 1, 
#     cutoff = 180, 
#     id="Id",
#     event = "Event",
#     start = "Start",
#     stop = "Stop",
#     expos = "dose",
#     constrained = "Right",
#     covariates = c("age","sex"),
#     verbosity = 0,
#     # nbootstraps = 100
# )

# model_2 = wceGPU(
#     data = drugdata,
#     nknots = 2, 
#     cutoff = 180, 
#     id="Id",
#     event = "Event",
#     start = "Start",
#     stop = "Stop",
#     expos = "dose",
#     constrained = "Right",
#     covariates = c("age","sex"),
#     verbosity = 0,
#     # nbootstraps = 100
# )


# model_3 = wceGPU(
#     data = drugdata,
#     nknots = 3, 
#     cutoff = 180, 
#     id="Id",
#     event = "Event",
#     start = "Start",
#     stop = "Stop",
#     expos = "dose",
#     constrained = "Right",
#     covariates = c("age","sex"),
#     verbosity = 0,
#     # nbootstraps = 100
# )

# list_results = list(
#     "model_1" = model_1,
#     "model_2" = model_2,
#     "model_3" = model_3
# )

# list_results = 


# wceGPU_knots = function(data, nknots, cutoff, id, event, 
# start, stop, expos, constrained, covariates, verbosity = 0, 
# nbootstraps = 100){

#     model = wceGPU(
#         data = data,
#         nknots = nknots, 
#         cutoff = cutoff, 
#         id = id,
#         event = event,
#         start = start,
#         stop = stop,
#         expos = expos,
#         constrained = constrained,
#         covariates = covariates,
#         verbosity = verbosity,
#         nbootstraps = nbootstraps
#     )
#     return(model)
# }


# model_cpu = WCE::WCE(
#         data = drugdata,
#         analysis = "Cox",
#         nknots=c(1,2,3),
#         cutoff=180,
#         id="Id",
#         event = "Event",
#         start = "Start",
#         stop = "Stop",
#         expos = "dose",
#         constrained = "Right",
#         covariates = c("age", "sex")
#     )


# print(names(model_cpu$est))


# quit()

wce_gpu_60_unconstrained = wceGPU(
    data = drugdata,
    nknots = 1, 
    cutoff = 60, 
    id="Id",
    event = "Event",
    start = "Start",
    stop = "Stop",
    expos = "dose",
    constrained = FALSE,
    covariates = c("age","sex"),
    verbosity = 0,
)


wce_gpu_60_right_constrained = wceGPU(
    data = drugdata,
    nknots = 1, 
    cutoff = 60, 
    id="Id",
    event = "Event",
    start = "Start",
    stop = "Stop",
    expos = "dose",
    constrained = "Right",
    covariates = c("age","sex"),
    verbosity = 0,
)

print(wce_gpu_60_unconstrained$info.criterion)
print(wce_gpu_60_right_constrained$info.criterion)



quit()

wce_gpu = wceGPU(
    data = drugdata,
    nknots = 1, 
    cutoff = 180, 
    id="Id",
    event = "Event",
    start = "Start",
    stop = "Stop",
    expos = "dose",
    constrained = FALSE,
    covariates = c("age","sex"),
    verbosity = 0,
   # nbootstraps = 100
)

summary(wce_gpu)

# plot(wce_gpu)

# exposed   <- rep(1, 180)
# unexposed <- rep(0, 180)
# HR = HR(wce_gpu, exposed, unexposed)
# print(HR)

stop()


wceGPU_knots <- function(data, nknots_list, cutoff, constrained = FALSE, aic = FALSE, id,
                   event, start, stop, expos, covariates = NULL,
                   nbootstraps = 1, batchsize = 0, confint = 0.95,
                   controls = NULL, verbosity = 1,...) {

  wce_results = list()

  for (nknots in nknots_list){

    print(nknots)

    model = wceGPU(data, nknots, cutoff, constrained = constrained, aic = aic, id,
                   event, start, stop, expos, covariates = covariates,
                   nbootstraps = 1, batchsize = 0, confint = 0.95,
                   controls = controls, verbosity = 1,...)

    wce_results[[paste0(as.character(nknots), " knot(s)")]] = model


knot_names <- names(wce_results[[1]])
metric_names <- names(wce_results)
wce_by_knot <- setNames(vector("list", length(knot_names)), knot_names)



for (knot in knot_names) {
    wce_by_knot[[knot]] <- lapply(wce_results, function(metric) metric[[knot]])
    names(wce_by_knot[[knot]]) <- metric_names
    }


  }

  return(wce_by_knot)
}


wce_by_knot = wceGPU_knots(
    data = drugdata,
    nknots = c(1,2,3), 
    cutoff = 180, 
    id="Id",
    event = "Event",
    start = "Start",
    stop = "Stop",
    expos = "dose",
    constrained = "Right",
    covariates = c("age","sex"),
    verbosity = 0,
    # nbootstraps = 100
)
BIC_list <- wce_by_knot$info.criterion
best_knot <- names(BIC_list)[which.min(BIC_list)]
best_model <- lapply(wce_by_knot, function(x) x[[best_knot]])
names(best_model) <- names(wce_by_knot)
summary(best_model)

#print(wce_by_knot$"1 knot(s)"$info.criterion)

# print(model_list$"1_knots"$info.criterion)
# print(model_list$"2_knots"$info.criterion)
# print(model_list$"3_knots"$info.criterion)


# print(names(wce_by_knot))
# print(names(wce_by_knot$info.criterion))
# print(wce_by_knot$info.criterion)



