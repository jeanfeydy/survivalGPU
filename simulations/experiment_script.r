# import

library(dplyr)
library(WCE)
library(purrr)

source("src/data_simulation.r")
source("src/weight_functions.r")
source("../R/R/wceGPU.R")

# def scenario


# def Xmat 
n_patients <- 1000
observation_time <- 365
doses <- c(0.5,1,1.5,2,2.5,3)


normalization <- 1

Xmat <- generate_Xmat(observation_time,n_patients,doses)

# def experiments

scneario_list <- c(exponential_weight,bi_linear_weight)

experiments_variables_list <- list(
    list(n_bootstraps = 1, n_samples =50, n_knots = 1, cutoff = 180),
    list(n_bootstraps = 1, n_samples =100, n_knots = 1, cutoff = 180)
)


get_df_wce <- function(Xmat, scenario, normalization){
    wce_mat <- do.call("rbind", lapply(1:dim(Xmat)[1], wce_vector, scenario = scenario, Xmat = Xmat,normalization = normalization))
    df_wce <- get_dataset(Xmat = Xmat, wce_mat = wce_mat)
    return(df_wce)
}

sample_df <- function(dataframe, n) {
   
    sample_indices <- sample(1:nrow(dataframe), n)
    sampled_dataframe <- dataframe[sample_indices, , drop = FALSE]
    
    return(sampled_dataframe)
}
# TODO find why sometime I have Na in probability vector 

experiment <- function(
    df_wce,
    normalization,
    scenario,
    experiment_variables){

    print(experiment_variables)

    n_knots <- experiment_variables$n_knots
    cutoff <- experiment_variables$cutoff
    n_bootstraps <- experiment_variables$n_bootstraps
    n_samples <- experiment_variables$n_samples



    df_wce_sample <-  sample_df(df_wce,n_samples)



    wce_right_constrained <- WCE(df_wce, "Cox", n_knots, cutoff, constrained = "right",
                         id = "patient", event = "event", start = "start",
                         stop = "stop", expos = "dose")

    wce_unconstrained <- WCE(df_wce, "Cox", n_knots, cutoff, constrained = FALSE,
                             id = "patient", event = "event", start = "start",
                             stop = "stop", expos = "dose")

    return(list(
        righ_constrained <- wce_right_constrained,
        unconstrained    <- wce_unconstrained
    ))                   
    }





for (scenario in scneario_list){
    df_wce <- get_df_wce(Xmat, scenario, normalization)
    print(head(df_wce))

    print(sample_df(df_wce,10))

    print(experiments_variables_list)

    for(experiment_variables in experiments_variables_list){
        print(experiment_variables)
        experiment(df_wce,normalization,scenario,experiment_variables)


    }


}
