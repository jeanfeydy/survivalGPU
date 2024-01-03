library(dplyr)
library(WCE)
library(purrr)



# Function to generate an individual time-dependent exposure history
# e.g. generate prescriptions of different durations and doses.
TDhist <- function(observation_time,doses) {
  # Duration : lognormal distribution(0.5,0.8)
  duration <- 7 + 7 * round(rlnorm(1, meanlog = 0.5, sdlog = 0.8), 0)
  # in weeks

  # Dose : random assignment of values 0.5, 1, 1.5, 2, 2.5 and 3
  dose <- sample(doses, size = 1)

  # Start with drug exposure
  vec <- rep(dose, duration)

  # Repeat until the vector is larger than observation_time
  while (length(vec) <= observation_time) {
      intermission <- 7 + 7 * round(rlnorm(1, meanlog = 0.5, sdlog = 0.8), 0) # in weeks
      duration <- 7 + 7 * round(rlnorm(1, meanlog = 0.5, sdlog = 0.8), 0) # in weeks
      dose <- sample(doses, size = 1)
      vec <- append(vec, c(rep(0, intermission), rep(dose, duration)))
  }

  return(vec[1:observation_time])
}

generate_Xmat <- function(observation_time,n_patients,doses){
  Xmat = matrix(ncol = 1,
                nrow = n_patients * observation_time)
  Xmat[, 1] <- do.call("c", lapply(1:n_patients, function(i) TDhist(observation_time,doses)))
  dim(Xmat) <- c(observation_time, n_patients)
  return(Xmat)
  }


  # Function to obtain WCE vector
wce_vector <- function(u, scenario, Xmat,normalization) {
    t <- 1:u

    scenario_function <- do.call(scenario, list((u - t) / 365))/normalization

    # Normalization (THIS SHOULD NOT BE HERE, we have to rethink the codebase and enter the scenario as a vector )
    wce <- scenario_function * Xmat[t,]

    if (u == 1) {
        res <- wce
    } else {
        res <- apply(wce, 2, sum)
    }

    return(res)
}

# Function to generate event times and censoring times
event_censor_generation <- function(max_time,n_patients) {
    # Event times : Uniform[1;365] for all scenarios
  
    eventRandom <- round(runif(n_patients, 1, max_time), 0)

    # Censoring times : Uniform[1;730] for all scenarios
    censorRandom <- round(runif(n_patients, 1, max_time*2), 0)

    return(list(eventRandom = eventRandom,
                censorRandom = censorRandom))
}

# Function for 'the final step of the permutational algorithm'
matching_algo <- function(wce_mat) {
    n_patient <- ncol(wce_mat)
    events_generation <- event_censor_generation(dim(wce_mat)[1],dim(wce_mat)[2])
    
    df_event <- data.frame(patient = 1:n_patient,
                           eventRandom = events_generation$eventRandom,
                           censorRandom = events_generation$censorRandom)
    df_event <- df_event %>%
        group_by(patient) %>%
        mutate(FUP_Ti = min(eventRandom, censorRandom)) %>%
        mutate(event = ifelse(FUP_Ti == eventRandom, 1, 0)) %>%
        ungroup() %>%
        arrange(FUP_Ti)

    # init
    patient_order <- df_event$patient
    j = 1
    id <- 1:n_patient
    wce_mat_df <- wce_mat %>% as.data.frame()
    matching_result <- data.frame()

    # Iterative matching, start with the lowest FUP
    for (i in patient_order) {
        event <- df_event[j, "event"] %>% pull()
        time_event <- df_event[j, "FUP_Ti"] %>% pull()
        
        
        if(event == 0) {
            # If no event, all probabilities are the same
            sample_id <- sample(id, 1)
        } else if(event == 1) {
            # If event, matching with different probabilities
          
            wce_matrix <- wce_mat_df %>% select(paste0("V", id)) %>% as.matrix()
            proba <- (4 * wce_matrix[time_event,]) / sum(4 * wce_matrix[time_event,])
            sample_id <- sample(id, 1, prob = proba)
        }

        matching_result <- rbind(matching_result,
                                 data.frame(id_patient = i,
                                            id_dose_wce = sample_id))
        id <- id[!id %in% sample_id]
        j = j + 1

        # Stop when last id of iterative algo
        if(length(id) == 1) {
            matching_result <- rbind(matching_result,
                                     data.frame(id_patient = patient_order[n_patient],
                                                id_dose_wce = id))
            return(list(matching_result = matching_result,
                        df_event = df_event,
                        patient_order = patient_order))
        }
    }
}

# Function to render dataset after the matching algo
get_dataset <- function(Xmat, wce_mat) {
    df_wce <- data.frame()
    Xmat_df <- Xmat %>%
        as.data.frame()
    matching_result <- matching_algo(wce_mat)

    for (i in matching_result$patient_order) {
        fu <- matching_result$df_event[matching_result$df_event$patient == i, "FUP_Ti"] %>% pull()
        event_patient <- matching_result$df_event[matching_result$df_event$patient == i, "event"] %>% pull()

        if(event_patient == 1) {
            event_vec <- c(rep(0, (fu-1)), 1)
        } else {
            event_vec <- rep(0, fu)
        }
        
        id_dose <- matching_result$matching_result[matching_result$matching_result$id_patient == i, "id_dose_wce"]
        df_dose <- data.frame(patient = rep(i, fu),
                              start = 0:(fu-1),
                              stop = 1:fu,
                              event = event_vec,
                              dose = Xmat_df[1:fu, paste0("V", id_dose)])
        df_wce <- rbind(df_wce, df_dose)
    }

    return(df_wce)
}

calcul_exposition <- function(scenario,normalization){
    expo_list <- lapply((1:180)/365, scenario)
    expo <- do.call("rbind", expo_list)/365
    return(expo/normalization)
}


simulate_dataset <- function(scenario,Xmat,normalization){
    wce_mat <- do.call("rbind", lapply(1:dim(Xmat)[1], wce_vector, scenario = scenario, Xmat = Xmat,normalization = normalization))
    df_wce <- get_dataset(Xmat = Xmat, wce_mat = wce_mat)
    return(df_wce)
}

# Function to simulate right constrained and unconstrained WCE with the same1
# dataset according to a specif@installed R Toolsic scenario
Simulate_WCE_GPU<- function(scenario, def_wce,cutoff,normalization) {
   

    # cutoff at 180 - right constrained and unconstrained with the same dataset
    wce_right_constrained <- wceGPU(df_wce, 1:3, cutoff, constrained = "right",
                                 id = "patient", event = "event", start = "start",
                                 stop = "stop", expos = "dose")

    wce_unconstrained <- wceGPU(df_wce, 1:3, cutoff, constrained = FALSE,
                             id = "patient", event = "event", start = "start",
                             stop = "stop", expos = "dose")

    WCEmat_right_constrained <- wce_right_constrained$WCEmat[which.min(wce_right_constrained$info.criterion),1:180]

    MSE_right_constrained <-  mean((calcul_exposition(scenario,normalization) - WCEmat_right_constrained )^2)



    WCEmat_unconstrained <- wce_unconstrained$WCEmat[which.min(wce_unconstrained$info.criterion),1:180]

    MSE_unconstrained <-  mean((calcul_exposition(scenario,normalization) - WCEmat_unconstrained )^2)
    
    results_right_constrained = list(WCEmat = WCEmat_right_constrained,
                                     BIC = min(wce_right_constrained$info.criterion),
                                     MSE = MSE_right_constrained)
    
    results_unconstrained = list(WCEmat = WCEmat_unconstrained,
                                 BIC = min(wce_unconstrained$info.criterion),
                                 MSE = MSE_unconstrained)
        
    # Best result according to BIC
    return(list(results_right_constrained = results_right_constrained,
                results_unconstrained = results_unconstrained))
}

simulate_with_bootstraps <- function(n_bootstraps, number_patients, observation_time, scenario,cutoff,normalization, Xmat){


    # Flag !!! to change because my understanding of bootstraps was terrible 
    print("################ Start simulation #################")
  
    start_time = Sys.time()
    
    print(paste("simulation of ",n_bootstraps,"bootstraps"))
    
    
    right_constrained <- list()
    right_constrained_BIC <- list()
    right_constrained_MSE <- list()
    unconstrained <- list()
    unconstrained_BIC <-  list()
    unconstrained_MSE <-  list()


    for (i in 1:n_bootstraps){

        print(paste("Simulation of bootstrap",i))
    
        #Xmat <- generate_Xmat(observation_time,number_patients)
        Xmat <- Xmat_list[i]
        simulation <- Simulate_WCE(scenario,Xmat,cutoff,normalization)
        right_constrained <- append(right_constrained,list(simulation$results_right_constrained$WCEmat))
        right_constrained_BIC <- append(right_constrained_BIC,list(simulation$results_right_constrained$BIC))
        right_constrained_MSE <- append(right_constrained_MSE,list(simulation$results_right_constrained$MSE))
        unconstrained <- append(unconstrained,list(simulation$results_unconstrained$WCEmat))
        unconstrained_BIC <- append(unconstrained_BIC,list(simulation$results_unconstrained$BIC))   
        unconstrained_MSE <- append(unconstrained_MSE,list(simulation$results_unconstrained$MSE))

    

    }    


    list_simulations <- list(right_constrained=right_constrained,
                             right_constrained_BIC = right_constrained_BIC,
                             right_constrained_MSE = right_constrained_MSE,
                             unconstrained = unconstrainematchingd,
                             unconstrained_BIC = unconstrained_BIC,
                             unconstrained_MSE = unconstrained_MSE)
    end_time = Sys.time()
    time = difftime(end_time, start_time, units = "sec")
    print(paste("This simulation took : ", round(time), " seconds"))

    return(list_simulations)
}

normalize_function <- function(scenario, sum, upper_time){
  integration <- integrate(scenario, lower = 0, upper = upper_time)$value
  return(integration)
}