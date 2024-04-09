library(dplyr)
# source("src/weight_functions.r")

# library(WCE)
# library(purrr)

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
wce_vector <- function(u, scenario, Xmat,normalization_factor) {
    t <- 1:u




    # scenario_2 = exponential_weight
    # print(u)
    scenario_shape <- do.call(scenario, list((u - t) / 365))/365*normalization_factor

    # t = 1:365
    # u = 365
    # scenario_shape_total <- do.call(scenario, list((u - t) / 365))/365*normalization_factor

    # print(sum(scenario_shape_total))
    # quit()
    


    
    # print(sum(scenario_shape))

    wce <- scenario_shape[1:u] * Xmat[t,]

    if (u == 1) {
        res <- wce
    } else {
        res <- apply(wce, 2, sum)
    }


    return(res)
}

calculate_normalization_factor <- function(scenario, HR_target,observation_time){
    t<- 0:365


    normalization_target <- log(HR_target)


    t<- 0:365
    scenario_function <- do.call(scenario, list((365 - t) / 365))
    scenario_sum <- sum(scenario_function)/365


    

    if (scenario_sum ==0 ){
        normalization_factor <- 1
    }else{
        normalization_factor <- normalization_target/scenario_sum
    }

 

    return(normalization_factor)
}

calcul_exposition <- function(scenario,HR_target,cutoff){

    expo_list <- lapply((1:cutoff)/365, scenario)
    expo <- do.call("rbind", expo_list)/365

    normalization_factor <- calculate_normalization_factor(scenario,HR_target,cutoff)


    return(expo*normalization_factor)
}


#### TEST
# Xmat = 
# wce_vector(180,"exponential_weight","Xmat",1,180)

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
matching_algo <- function(wce_mat,events_generation) {
    n_patient <- ncol(wce_mat)
    
    
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

        first <- TRUE
        
   
        if(event == 0) {
            # If no event, all probabilities are the same
            sample_id <- sample(id, 1)
        } else if(event == 1) {
            # If event, matching with different probabilities
          
            wce_matrix <- wce_mat_df %>% select(paste0("V", id)) %>% as.matrix()
            if (4 * wce_matrix[time_event,] == 0){
                sample_id <- sample(id, 1)
            }else{


            proba <- (exp(wce_matrix[time_event,])) / sum(exp(wce_matrix[time_event,]))
            # proba <- (wce_matrix[time_event,]) / sum(wce_matrix[time_event,])

            sample_id <- sample(id, 1, prob = proba)

            
            # wce_matrix[time_event,])))


            }            
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
get_dataset <- function(Xmat, wce_mat,matching_result) {
    df_wce <- data.frame()
    Xmat_df <- Xmat %>%
        as.data.frame()


    

    


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
    df_wce <- df_wce[order(df_wce$patient),]
    return(df_wce)
}

binarization_dose_function <- function(doses){
    binary_doses = c()
    for(dose in doses){
        binary_dose <- dose
        if (dose > 1){
         
            binary_dose = 1
        }
        binary_doses <- append(binary_doses,binary_dose)
    }
    return(binary_doses)
}

genrate_dataset_process = function(doses,observation_time, n_patients,scenario,cutoff,HR_target){
    Xmat <- generate_Xmat(observation_time,n_patients,doses)
    scenario_function <- scenario_translator(scenario)
    normalization_factor <- calculate_normalization_factor(scenario_function,HR_target,365)
    wce_mat <- do.call("rbind", lapply(1:observation_time, wce_vector, scenario = scenario_function, Xmat = Xmat,normalization_factor = normalization_factor)) 
    events_generation <- event_censor_generation(dim(wce_mat)[1],dim(wce_mat)[2])
    matching_result <- matching_algo(wce_mat,events_generation)
    df_wce <- get_dataset(Xmat = Xmat, wce_mat = wce_mat_batch,matching_result = matching_result)
    return(df_wce)

}


# A function that generate a dataset with the batch method (if necessary)
generate_dataset_batch <- function(doses,observation_time, n_patients,scenario,cutoff,HR_target,batchsize){ 
    
    # Generation of the dose amtrix for the number of desired patients
    Xmat <- generate_Xmat(observation_time,n_patients,doses)

    scenario_time_list <- list()
    
    # Translating the wieght function form the scenario as a string
    scenario_function <- scenario_translator(scenario)


    # TODO Will need to change the nomraliszation factor calcul in order to take into account the 
    normalization_factor <- calculate_normalization_factor(scenario_function,HR_target,observation_time)



    # Generation of the WCEmat that calculate the WCE weight of the patient at every time        
    wce_mat <- do.call("rbind", lapply(1:observation_time, wce_vector, scenario = scenario_function, Xmat = Xmat,normalization_factor = normalization_factor)) 

    # Look if HR is 1
    if(HR_target == 1){
        is_null_weight <- TRUE
    }else{
        is_null_weight <- FALSE
    }

    # simple generation of the data if n_patients <= BATCHSIZE
    if (n_patients <= batchsize){
        df_wce <- get_dataset(Xmat = Xmat, wce_mat = wce_mat,is_null_weight)
    }
    # Batch generation of the data if n_patients <= BATCHSIZE
    else{
        current_n <- 1
        print(current_n)
        max_n <- current_n + batchsize -1
        wce_mat_batch <- wce_mat[,current_n:max_n]
        Xmat_batch <- Xmat[,current_n:max_n]
        current_n <- current_n + batchsize
        df_wce_batch <- get_dataset(Xmat = Xmat_batch, wce_mat = wce_mat_batch,is_null_weight)
        df_wce_batch <- df_wce_batch[order(df_wce_batch$patient),]
        df_wce <- df_wce_batch

        while(current_n < n_patients){

               max_n <- min(current_n + batchsize -1,n_patients)
               wce_mat_batch <- wce_mat[,current_n:max_n]
               Xmat_batch <- Xmat[,current_n:max_n]
               df_wce_batch <- get_dataset(Xmat = Xmat_batch, wce_mat = wce_mat_batch,is_null_weight)
               df_wce_batch <- df_wce_batch[order(df_wce_batch$patient),]
               df_wce_batch["patient"] <- df_wce_batch["patient"] + current_n -1
               df_wce <- rbind(df_wce,df_wce_batch)
               current_n <- current_n + batchsize
           }
    }    

}

