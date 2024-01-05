library(dplyr)
library(WCE)
library(purrr)

doses = c(1,1.5,2,2.5,3)
print("toto")
print("???")

TDhist <- function(observation_time,doses) {
  # Duration : lognormal distribution(0.5,0.8)
  duration <- 7 + 7 * round(rlnorm(1, meanlog = 0.5, sdlog = 0.8), 0)
  # in weeks

  # Dose : random assignment of values 0.5, 1, 1.5, 2, 2.5 and 3
  #dose <- sample(seq(from = 0.5, to = 3, by = 0.5), size = 1)

  # Start with drug exposure
  vec <- rep(doses, duration)

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

generate_Xmat_list<- function(observation_time,n_patients,n_bootstraps,doses){

    print("start test")
    Xmat_list = list()
    for(i in 1:n_bootstraps){
        print(i)
        Xmat <- generate_Xmat(observation_time,n_patients,doses)
        Xmat_list <- append(Xmat_list, Xmat)
        print(Xmat_list)
    }

    return(Xmat_list)

}


Xmat_list <- generate_Xmat_list(365,500,10,doses)
Xmat_list