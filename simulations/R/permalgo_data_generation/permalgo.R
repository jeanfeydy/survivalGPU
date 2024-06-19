library(PermAlgo)
library(survival)


n_patients = 1000   
max_time = 365




event_censor_generation <- function(max_time,n_patients) {
    # Event times : Uniform[1;365] for all scenarios
  
    eventRandom <- round(runif(n_patients, 1, max_time), 0)

    # Censoring times : Uniform[1;730] for all scenarios
    censorRandom <- round(runif(n_patients, 1, max_time*2), 0)

    return(list(eventRandom = eventRandom,
                censorRandom = censorRandom))
}


event_censor = event_censor_generation(max_time,n_patients)

eventRandom = event_censor$eventRandom
censorRandom = event_censor$censorRandom



ConstantCovariate <- function(name, values, weights, beta) {
    covariate <- list(
        name = name,
        values = values,
        weights = weights,
        beta = beta,
        generate_Xmat = function(max_time, n_patients) {
            proba <- weights / sum(weights)
            Xvect <- sample(values, size = n_patients, replace = TRUE, prob = proba)
            Xmat <- matrix(rep(Xvect, max_time), nrow = max_time, ncol = n_patients, byrow = TRUE)
            return(Xmat)
        }
    )
    return(covariate)
}

# Example usage

values <- c(0, 1)
weights <- c(1, 2)

beta <- 1
covariate <- ConstantCovariate("Covariate1", values, weights, beta)
Xmat <- covariate$generate_Xmat(365, 1000)
Xmat <- matrix(Xmat, nrow = max_time * n_patients, ncol = 1)




data <- permalgorithm(n_patients, max_time, Xmat, XmatNames=c("drug_1"),
eventRandom = eventRandom, censorRandom=censorRandom, betas=c(log(2)), groupByD=FALSE )


# Save data as CSV
write.csv(data, file = "data_1.csv", row.names = FALSE)



print(head(data))
print(summary(data))


model = coxph(Surv(Start,Stop,Event) ~ drug_1, data = data) 
print(summary(model))




covariate_1 <- ConstantCovariate("Covariate1", c(0,1), c(1,1), 0.7)
covariate_2 <- ConstantCovariate("Covariate2", c(0,1), c(1,1), 0.5)
# covariate_3 <- ConstantCovariate("Covariate2", values, weights, beta)


eventRandom = event_censor$eventRandom
censorRandom = event_censor$censorRandom


Xmat_1 <- covariate$generate_Xmat(365, 1000)
Xmat_2 <- covariate$generate_Xmat(365, 1000)


Xmat_1_line <- matrix(Xmat_1, nrow = max_time * n_patients, ncol = 1)
Xmat_2_line <- matrix(Xmat_2, nrow = max_time * n_patients, ncol = 1)

Xmat = cbind(Xmat_1_line, Xmat_2_line)


data <- permalgorithm(n_patients, max_time, Xmat, XmatNames=c("drug_1","drug_2"),
eventRandom = eventRandom, censorRandom=censorRandom, betas=c(log(1.5),log(1)), groupByD=FALSE )


model = coxph(Surv(Start,Stop,Event) ~ drug_1 + drug_2, data = data) 


print(summary(model))

# Save data as CSV
write.csv(data, file = "data_2.csv", row.names = FALSE)


