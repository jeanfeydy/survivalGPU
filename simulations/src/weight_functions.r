library(pracma)

# Scenario 1 - Exponential
exponential_weight <- function(u_t) {
  return(7 * exp(-7 * u_t))
}


# Scenario 1 - Exponential
exponential_weight_2 <- function(u_t) {
  return((7 * exp(-7 * u_t))*0.5)
}

# Scenario 2 - Bi-Linear 
bi_linear_weight <- function(u_t) {
  ifelse((u_t) <= (50 / 365),
    1 - ((u_t) / (50 / 365)),
    0
  )
}

## Scenario 3 - Early peak

early_peak_weight <- function(u_t) {
  return(dnorm(u_t, 0.04, 0.05))
}

## Scenario 4 - Inverted U
inverted_u_weight <- function(u_t) {
  return(dnorm(u_t, 0.2, 0.06))
}

## Scenario 5 - Constant
constant_weight <- function(u_t) {
  ifelse(u_t >= 0 & u_t <= 180 / 365,
    1 / 180,
    0
  )
}

## Scenario 6 - late effect
late_effect_weight <- function(u_t) {
  return(dnorm(u_t, 0.35, 0.04))
}

## Scenario 7 - Constant
null_weight <- function(u_t) {
  return(0.00000000000000000000000000000)
}


scenario_translator <- function(scenario){
  if(scenario == "exponential_weight"){
    return(exponential_weight)
  }
  if(scenario == "bi_linear_weight"){
    return(bi_linear_weight)
  }
  if(scenario == "exponential_weight"){
    return(exponential_weight)
  }
  if(scenario == "early_peak_weight"){
    return(early_peak_weight)
  }
  if(scenario == "inverted_u_weight"){
    return(inverted_u_weight)
  }
  if(scenario == "constant_weight"){
    return(constant_weight)
  }  
  if(scenario == "late_effect_weight"){
    return(late_effect_weight)
  }
  if(scenario == "exponential_weight_2"){
    return(exponential_weight_2)
  }
  if(scenario == "null_weight"){
    return(null_weight)
  }
}