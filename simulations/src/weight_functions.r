library(pracma)



# Scenario 1 - Exponential
exponential_weight <- function(u_t) {
  return(7 * exp(-7 * u_t))
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