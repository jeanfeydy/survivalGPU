# Scénario 1 - Exponential
exponential_weight <- function(u_t) {
  return(7 * exp(-7 * u_t))
}

# Scénario 2 - Bi-Linearexponential_weight <- function(u_t) {

bi_linear_weight <- function(u_t) {
  ifelse((u_t) <= (50 / 365),
    1 - ((u_t) / (50 / 365)),
    0
  )
}

## Scénario 3 - Early peak

early_peak_weight <- function(u_t) {
  return(dnorm(u_t, 0.04, 0.05))
}

## Scénario 4 - Inverted U
inverted_u_weight <- function(u_t) {
  return(dnorm(u_t, 0.2, 0.06))
}

## Scénario 5 - Constant
constant_weight <- function(u_t) {
  ifelse(u_t >= 0 & u_t <= 180 / 365,
    1 / 180,
    0
  )
}

late_effect_weight <- function(u_t) {
  return(dnorm(u_t, 0.35, 0.04))
}
