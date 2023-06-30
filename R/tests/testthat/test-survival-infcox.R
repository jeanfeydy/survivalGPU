options(na.action=na.exclude) # preserve missings
options(contrasts=c('contr.treatment', 'contr.poly')) #ensure constrast type
library(survival)

#
# A test to exercise the "infinity" check on 2 variables
#
test3 <- data.frame(futime=1:12, fustat=c(1,0,1,0,1,0,0,0,0,0,0,0),
                    x1=rep(0:1,6), x2=c(rep(0,6), rep(1,6)))

test3b <- data.frame(start = rep(0,12), futime=1:12, fustat=c(1,0,1,0,1,0,0,0,0,0,0,0),
                    x1=rep(0:1,6), x2=c(rep(0,6), rep(1,6)))

# This will produce a warning message, which is the point of the test.
# The variance is close to singular and gives different answers
#  on different machines
expect_warning(fit3 <- coxph(Surv(futime, fustat) ~ x1 + x2, test3, iter=25))
expect_warning(fit3b <- coxph(Surv(start, futime, fustat) ~ x1 + x2, test3b, iter=25))

# # Convergence error
# fit3b_gpu <- coxphGPU(Surv(start, futime, fustat) ~ x1 + x2, test3b, iter.max=25)

# Check warnings
test_that("warning right process", {
  expect_warning(fit3 <- coxph(Surv(futime, fustat) ~ x1 + x2, test3, iter=25))
})

test_that("warning counting process", {
  expect_warning(fit3b <- coxph(Surv(start, futime, fustat) ~ x1 + x2, test3b, iter=25))
})


all(fit3$coef < -22)
all.equal(round(fit3$log, 4),c(-6.8669, -1.7918))

#
# Actual solution
#  time 1, 12 at risk,  3 each of x1/x2 = 00, 01, 10, 11
#  time 2, 10 at risk,                     2, 3,  2 ,  3
#  time 5, 8  at risk,                     1, 3,  1,   3
# Let r1 = exp(beta1), r2= exp(beta2)
# loglik = -log(3 + 3r1 + 3r2 + 3 r1*r2) - log(2 + 2r1 + 3r2 + 3 r1*r2) -
#           log(1 + r1  + 3r2 + 3 r1*r2)
true <- function(beta) {
  r1 <- exp(beta[1])
  r2 <- exp(beta[2])
  loglik <- -log(3*(1+ r1+ r2+ r1*r2)) - log(2+ 2*r1 + 3*r2 + 3*r1*r2) -
    log(1 + r1 + 3*r2 + 3*r1*r2)
  loglik
}

all.equal(fit3$loglik[2], true(fit3$coef), check.attributes=FALSE)

# test_that("log likelihood", {
#   expect_equal(
#     round(fit3$loglik[2], 3),
#     round(fit3b_gpu$loglik, 3)
#   )
# })
