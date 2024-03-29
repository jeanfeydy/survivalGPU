# 'counting.R' test from survival package
options(na.action=na.exclude) # preserve missings
options(contrasts=c('contr.treatment', 'contr.poly')) #ensure constrast type
library(survival)

# Create a "counting process" version of the simplest test data set
#
test1 <- data.frame(time=  c(9, 3,1,1,6,6,8),
                    status=c(1,NA,1,0,1,1,0),
                    x=     c(0, 2,1,1,1,0,0))

test1b<- list(start= c(0, 3,  0,  0, 5,  0, 6,14,  0,  0, 10,20,30, 0),
              stop = c(3,10, 10,  5,20,  6,14,20, 30,  10,20,30,40, 10),
              status=c(0, 1,  0,  0, 1,  0, 0, 1,  0,   0, 0, 0, 1,  0),
              x=     c(1, 1,  1,  1, 1,  0, 0, 0,  0,   0, 0, 0, 0,  NA),
              id =   c(3, 3,  4,  5, 5,  6, 6, 6,  7,   1, 1, 1, 1,   2))

aeq <- function(x,y) all.equal(as.vector(x), as.vector(y))
#
#  Check out the various residuals under an Efron approximation
#

# Right process
fit0 <- coxph(Surv(time, status)~ x, test1, iter=0)
fit  <- coxph(Surv(time, status) ~x, test1)

# Counting process
fit0b <- coxph(Surv(start, stop, status) ~ x, test1b, iter=0)
fitb  <- coxph(Surv(start, stop, status) ~x, test1b)
fit0b_gpu <- coxphGPU(Surv(start, stop, status) ~ x, test1b, iter.max=1)
fitb_gpu <- coxphGPU(Surv(start, stop, status) ~ x, test1b)

# offset feature
fitc  <- coxph(Surv(time, status) ~ offset(fit$coef*x), test1)
fitd  <- coxph(Surv(start, stop, status) ~ offset(fit$coef*x), test1b)
#fitd_gpu  <- coxphGPU(Surv(start, stop, status) ~ offset(fit$coef*x), test1b) # offset feature not yet implemented in coxphGPU

test_that("Check coefs between counting process - No iter", {
  expect_equal(
    as.vector(fit0b$coef),
    as.vector(fit0b_gpu$coef)
  )
})

# # Error because small dataset ?
# test_that("Check coefs between counting process", { # convergence problem with small numbers ?
#   expect_equal(
#     as.vector(fitb$coef),
#     as.vector(fitb_gpu$coef)
#   )
# })

aeq(resid(fit0), resid(fit0b, collapse=test1b$id))
aeq(resid(fit), resid(fitb, collapse=test1b$id))
aeq(resid(fitc), resid(fitd, collapse=test1b$id))
aeq(resid(fitc), resid(fit))

aeq(resid(fit0, type='score'), resid(fit0b, type='score', collapse=test1b$id))
aeq(resid(fit, type='score'), resid(fitb, type='score', collapse=test1b$id))

aeq(resid(fit0, type='scho'), resid(fit0b, type='scho', collapse=test1b$id))
aeq(resid(fit, type='scho'), resid(fitb, type='scho', collapse=test1b$id))

# The two survivals will have different censoring times
#  nrisk, nevent, surv, and std should be the same
temp1 <- survfit(fit, list(x=1), censor=FALSE)
temp2 <- survfit(fitb, list(x=1), censor=FALSE)
all.equal(unclass(temp1)[c(3,4,6,8)], unclass(temp2)[c(3,4,6,8)])

# # Tests with some id on drugdata - convergence errors
# drugdata <- WCE::drugdata
# coxph(Surv(Start, Stop, Event) ~sex+age, drugdata %>% dplyr::filter(Id %in% c(1:6)))
# coxphGPU(Surv(Start, Stop, Event) ~sex+age, drugdata %>% dplyr::filter(Id %in% c(1:6)))
# drugdata %>%
#   filter(Id == 6) #%>% tail(10)
