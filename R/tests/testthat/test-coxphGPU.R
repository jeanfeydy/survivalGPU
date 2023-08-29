# Dataset
drugdata <- WCE::drugdata

# drugdata2 is drugdata with the last observation for each Id
drugdata2 <- drugdata |>
  dplyr::arrange(Stop |> dplyr::desc()) |>
  dplyr::distinct(Id, .keep_all = TRUE) |>
  dplyr::arrange(Id)

# TODO: remove this when the re-implementation of coxph is over
# We currently do not support Efron ties, only Breslow
ties <- "breslow"

## Original Coxph model ------
library(survival)

# Surv type counting
coxph <- coxph(
  Surv(Start, Stop, Event) ~ sex + age,
  drugdata,
  ties = ties
)

# Surv type right
coxph_right <- coxph(
  Surv(Stop, Event) ~ sex + age,
  drugdata,
  ties = ties
)


## CoxphGPU model ------

# Counting
coxphGPU <- coxphGPU(Surv(Start, Stop, Event) ~ sex + age,
  drugdata,
  ties = ties,
  bootstrap = 1
)

coxphGPU_bootstrap <- coxphGPU(Surv(Start, Stop, Event) ~ sex + age,
                               drugdata,
                               ties = ties,
                               bootstrap = 15
)

# Right
coxphGPU_right <- coxphGPU(Surv(Stop, Event) ~ sex + age,
                           drugdata,
                           ties = ties,
                           bootstrap = 1
)


# Tests

## Comparison tests between coxphGPU with and without bootstrap
test_that("coxphGPU with and without bootstrap - Coefs", {
  expect_equal(
    round(as.numeric(coxphGPU$coefficients), 5),
    round(as.numeric(coxphGPU_bootstrap$coefficients), 5)
  )
})

test_that("coxphGPU with and without bootstrap - Covar matrix", {
  expect_equal(
    round(as.numeric(coxphGPU$var), 5),
    round(as.numeric(coxphGPU_bootstrap$var), 5)
  )
})

## Comparison tests between coxph and coxphGPU (counting and right Surv type)
test_that("Coxph counting - Coefs", {
  expect_equal(
    round(as.numeric(coxph$coefficients), 5),
    round(as.numeric(coxphGPU_bootstrap$coefficients), 5)
  )
})

test_that("Coxph right - Coefs", {
  expect_equal(
    round(as.numeric(coxph_right$coefficients), 5),
    round(as.numeric(coxphGPU_right$coefficients), 5)
  )
})

test_that("Coxph counting - Covar matrix", {
  expect_equal(
    round(coxph$var, 7),
    round(coxphGPU_bootstrap$var, 7)
  )
})

test_that("Coxph right - Covar matrix", {
  expect_equal(
    round(coxph_right$var, 7),
    round(coxphGPU_right$var, 7)
  )
})

test_that("Coxph counting - log likelihood", {
  expect_equal(
    round(coxph$loglik[2], 2),
    round(coxphGPU_bootstrap$loglik, 2)
  )
})

test_that("Coxph right - log likelihood", {
  expect_equal(
    round(coxph_right$loglik[2], 2),
    round(coxphGPU_right$loglik, 2)
  )
})

# same lp ? (because not the same colMeans)
test_that("Coxph counting - linears predictors", {
  expect_equal(
    round(coxph$linear.predictors, 3),
    c(round(coxphGPU_bootstrap$linear.predictors, 3))
  )
})

# test_that("Coxph right - linears predictors", {
#   expect_equal(
#     round(coxph_right$linear.predictors, 3),
#     c(round(coxphGPU_right$linear.predictors, 3))
#   )
# })

test_that("Coxph counting - residuals", {
  expect_equal(
    round(coxph$residuals, 3),
    round(coxphGPU_bootstrap$residuals, 3)
  )
})

test_that("Coxph counting - resid method", {
  expect_equal(
    round(resid(coxph), 3),
    round(resid(coxphGPU_bootstrap), 3)
  )
})

test_that("Coxph counting - resid method score type", {
  expect_equal(
    round(resid(coxph, type = "score"), 2),
    round(resid(coxphGPU_bootstrap, type = "score"), 2)
  )
})

test_that("Coxph counting - resid method schoenfeld type", {
  expect_equal(
    round(resid(coxph, type = "schoenfeld"), 2),
    round(resid(coxphGPU_bootstrap, type = "schoenfeld"), 2)
  )
})

test_that("Coxph counting - predict method survival type", {
  expect_equal(
    round(predict(coxph, type = "survival"), 3),
    round(predict(coxphGPU, type = "survival"), 3)
  )
})

test_that("Coxph counting - predict method lp type", {
  expect_equal(
    round(predict(coxph, type = "lp"), 3),
    round(predict(coxphGPU, type = "lp"), 3)
  )
})

test_that("Coxph counting - predict method lp type - linears.predictors check", {
  expect_equal(
    round(coxphGPU$linear.predictors, 5),
    round(predict(coxphGPU, type = "lp"), 5)
  )
})

test_that("Coxph counting - predict method risk type", {
  expect_equal(
    round(predict(coxph, type = "risk"), 3),
    round(predict(coxphGPU, type = "risk"), 3)
  )
})

test_that("Coxph counting - predict method expected type", {
  expect_equal(
    round(predict(coxph, type = "expected"), 3),
    round(predict(coxphGPU, type = "expected"), 3)
  )
})

# test_that("Coxph counting - predict method terms type", {
#   expect_equal(
#     round(predict(coxph, type = "terms"), 3),
#     round(predict(coxphGPU, type = "terms"), 3)
#   )
# })

test_that("Coxph right - residuals", {
  expect_equal(
    round(coxph_right$residuals, 2),
    round(coxphGPU_right$residuals, 2)
  )
})

test_that("Coxph right - resid method schoenfeld type", {
  expect_equal(
    round(resid(coxph_right, type = "schoenfeld"), 2),
    round(resid(coxphGPU_right, type = "schoenfeld"), 2)
  )
})


test_that("Coxph right - predict method survival type", {
  expect_equal(
    round(predict(coxph_right, type = "survival"), 3),
    round(predict(coxphGPU_right, type = "survival"), 3)
  )
})

# test_that("Coxph right - predict method risk type", {
#   expect_equal(
#     round(predict(coxph_right, type = "risk"), 3),
#     round(predict(coxphGPU_right, type = "risk"), 3)
#   )
# })



################################################################################

# Cox model with no iterations
coxphGPU_no_iter <- coxphGPU(Surv(Start, Stop, Event) ~ sex + age,
                             drugdata,
                             ties = ties,
                             iter.max = 0)

test_that("No Newton iterations - Null Coefs", {
  expect_equal(
    as.vector(coxphGPU_no_iter$coefficients),
    c(0,0)
  )
})


coxphGPU_right_drugdata2 <- coxphGPU(Surv(Stop, Event) ~ sex + age,
                                     drugdata2,
                                     ties = ties,
                                     bootstrap = 1)

test_that("CoxphGPU counting/right distinct - Coefs", {
  expect_equal(
    round(coxphGPU$coefficients, 3),
    round(coxphGPU_right_drugdata2$coefficients, 3)
  )
})

test_that("CoxphGPU counting/right distinct - Covar matrix", {
  expect_equal(
    round(coxphGPU$var, 3),
    round(coxphGPU_right_drugdata2$var, 3)
  )
})

################################################################################

# Test strata
coxph_strata <- coxph(Surv(Start, Stop, Event) ~ strata(sex) + age,
                      drugdata,
                      ties = ties)

coxphGPU_strata <- coxphGPU(Surv(Start, Stop, Event) ~ strata(sex) + age,
                            drugdata,
                            ties = ties)

coxph_right_strata <- coxph(Surv(Stop, Event) ~ strata(sex) + age,
                            drugdata,
                            ties = ties)

# coxphGPU_right_strata <- coxphGPU(Surv(Stop, Event) ~ strata(sex) + age,
#                                   drugdata,
#                                   ties = ties)


test_that("CoxphGPU counting with strata - Coefs", {
  expect_equal(
    round(coxph_strata$coefficients, 3),
    round(coxphGPU_strata$coefficients, 3)
  )
})

# test_that("CoxphGPU right with strata - Coefs", {
#   expect_equal(
#     round(coxph_right_strata$coefficients, 3),
#     round(coxphGPU_right_strata$coefficients, 3)
#   )
# })


test_that("CoxphGPU counting with strata - predict", {
  expect_equal(
    round(predict(coxph_strata, type = "survival"), 3),
    round(predict(coxphGPU_strata, type = "survival"), 3)
  )
})

test_that("CoxphGPU counting with strata - new data", {
  expect_equal(
    round(predict(coxph_strata, newdata = head(drugdata)), 3),
    round(predict(coxphGPU_strata, newdata = head(drugdata)), 3)
  )
})

################################################################################

# snapshot

# test_that("CoxPH counting - Breslow", {
#   expect_snapshot({
#     coxphGPU(Surv(Start, Stop, Event) ~ sex + age,
#              data = drugdata,
#              ties = "breslow")
#   })
#   expect_snapshot({
#     coxphGPU(Surv(Start, Stop, Event) ~ sex,
#              data = drugdata,
#              ties = "breslow")
#   })
# })
#
# test_that("CoxPH right - Breslow", {
#   expect_snapshot({
#     coxphGPU(Surv(Stop, Event) ~ sex + age,
#              data = drugdata,
#              ties = "breslow")
#   })
#   expect_snapshot({
#     coxphGPU(Surv(Stop, Event) ~ sex,
#              data = drugdata,
#              ties = "breslow")
#   })
# })

################################################################################

# Software validation vignette of Terry Therneau

test1 <- data.frame(time = c(1,1,6,6,8,9),
                    status = c(1,0,1,1,0,1),
                    x = c(1,1,1,0,0,0))

temp <- temp2 <- matrix(0, nrow = 6, ncol = 4,
                        dimnames = list(1:6, c("iter", "beta", "loglik", "H")))

# routine précise ? car bug aléatoire
convergence_warning <- capture_warnings(
  for (i in 0:5) {
    # coxph (add expect_warning for devtools::test())
    tfit <- coxph(Surv(time, status) ~ x,
                  data = test1,
                  ties = "breslow",
                  iter.max = i)
    # coxphGPU
    tfit2 <- coxphGPU(Surv(time, status) ~ x,
                      data = test1,
                      ties = "breslow",
                      iter.max = i)

    temp[i+1,] <- c(tfit$iter, coef(tfit), tfit$loglik[2], 1/vcov(tfit))
    temp2[i+1,] <- c(tfit2$iter, coef(tfit2), tfit2$loglik, 1/tfit2$var)
  }
)

test_that("Convergence warnings ?", {
  expect_match(convergence_warning,
               "Ran out of iterations and did not converge",
               all = TRUE)
})

test_that("test1 - beta", {
  expect_equal(
    round(temp[,"beta"], 3),
    round(temp2[,"beta"], 3)
  )
})

test_that("test1 - loglik", {
  expect_equal(
    round(temp[,"loglik"], 3),
    round(temp2[,"loglik"], 3)
  )
})

test_that("test1 - H", {
  expect_equal(
    round(temp[,"H"], 3),
    round(temp2[,"H"], 3)
  )
})



# check example ?predict.coxph

lung2 <- lung |>
  dplyr::mutate(status = status - 1) |>
  tidyr::drop_na()
fit <- coxph(Surv(time, status) ~ age + ph.ecog + strata(inst), lung2)
# fit_gpu <- coxphGPU(Surv(time, status) ~ age + ph.ecog + strata(inst), lung2,bootstrap = 1) # error avec strata

# #lung data set has status coded as 1/2
# mresid <- (lung$status-1) - predict(fit, type='expected') #Martingale resid
# predict(fit,type="lp")
# predict(fit,type="expected")
# predict(fit,type="risk",se.fit=TRUE)
# predict(fit,type="terms",se.fit=TRUE)

