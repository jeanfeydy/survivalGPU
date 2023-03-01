# Dataset
drugdata <- WCE::drugdata
# library(survival)

# Original Coxph model
coxph <- coxph(
  Surv(Start, Stop, Event) ~ sex + age,
  drugdata
)

# CoxphGPU
coxphGPU <- coxphGPU(Surv(Start, Stop, Event) ~ sex + age,
  drugdata,
  bootstrap = 1
)

coxphGPU_bootstrap <- coxphGPU(Surv(Start, Stop, Event) ~ sex + age,
  drugdata,
  bootstrap = 15
)

coxphGPU_bootstrap_all_result <- coxphGPU(Surv(Start, Stop, Event) ~ sex + age,
  drugdata,
  bootstrap = 15,
  all.results = TRUE
)

# Tests
test_that("Coxph Coefs", {
  expect_equal(
    round(as.numeric(coxph$coefficients), 5),
    round(as.numeric(coxphGPU_bootstrap$coefficients), 5)
  )
})

test_that("Covar matrix", {
  expect_equal(
    round(coxph$var, 7),
    round(coxphGPU_bootstrap$var[[1]], 7)
  )
})

test_that("log likelihood", {
  expect_equal(
    round(coxph$loglik[2], 3),
    round(coxphGPU_bootstrap$loglik, 3)
  )
})

# same lp ? (because not the same colMeans)
test_that("linears predictors", {
  expect_equal(
    round(coxph$linear.predictors, 3),
    c(round(coxphGPU_bootstrap$linear.predictors, 3))
  )
})

test_that("residuals", {
  expect_equal(
    round(coxph$residuals, 3),
    round(coxphGPU_bootstrap$residuals, 3)
  )
})


# # Right Surv on coxph with drugdata
# drugdata2 <- drugdata %>%
#   dplyr::arrange(Stop %>% dplyr::desc()) %>%
#   dplyr::distinct(Id, .keep_all = TRUE) %>%
#   dplyr::arrange(Id)
#
# # same model but not the same lp, residuals, ...
# counting_coxph<-coxph(Surv(Stop, Event) ~ sex + age,
#       drugdata)
#
# right_coxph<-coxph(Surv(Stop, Event) ~ sex + age,
#                    drugdata2)


# snapshot
test_that("CoxPH counting", {
  expect_snapshot({
    coxphGPU(Surv(Start,Stop, Event) ~ sex + age,
             data = drugdata)
  })
  expect_snapshot({
    coxphGPU(Surv(Start,Stop, Event) ~ sex,
             data = drugdata)
  })
})
