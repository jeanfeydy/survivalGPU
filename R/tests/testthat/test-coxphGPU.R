# Dataset
drugdata <- WCE::drugdata
# library(survival)

# Original Coxph model
coxph <- coxph(
  Surv(Start, Stop, Event) ~ sex + age,
  drugdata,
  ties = "efron"
)

# CoxphGPU
coxphGPU <- coxphGPU(Surv(Start, Stop, Event) ~ sex + age,
  drugdata,
  ties = "efron",
  bootstrap = 1
)

coxphGPU_bootstrap <- coxphGPU(Surv(Start, Stop, Event) ~ sex + age,
  drugdata,
  ties = "efron",
  bootstrap = 15
)

coxphGPU_bootstrap_all_result <- coxphGPU(Surv(Start, Stop, Event) ~ sex + age,
  drugdata,
  ties = "efron",
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


# Test with Breslow method
coxph_breslow <- coxph(
  Surv(Start, Stop, Event) ~ sex + age,
  drugdata,
  ties = "breslow"
)

coxphGPU_breslow <- coxphGPU(Surv(Start, Stop, Event) ~ sex + age,
                             drugdata,
                             ties = "breslow",
                             bootstrap = 1
)

# Tests
test_that("Coxph Coefs - Breslow", {
  expect_equal(
    round(as.numeric(coxph_breslow$coefficients), 5),
    round(as.numeric(coxphGPU_breslow$coefficients), 5)
  )
})


coxphGPU_no_iter <- coxphGPU(Surv(Start, Stop, Event) ~ sex + age,
                             drugdata,
                             ties = "efron",
                             bootstrap = 1,
                             iter.max = 1
)

test_that("No Newton iterations - Null Coefs", {
  expect_equal(
    as.vector(coxphGPU_no_iter$coefficients),
    c(0,0)
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

# Actually, no Efron method
# test_that("CoxPH counting - Efron", {
#   expect_snapshot({
#     coxphGPU(Surv(Start,Stop, Event) ~ sex + age,
#              data = drugdata,
#              ties = "efron")
#   })
#   expect_snapshot({
#     coxphGPU(Surv(Start,Stop, Event) ~ sex,
#              data = drugdata,
#              ties = "efron")
#   })
# })

test_that("CoxPH counting - Breslow", {
  expect_snapshot({
    coxphGPU(Surv(Start,Stop, Event) ~ sex + age,
             data = drugdata,
             ties = "breslow")
  })
  expect_snapshot({
    coxphGPU(Surv(Start,Stop, Event) ~ sex,
             data = drugdata,
             ties = "breslow")
  })
})
