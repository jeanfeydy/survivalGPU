# Dataset
drugdata <- WCE::drugdata

# drugdata2 is drugdata with the last observation for each Id
drugdata2 <- drugdata |>
  dplyr::arrange(Stop |> dplyr::desc()) |>
  dplyr::distinct(Id, .keep_all = TRUE) |>
  dplyr::arrange(Id)

drugdata2$Start <- 0

# TODO: remove this when the re-implementation of coxph is over
# We currently do not support Efron ties, only Breslow
ties <- "efron"

## Original Coxph model ------
library(survival)

# Surv type counting
coxph <- coxph(
  Surv(Start, Stop, Event) ~ sex + age,
  drugdata,
  ties = ties
)

print("Coxph counting done")

# Surv type right
coxph_right <- coxph(
  Surv(Stop, Event) ~ sex + age,
  drugdata2,
  ties = ties
)

print("Coxph right done")

## CoxphGPU model ------

# Counting
coxphGPU <- coxphGPU(
  Surv(Start, Stop, Event) ~ sex + age,
  drugdata,
  ties = ties,
  bootstrap = 1
)

print("CoxphGPU counting done")

coxphGPU_bootstrap <- coxphGPU(
  Surv(Start, Stop, Event) ~ sex + age,
  drugdata,
  ties = ties,
  bootstrap = 15
)

print("CoxphGPU counting with bootstrap done")

coxphGPU_right <- coxphGPU(
  Surv(Stop, Event) ~ sex + age,
  drugdata2,
  ties = ties,
  bootstrap = 1
)

print("CoxphGPU right done")

# Tests

## Comparison tests between coxphGPU with and without bootstrap
test_that("coxphGPU with and without bootstrap - Coefs", {
  expect_equal(
    as.numeric(coxphGPU$coefficients),
    as.numeric(coxphGPU_bootstrap$coefficients),
    tolerance = 1e-4
  )
})

test_that("coxphGPU with and without bootstrap - Covar matrix", {
  expect_equal(
    as.numeric(coxphGPU$var),
    as.numeric(coxphGPU_bootstrap$var),
    tolerance = 1e-4
  )
})

## Comparison tests between coxph and coxphGPU (counting and right Surv type)
test_that("Coxph counting - Coefs", {
  expect_equal(
    as.numeric(coxph$coefficients),
    as.numeric(coxphGPU_bootstrap$coefficients),
    tolerance = 1e-4
  )
})

test_that("Coxph right - Coefs", {
  expect_equal(
    as.numeric(coxph_right$coefficients),
    as.numeric(coxphGPU_right$coefficients),
    tolerance = 1e-4
  )
})

test_that("Coxph counting - Covar matrix", {
  expect_equal(
    coxph$var,
    coxphGPU_bootstrap$var,
    tolerance = 1e-4
  )
})

test_that("Coxph right - Covar matrix", {
  expect_equal(
    coxph_right$var,
    coxphGPU_right$var,
    tolerance = 1e-4
  )
})

test_that("Coxph counting - log likelihood", {
  expect_equal(
    coxph$loglik[2],
    coxphGPU_bootstrap$loglik[2],
    tolerance = 1e-4
  )
})

test_that("Coxph right - log likelihood", {
  expect_equal(
    coxph_right$loglik[2],
    coxphGPU_right$loglik[2],
    tolerance = 1e-4
  )
})

# same lp ? (because not the same colMeans)
test_that("Coxph counting - linears predictors", {
  expect_equal(
    coxph$linear.predictors,
    c(coxphGPU_bootstrap$linear.predictors),
    tolerance = 1e-4
  )
})

test_that("Coxph counting - residuals", {
  expect_equal(
    coxph$residuals,
    coxphGPU_bootstrap$residuals,
    tolerance = 1e-4
  )
})

test_that("Coxph counting - resid method", {
  expect_equal(
    resid(coxph),
    resid(coxphGPU_bootstrap),
    tolerance = 1e-4
  )
})

test_that("Coxph counting - resid method score type", {
  expect_equal(
    resid(coxph, type = "score"),
    resid(coxphGPU_bootstrap, type = "score"),
    tolerance = 1e-4
  )
})

test_that("Coxph counting - resid method schoenfeld type", {
  expect_equal(
    resid(coxph, type = "schoenfeld"),
    resid(coxphGPU_bootstrap, type = "schoenfeld"),
    tolerance = 1e-4
  )
})

test_that("Coxph counting - predict method survival type", {
  expect_equal(
    predict(coxph, type = "survival"),
    predict(coxphGPU, type = "survival"),
    tolerance = 1e-4
  )
})

test_that("Coxph counting - predict method lp type", {
  expect_equal(
    predict(coxph, type = "lp"),
    predict(coxphGPU, type = "lp"),
    tolerance = 1e-4
  )
})

test_that("Coxph counting - predict method lp type - linears.predictors check", {
  expect_equal(
    coxphGPU$linear.predictors,
    predict(coxphGPU, type = "lp"),
    tolerance = 1e-4
  )
})

test_that("Coxph counting - predict method risk type", {
  expect_equal(
    predict(coxph, type = "risk"),
    predict(coxphGPU, type = "risk"),
    tolerance = 1e-4
  )
})

test_that("Coxph counting - predict method expected type", {
  expect_equal(
    predict(coxph, type = "expected"),
    predict(coxphGPU, type = "expected"),
    tolerance = 1e-4
  )
})

test_that("Coxph counting - predict method expected type - se.fit", {
  expect_equal(
    predict(coxph, type = "expected", se.fit = TRUE)[[2]],
    predict(coxphGPU, type = "expected", se.fit = TRUE)[[2]],
    tolerance = 1e-4
  )
})

test_that("Coxph right - residuals", {
  expect_equal(
    coxph_right$residuals,
    coxphGPU_right$residuals,
    tolerance = 1e-4
  )
})

test_that("Coxph right - resid method schoenfeld type", {
  expect_equal(
    resid(coxph_right, type = "schoenfeld"),
    resid(coxphGPU_right, type = "schoenfeld"),
    tolerance = 1e-4
  )
})

test_that("Coxph right - predict method survival type", {
  expect_equal(
    predict(coxph_right, type = "survival"),
    predict(coxphGPU_right, type = "survival"),
    tolerance = 1e-4
  )
})


# TODO: FIX THE NO ITER

# Cox model with no iterations
coxphGPU_no_iter <- coxphGPU(
  Surv(Start, Stop, Event) ~ sex + age,
  drugdata,
  ties = ties,
  iter.max = 0
)

test_that("No Newton iterations - Null Coefs", {
  expect_equal(
    as.vector(coxphGPU_no_iter$coefficients),
    c(0, 0),
    tolerance = 1e-4
  )
})

coxphGPU_right_drugdata2 <- coxphGPU(
  Surv(Stop, Event) ~ sex + age,
  drugdata2,
  ties = ties,
  bootstrap = 1
)

test_that("CoxphGPU counting/right distinct - Coefs", {
  expect_equal(
    coxphGPU$coefficients,
    coxphGPU_right_drugdata2$coefficients,
    tolerance = 1e-4
  )
})

test_that("CoxphGPU counting/right distinct - Covar matrix", {
  expect_equal(
    coxphGPU$var,
    coxphGPU_right_drugdata2$var,
    tolerance = 1e-4
  )
})

################################################################################
# Test strata

coxph_strata <- coxph(
  Surv(Start, Stop, Event) ~ strata(sex) + age,
  drugdata,
  ties = ties
)

coxphGPU_strata <- coxphGPU(
  Surv(Start, Stop, Event) ~ strata(sex) + age,
  drugdata,
  ties = ties
)

coxph_right_strata <- coxph(
  Surv(Stop, Event) ~ strata(sex) + age,
  drugdata,
  ties = ties
)

# coxphGPU_right_strata <- coxphGPU(
#   Surv(Stop, Event) ~ strata(sex) + age,
#   drugdata,
#   ties = ties
# )

test_that("CoxphGPU counting with strata - Coefs", {
  expect_equal(
    coxph_strata$coefficients,
    coxphGPU_strata$coefficients,
    tolerance = 1e-4
  )
})

# test_that("CoxphGPU right with strata - Coefs", {
#   expect_equal(
#     coxph_right_strata$coefficients,
#     coxphGPU_right_strata$coefficients,
#     tolerance = 1e-4
#   )
# })

test_that("CoxphGPU counting with strata - predict", {
  expect_equal(
    predict(coxph_strata, type = "survival"),
    predict(coxphGPU_strata, type = "survival"),
    tolerance = 1e-4
  )
})

test_that("CoxphGPU counting with strata - new data", {
  expect_equal(
    predict(coxph_strata, newdata = head(drugdata)),
    predict(coxphGPU_strata, newdata = head(drugdata)),
    tolerance = 1e-4
  )
})

################################################################################
# Software validation vignette of Terry Therneau

test1 <- data.frame(
  time = c(1, 1, 6, 6, 8, 9),
  status = c(1, 0, 1, 1, 0, 1),
  x = c(1, 1, 1, 0, 0, 0)
)

temp  <- matrix(0, nrow = 6, ncol = 4,
                dimnames = list(1:6, c("iter", "beta", "loglik", "H")))
temp2 <- matrix(0, nrow = 6, ncol = 4,
                dimnames = list(1:6, c("iter", "beta", "loglik", "H")))

# routine précise ? car bug aléatoire
convergence_warning <- capture_warnings(
  for (i in 0:5) {
    # coxph
    tfit <- coxph(
      Surv(time, status) ~ x,
      data = test1,
      ties = "breslow",
      iter.max = i
    )

    # coxphGPU
    tfit2 <- coxphGPU(
      Surv(time, status) ~ x,
      data = test1,
      ties = "breslow",
      iter.max = i
    )

    temp[i + 1, ]  <- c(tfit$iter, coef(tfit), tfit$loglik[2], 1 / vcov(tfit))
    temp2[i + 1, ] <- c(tfit2$iter, coef(tfit2), tfit2$loglik[2], 1 / tfit2$var)
  }
)

test_that("Convergence warnings ?", {
  expect_match(
    convergence_warning,
    "Ran out of iterations and did not converge",
    all = TRUE
  )
})

test_that("test1 - beta", {
  expect_equal(
    temp[, "beta"],
    temp2[, "beta"],
    tolerance = 1e-4
  )
})

test_that("test1 - loglik", {
  expect_equal(
    temp[, "loglik"],
    temp2[, "loglik"],
    tolerance = 1e-4
  )
})

test_that("test1 - H", {
  expect_equal(
    temp[, "H"],
    temp2[, "H"],
    tolerance = 1e-4
  )
})
