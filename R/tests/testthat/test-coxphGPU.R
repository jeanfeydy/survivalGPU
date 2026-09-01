testthat::skip_if_not(reticulate::py_module_available("survivalgpu"))

# Dataset
drugdata <- WCE::drugdata

# drugdata2 is drugdata with the last observation for each Id
drugdata2 <- drugdata |>
  dplyr::arrange(Stop |> dplyr::desc()) |>
  dplyr::distinct(Id, .keep_all = TRUE) |>
  dplyr::arrange(Id)

drugdata2$Start <- 0


ties <- "efron"

## Original Coxph model ------
library(survival)

# Surv type counting
coxph <- coxph(
  Surv(Start, Stop, Event) ~ sex + age,
  drugdata,
  ties = ties,

)


# Surv type right
coxph_right <- coxph(
  Surv(Stop, Event) ~ sex + age,
  drugdata2,
  ties = ties,
)

# CoxphGPU model ------

# Counting
coxphGPU <- coxphGPU(
  Surv(Start, Stop, Event) ~ sex + age,
  drugdata,
  ties = ties,
  double_precision = FALSE
)


coxphGPU_bootstrap <- coxphGPU(
  Surv(Start, Stop, Event) ~ sex + age,
  drugdata2,
  ties = ties,
  double_precision = FALSE,
  x = TRUE
)
coxphGPU_bootstrap <- bootstrap(
  coxphGPU_bootstrap, R = 15, patient_id = "Id", data = drugdata2
)


coxphGPU_right <- coxphGPU(
  Surv(Stop, Event) ~ sex + age,
  drugdata2,
  ties = ties,
  double_precision = FALSE
)


# Tests

## Comparison tests between coxphGPU with and without bootstrap
test_that("coxphGPU with and without bootstrap - Coefs", {
  expect_equal(
    as.numeric(coxphGPU$coefficients),
    as.numeric(coxphGPU_bootstrap$coefficients),
    tolerance = 1e-5
  )
})

test_that("coxphGPU with and without bootstrap - Covar matrix", {
  expect_equal(
    as.numeric(coxphGPU$var),
    as.numeric(coxphGPU_bootstrap$var),
    tolerance = 1e-5
  )
})

## Comparison tests between coxph and coxphGPU (counting and right Surv type)
test_that("Coxph counting - Coefs", {
  expect_equal(
    as.numeric(coxph$coefficients),
    as.numeric(coxphGPU_bootstrap$coefficients),
    tolerance = 1e-5
  )
})

test_that("Coxph right - Coefs", {
  expect_equal(
    as.numeric(coxph_right$coefficients),
    as.numeric(coxphGPU_right$coefficients),
    tolerance = 1e-5
  )
})

test_that("Coxph counting - Covar matrix", {
  expect_equal(
    coxph$var,
    coxphGPU_bootstrap$var,
    tolerance = 1e-5
  )
})

test_that("Coxph right - Covar matrix", {
  expect_equal(
    coxph_right$var,
    coxphGPU_right$var,
    tolerance = 1e-5
  )
})

test_that("Coxph counting - log likelihood", {
  expect_equal(
    coxph$loglik[2],
    coxphGPU_bootstrap$loglik[2],
    tolerance = 1e-5
  )
})

test_that("Coxph right - log likelihood", {
  expect_equal(
    coxph_right$loglik[2],
    coxphGPU_right$loglik[2],
    tolerance = 1e-5
  )
})

# N.B.: these are per-observation quantities, so they must be compared
# against a model fit on the same data (drugdata, 1 row per interval).
# coxphGPU_bootstrap is fit on drugdata2 (1 row per patient), so it has a
# different number of rows and cannot be compared element-wise; coxphGPU
# (above) is the right counterpart here.
test_that("Coxph counting - linears predictors", {
  expect_equal(
    coxph$linear.predictors,
    c(coxphGPU$linear.predictors),
    tolerance = 1e-5
  )
})

test_that("Coxph counting - residuals", {
  expect_equal(
    coxph$residuals,
    coxphGPU$residuals,
    tolerance = 1e-5
  )
})

test_that("Coxph counting - resid method", {
  expect_equal(
    resid(coxph),
    resid(coxphGPU),
    tolerance = 1e-5
  )
})

test_that("Coxph counting - resid method score type", {
  expect_equal(
    resid(coxph, type = "score"),
    resid(coxphGPU, type = "score"),
    tolerance = 1e-5
  )
})

test_that("Coxph counting - resid method schoenfeld type", {
  expect_equal(
    resid(coxph, type = "schoenfeld"),
    resid(coxphGPU, type = "schoenfeld"),
    tolerance = 1e-5
  )
})

test_that("Coxph counting - predict method survival type", {
  expect_equal(
    predict(coxph, type = "survival"),
    predict(coxphGPU, type = "survival"),
    tolerance = 1e-5
  )
})

test_that("Coxph counting - predict method lp type", {
  expect_equal(
    predict(coxph, type = "lp"),
    predict(coxphGPU, type = "lp"),
    tolerance = 1e-5
  )
})

test_that("Coxph counting - predict method lp type - linears.predictors check", {
  expect_equal(
    coxphGPU$linear.predictors,
    predict(coxphGPU, type = "lp"),
    tolerance = 1e-5
  )
})

test_that("Coxph counting - predict method risk type", {
  expect_equal(
    predict(coxph, type = "risk"),
    predict(coxphGPU, type = "risk"),
    tolerance = 1e-5
  )
})

test_that("Coxph counting - predict method expected type", {
  expect_equal(
    predict(coxph, type = "expected"),
    predict(coxphGPU, type = "expected"),
    tolerance = 1e-5
  )
})

test_that("Coxph counting - predict method expected type - se.fit", {
  expect_equal(
    predict(coxph, type = "expected", se.fit = TRUE)[[2]],
    predict(coxphGPU, type = "expected", se.fit = TRUE)[[2]],
    tolerance = 1e-5
  )
})

test_that("Coxph right - residuals", {
  expect_equal(
    coxph_right$residuals,
    coxphGPU_right$residuals,
    tolerance = 1e-5
  )
})

test_that("Coxph right - resid method schoenfeld type", {
  expect_equal(
    resid(coxph_right, type = "schoenfeld"),
    resid(coxphGPU_right, type = "schoenfeld"),
    tolerance = 1e-5
  )
})

test_that("Coxph right - predict method survival type", {
  expect_equal(
    predict(coxph_right, type = "survival"),
    predict(coxphGPU_right, type = "survival"),
    tolerance = 1e-5
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
    tolerance = 1e-5
  )
})

coxphGPU_right_drugdata2 <- coxphGPU(
  Surv(Stop, Event) ~ sex + age,
  drugdata2,
  ties = ties,
)

test_that("CoxphGPU counting/right distinct - Coefs", {
  expect_equal(
    coxphGPU$coefficients,
    coxphGPU_right_drugdata2$coefficients,
    tolerance = 1e-5
  )
})

test_that("CoxphGPU counting/right distinct - Covar matrix", {
  expect_equal(
    coxphGPU$var,
    coxphGPU_right_drugdata2$var,
    tolerance = 1e-5
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
    tolerance = 1e-5
  )
})

# test_that("CoxphGPU right with strata - Coefs", {
#   expect_equal(
#     coxph_right_strata$coefficients,
#     coxphGPU_right_strata$coefficients,
#     tolerance = 1e-5
#   )
# })

test_that("CoxphGPU counting with strata - predict", {
  expect_equal(
    predict(coxph_strata, type = "survival"),
    predict(coxphGPU_strata, type = "survival"),
    tolerance = 1e-5
  )
})

test_that("CoxphGPU counting with strata - new data", {
  expect_equal(
    predict(coxph_strata, newdata = head(drugdata)),
    predict(coxphGPU_strata, newdata = head(drugdata)),
    tolerance = 1e-5
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
    tolerance = 1e-5
  )
})

test_that("test1 - loglik", {
  expect_equal(
    temp[, "loglik"],
    temp2[, "loglik"],
    tolerance = 1e-5
  )
})

test_that("test1 - H", {
  expect_equal(
    temp[, "H"],
    temp2[, "H"],
    tolerance = 1e-5
  )
})

################################################################################
# Modular fit-then-bootstrap: bootstrap()
#
# coxphGPU() only ever fits a point estimate; bootstrap() is the only way to
# get bootstrap-based inference. coxphGPU_bootstrap (defined near the top of
# this file) is already such a fit-then-bootstrap() result, and is compared
# against plain coxph()/coxphGPU() above like any other fit.

test_that("bootstrap() leaves the point estimate untouched", {
  expect_equal(coxphGPU_bootstrap$nbootstraps, 15)
  expect_equal(dim(coxphGPU_bootstrap$coef_bootstrap), c(15, 2))
})

test_that("bootstrap() feeds into summary()/print() correctly", {
  s <- summary(coxphGPU_bootstrap)
  expect_s3_class(s, "summary.coxphGPU")
  expect_equal(s$nbootstraps, 15)
  expect_false(is.null(s$conf.int_bootstrap))
})

test_that("bootstrap() errors clearly when x was not stored at fit time", {
  fit_no_x <- coxphGPU(Surv(Start, Stop, Event) ~ sex + age, data = drugdata2)
  expect_error(
    bootstrap(fit_no_x, R = 10, patient_id = "Id", data = drugdata2),
    "x = TRUE"
  )
})

test_that("bootstrap() errors clearly when patient_id/data are missing", {
  fit_x <- coxphGPU(Surv(Start, Stop, Event) ~ sex + age, data = drugdata2, x = TRUE)
  expect_error(
    bootstrap(fit_x, R = 10),
    "patient_id and data are required"
  )
})

test_that("bootstrap() realigns patient_id correctly under subset=", {
  fit_subset <- coxphGPU(
    Surv(Stop, Event) ~ sex + age,
    data = drugdata2, x = TRUE, subset = (Id <= 40)
  )
  fit_subset_boot <- bootstrap(
    fit_subset, R = 20, patient_id = "Id", data = drugdata2, batchsize = 10
  )
  expect_equal(fit_subset$coefficients, fit_subset_boot$coefficients, tolerance = 1e-8)
  expect_equal(fit_subset_boot$nbootstraps, 20)
})

test_that("bootstrap() works for a stratified counting-type model", {
  fit_strata <- coxphGPU(
    Surv(Start, Stop, Event) ~ strata(sex) + age, data = drugdata, x = TRUE
  )
  fit_strata_boot <- bootstrap(
    fit_strata, R = 15, patient_id = "Id", data = drugdata, batchsize = 10
  )
  expect_equal(fit_strata$coefficients, fit_strata_boot$coefficients, tolerance = 1e-8)
  expect_equal(fit_strata_boot$nbootstraps, 15)
})
