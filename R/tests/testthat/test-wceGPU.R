testthat::skip_if_not(reticulate::py_module_available("survivalgpu"))
skip_if_no_pykeops()

# Dataset
drugdata <- WCE::drugdata

# WCE models -------
# WCE GPU
wce_gpu <- wceGPU(
  data = drugdata, nknots = 1, cutoff = 90, id = "Id",
  event = "Event", start = "Start", stop = "Stop",
  expos = "dose", covariates = c("age", "sex"),
  constrained = FALSE, aic = FALSE, confint = 0.95,
  verbosity = 3, double_precision = TRUE
)

wce_gpu_bootstrap <- bootstrap(wce_gpu, R = 15, data = drugdata, batchsize = 0)

# Original WCE
wce <- WCE::WCE(
  data = drugdata, analysis = "Cox", nknots = 1, cutoff = 90,
  id = "Id", event = "Event", start = "Start", stop = "Stop",
  expos = "dose", covariates = c("age", "sex"),
  constrained = FALSE, aic = FALSE, double_precision = TRUE
)


# Tests
# Check weight function between WCE GPU and original WCE
test_that("WCE mat", {
  expect_equal(
    as.vector(wce$WCEmat),
    as.vector(wce_gpu$WCEmat),
    tolerance = 1e-4
  )
})

# Check coefs between WCE GPU and original WCE
test_that("coef covariates", {
  expect_equal(
    as.vector(wce$beta.hat.covariates),
    as.vector(wce_gpu$beta.hat.covariates),
    tolerance = 1e-4
  )
})

# Check SE between WCE GPU and original WCE
test_that("SE covariates", {
  expect_equal(
    as.vector(wce$se.covariates),
    as.vector(wce_gpu$se.covariates),
    tolerance = 1e-4
  )
})

# Check covariance matrix between WCE GPU and original WCE
test_that("Vcovmat", {
  expect_equal(
    wce$vcovmat[[1]],
    wce_gpu$vcovmat[[1]],
    tolerance = 1e-4
  )
})

# Check ll between WCE GPU and original WCE
test_that("Partial ll", {
  expect_equal(
    as.vector(wce$loglik),
    as.vector(wce_gpu$loglik),
    tolerance = 1e-4
  )
})

# Check AIC/BIC between WCE GPU and original WCE
test_that("info.criterion", {
  expect_equal(
    as.numeric(wce$info.criterion),
    as.numeric(wce_gpu$info.criterion),
    tolerance = 1e-4
  )
})


# Check HR between WCE GPU and original WCE
exposed   <- rep(1, 90)
unexposed <- rep(0, 90)

test_that("HR", {
  expect_equal(
    HR(wce_gpu_bootstrap, exposed, unexposed)[1],
    WCE::HR.WCE(wce, exposed, unexposed)[1],
    tolerance = 1e-4
  )
})


# Each of these compares wceGPU() against the reference WCE::WCE()
# implementation for the same arguments, rather than snapshotting the
# printed output: raw floating-point prints are not stable across R
# versions/BLAS/torch builds (trailing whitespace, last-digit rounding),
# which made the previous expect_snapshot()-based tests flaky across CI
# environments.
test_that("WCE - no covariates", {
  wce_ref <- WCE::WCE(
    data = drugdata, analysis = "Cox", nknots = 1, cutoff = 90,
    id = "Id", event = "Event", start = "Start", stop = "Stop",
    expos = "dose",
    constrained = FALSE, aic = FALSE, double_precision = TRUE
  )
  wce_test <- wceGPU(
    data = drugdata, nknots = 1, cutoff = 90, id = "Id",
    event = "Event", start = "Start", stop = "Stop",
    expos = "dose",
    constrained = FALSE, aic = FALSE, confint = 0.95
  )
  expect_equal(as.vector(wce_ref$WCEmat), as.vector(wce_test$WCEmat), tolerance = 1e-4)
})

test_that("WCE - one covariate", {
  wce_ref <- WCE::WCE(
    data = drugdata, analysis = "Cox", nknots = 1, cutoff = 90,
    id = "Id", event = "Event", start = "Start", stop = "Stop",
    expos = "dose", covariates = c("age"),
    constrained = FALSE, aic = FALSE, double_precision = TRUE
  )
  wce_test <- wceGPU(
    data = drugdata, nknots = 1, cutoff = 90, id = "Id",
    event = "Event", start = "Start", stop = "Stop",
    expos = "dose", covariates = c("age"),
    constrained = FALSE, aic = FALSE, confint = 0.95
  )
  expect_equal(as.vector(wce_ref$WCEmat), as.vector(wce_test$WCEmat), tolerance = 1e-4)
  expect_equal(as.vector(wce_ref$beta.hat.covariates), as.vector(wce_test$beta.hat.covariates), tolerance = 1e-4)
})

test_that("WCE - two covariates", {
  wce_ref <- WCE::WCE(
    data = drugdata, analysis = "Cox", nknots = 1, cutoff = 90,
    id = "Id", event = "Event", start = "Start", stop = "Stop",
    expos = "dose", covariates = c("age", "sex"),
    constrained = FALSE, aic = FALSE, double_precision = TRUE
  )
  wce_test <- wceGPU(
    data = drugdata, nknots = 1, cutoff = 90, id = "Id",
    event = "Event", start = "Start", stop = "Stop",
    expos = "dose", covariates = c("age", "sex"),
    constrained = FALSE, aic = FALSE, confint = 0.95
  )
  expect_equal(as.vector(wce_ref$WCEmat), as.vector(wce_test$WCEmat), tolerance = 1e-4)
  expect_equal(as.vector(wce_ref$beta.hat.covariates), as.vector(wce_test$beta.hat.covariates), tolerance = 1e-4)
})

test_that("WCE - AIC", {
  wce_ref <- WCE::WCE(
    data = drugdata, analysis = "Cox", nknots = 1, cutoff = 90,
    id = "Id", event = "Event", start = "Start", stop = "Stop",
    expos = "dose", covariates = c("age", "sex"),
    constrained = FALSE, aic = TRUE, double_precision = TRUE
  )
  wce_test <- wceGPU(
    data = drugdata, nknots = 1, cutoff = 90, id = "Id",
    event = "Event", start = "Start", stop = "Stop",
    expos = "dose", covariates = c("age", "sex"),
    constrained = FALSE, aic = TRUE, confint = 0.95
  )
  expect_equal(as.vector(wce_ref$WCEmat), as.vector(wce_test$WCEmat), tolerance = 1e-4)
  expect_equal(as.vector(wce_ref$beta.hat.covariates), as.vector(wce_test$beta.hat.covariates), tolerance = 1e-4)
  expect_equal(as.numeric(wce_ref$info.criterion), as.numeric(wce_test$info.criterion), tolerance = 1e-4)
  expect_true(wce_test$aic)
})

test_that("WCE - right constraint", {
  wce_ref <- WCE::WCE(
    data = drugdata, analysis = "Cox", nknots = 1, cutoff = 90,
    id = "Id", event = "Event", start = "Start", stop = "Stop",
    expos = "dose", covariates = c("age", "sex"),
    constrained = "R", aic = FALSE, double_precision = TRUE
  )
  wce_test <- wceGPU(
    data = drugdata, nknots = 1, cutoff = 90, id = "Id",
    event = "Event", start = "Start", stop = "Stop",
    expos = "dose", covariates = c("age", "sex"),
    constrained = "R", aic = FALSE, confint = 0.95
  )
  expect_equal(as.vector(wce_ref$WCEmat), as.vector(wce_test$WCEmat), tolerance = 1e-4)
  expect_equal(as.vector(wce_ref$beta.hat.covariates), as.vector(wce_test$beta.hat.covariates), tolerance = 1e-4)
})

test_that("WCE - left constraint", {
  wce_ref <- WCE::WCE(
    data = drugdata, analysis = "Cox", nknots = 1, cutoff = 90,
    id = "Id", event = "Event", start = "Start", stop = "Stop",
    expos = "dose", covariates = c("age", "sex"),
    constrained = "L", aic = FALSE, double_precision = TRUE
  )
  wce_test <- wceGPU(
    data = drugdata, nknots = 1, cutoff = 90, id = "Id",
    event = "Event", start = "Start", stop = "Stop",
    expos = "dose", covariates = c("age", "sex"),
    constrained = "L", aic = FALSE, confint = 0.95
  )
  expect_equal(as.vector(wce_ref$WCEmat), as.vector(wce_test$WCEmat), tolerance = 1e-4)
  expect_equal(as.vector(wce_ref$beta.hat.covariates), as.vector(wce_test$beta.hat.covariates), tolerance = 1e-4)
})

test_that("WCE - 3 knots", {
  wce_ref <- WCE::WCE(
    data = drugdata, analysis = "Cox", nknots = 3, cutoff = 90,
    id = "Id", event = "Event", start = "Start", stop = "Stop",
    expos = "dose", covariates = c("age", "sex"),
    constrained = FALSE, aic = FALSE, double_precision = TRUE
  )
  wce_test <- wceGPU(
    data = drugdata, nknots = 3, cutoff = 90, id = "Id",
    event = "Event", start = "Start", stop = "Stop",
    expos = "dose", covariates = c("age", "sex"),
    constrained = FALSE, aic = FALSE, confint = 0.95
  )
  expect_equal(as.vector(wce_ref$WCEmat), as.vector(wce_test$WCEmat), tolerance = 1e-4)
  expect_equal(as.vector(wce_ref$beta.hat.covariates), as.vector(wce_test$beta.hat.covariates), tolerance = 1e-4)
})
