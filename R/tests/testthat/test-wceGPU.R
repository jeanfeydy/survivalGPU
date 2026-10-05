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
  nbootstraps = 0, batchsize = 0, verbosity = 3, double_precision = TRUE
)

wce_gpu_bootstrap <- wceGPU(
  data = drugdata, nknots = 1, cutoff = 90, id = "Id",
  event = "Event", start = "Start", stop = "Stop",
  expos = "dose", covariates = c("age", "sex"),
  constrained = FALSE, aic = FALSE, confint = 0.95,
  nbootstraps = 15, batchsize = 0, double_precision = TRUE
)

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

# Regression: confint.wceGPU() used to crash with
# "'list' object cannot be coerced to type 'integer'", since object$vcovmat
# is a named list (holding the selected model's covariance matrix), and
# confint() called diag() on it directly instead of unwrapping it first.
test_that("confint.wceGPU returns Wald CIs matching beta.hat +/- z*se", {
  ci <- confint(wce_gpu)

  expected_age <- wce_gpu$beta.hat.covariates[1, "age"] +
    qnorm(c(0.025, 0.975)) * wce_gpu$se.covariates[1, "age"]
  expected_sex <- wce_gpu$beta.hat.covariates[1, "sex"] +
    qnorm(c(0.025, 0.975)) * wce_gpu$se.covariates[1, "sex"]

  expect_equal(unname(ci["age", ]), expected_age, tolerance = 1e-8)
  expect_equal(unname(ci["sex", ]), expected_sex, tolerance = 1e-8)
  expect_equal(rownames(ci), c("age", "sex"))

  # parm subsetting, by name and by index:
  expect_equal(confint(wce_gpu, parm = "age"), ci["age", , drop = FALSE])
  expect_equal(confint(wce_gpu, parm = 1), ci["age", , drop = FALSE])
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
    constrained = FALSE, aic = FALSE, confint = 0.95,
    nbootstraps = 0, batchsize = 0
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
    constrained = FALSE, aic = FALSE, confint = 0.95,
    nbootstraps = 0, batchsize = 0
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
    constrained = FALSE, aic = FALSE, confint = 0.95,
    nbootstraps = 0, batchsize = 0
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
    constrained = FALSE, aic = TRUE, confint = 0.95,
    nbootstraps = 0, batchsize = 0
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
    constrained = "R", aic = FALSE, confint = 0.95,
    nbootstraps = 0, batchsize = 0
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
    constrained = "L", aic = FALSE, confint = 0.95,
    nbootstraps = 0, batchsize = 0
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
    constrained = FALSE, aic = FALSE, confint = 0.95,
    nbootstraps = 0, batchsize = 0
  )
  expect_equal(as.vector(wce_ref$WCEmat), as.vector(wce_test$WCEmat), tolerance = 1e-4)
  expect_equal(as.vector(wce_ref$beta.hat.covariates), as.vector(wce_test$beta.hat.covariates), tolerance = 1e-4)
})

# Multi-knot selection: nknots accepts a vector of candidates, and the
# candidate that minimizes the information criterion is selected -- this
# should match what WCE::WCE()'s own multi-candidate selection picks.
test_that("WCE - multi-knot selection matches WCE::WCE()'s own best candidate", {
  wce_ref <- WCE::WCE(
    data = drugdata, analysis = "Cox", nknots = 1:3, cutoff = 90,
    id = "Id", event = "Event", start = "Start", stop = "Stop",
    expos = "dose", covariates = c("age", "sex"),
    constrained = FALSE, aic = FALSE, double_precision = TRUE
  )
  best_ref <- which.min(wce_ref$info.criterion)

  wce_test <- wceGPU(
    data = drugdata, nknots = 1:3, cutoff = 90, id = "Id",
    event = "Event", start = "Start", stop = "Stop",
    expos = "dose", covariates = c("age", "sex"),
    constrained = FALSE, aic = FALSE, confint = 0.95,
    nbootstraps = 0, batchsize = 0
  )

  expect_equal(wce_test$best.nknots, (1:3)[best_ref])
  expect_equal(
    as.vector(wce_ref$WCEmat[best_ref, ]), as.vector(wce_test$WCEmat),
    tolerance = 1e-4
  )
  expect_equal(
    as.vector(wce_ref$beta.hat.covariates[best_ref, ]), as.vector(wce_test$beta.hat.covariates),
    tolerance = 1e-4
  )
  expect_equal(as.vector(wce_ref$info.criterion), wce_test$info.criterion.grid, tolerance = 1e-4)
})

test_that("WCE - best.nknots is one of the tried candidates", {
  wce_test <- wceGPU(
    data = drugdata, nknots = 1:3, cutoff = 90, id = "Id",
    event = "Event", start = "Start", stop = "Stop",
    expos = "dose", covariates = c("age", "sex"),
    constrained = FALSE, aic = FALSE, confint = 0.95,
    nbootstraps = 0, batchsize = 0
  )
  expect_true(wce_test$best.nknots %in% c(1, 2, 3))
  expect_equal(wce_test$nknots.grid, c(1, 2, 3))
})

test_that("WCE - summary(allres = TRUE) shows every candidate tried", {
  wce_test <- wceGPU(
    data = drugdata, nknots = 1:3, cutoff = 90, id = "Id",
    event = "Event", start = "Start", stop = "Stop",
    expos = "dose", covariates = c("age", "sex"),
    constrained = FALSE, aic = FALSE, confint = 0.95,
    nbootstraps = 0, batchsize = 0
  )
  out <- capture.output(summary(wce_test, allres = TRUE))
  expect_true(any(grepl("Candidate models tried", out)))
  expect_true(any(grepl("Best model", out)))
})

test_that("WCE - best.nknots matches nknots for a single (scalar) candidate", {
  wce_test <- wceGPU(
    data = drugdata, nknots = 1, cutoff = 90, id = "Id",
    event = "Event", start = "Start", stop = "Stop",
    expos = "dose", covariates = c("age", "sex"),
    constrained = FALSE, aic = FALSE, confint = 0.95,
    nbootstraps = 0, batchsize = 0
  )
  expect_equal(wce_test$best.nknots, 1)
})

# Bootstrap + multi-knot: each replicate should independently select its own
# best nknots -- no production code changes were needed for this beyond the
# multi-knot support above, since Python already resolves per-replicate
# selection before the bootstrap arrays ever reach R.
test_that("WCE - bootstrap selects the best knot per replicate", {
  nbootstraps <- 20
  wce_boot <- wceGPU(
    data = drugdata, nknots = 1:3, cutoff = 90, id = "Id",
    event = "Event", start = "Start", stop = "Stop",
    expos = "dose", covariates = c("age", "sex"),
    constrained = FALSE, aic = FALSE, confint = 0.95,
    nbootstraps = nbootstraps, batchsize = 10
  )

  expect_true(wce_boot$is_bootstraps)
  expect_length(wce_boot$bootstrap.best.nknots, nbootstraps)
  expect_true(all(wce_boot$bootstrap.best.nknots %in% c(1, 2, 3)))
  expect_equal(sum(table(wce_boot$bootstrap.best.nknots)), nbootstraps)
  expect_equal(dim(wce_boot$WCEmat_bootstrap), c(nbootstraps, 90))
  expect_equal(dim(wce_boot$WCEmat_CI), c(2, 90))

  exposed   <- rep(1, 90)
  unexposed <- rep(0, 90)
  hr <- HR(wce_boot, exposed, unexposed)
  expect_true(all(c("HR", "CI 2.5%", "CI 97.5%") %in% colnames(hr)))
})
