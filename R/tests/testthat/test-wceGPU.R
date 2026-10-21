skip_if_no_backend()
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
  verbosity = 3, double_precision = TRUE, x = TRUE
)

wce_gpu_bootstrap <- bootstrap(wce_gpu, R = 15, batchsize = 0)

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

# HR() takes exposure vectors in reverse chronological order (first value =
# current time). Shorter vectors are padded with zeros, longer ones truncated
# to the cutoff. Where WCE::HR.WCE() can be used (full-length vectors, no
# bootstrap), HR() is compared to it; otherwise to the HR computed by hand
# from the estimated weight function(s).
hr_by_hand <- function(weights, vecnum, vecdenom) {
  exp(sum(weights * (vecnum - vecdenom)))
}
pad <- function(x, n = 90) c(x, rep(0, n - length(x)))

test_that("HR - matches WCE::HR.WCE() for full-length scenarios", {
  set.seed(42)
  scenarios <- list(
    list(rep(1, 90), rep(0, 90)),
    list(pad(rep(2, 10)), rep(0, 90)),
    list(c(rep(0, 60), rep(1, 30)), c(rep(1, 30), rep(0, 60))),
    list(runif(90, 0, 3), runif(90, 0, 3))
  )
  for (s in scenarios) {
    hr <- HR(wce_gpu, s[[1]], s[[2]])
    expect_equal(colnames(hr), "HR")
    expect_equal(hr[[1, "HR"]], WCE::HR.WCE(wce, s[[1]], s[[2]])[1],
                 tolerance = 1e-4)
    expect_equal(hr[[1, "HR"]], hr_by_hand(wce_gpu$WCEmat[1, ], s[[1]], s[[2]]))
  }
})

test_that("HR - shorter vectors are padded with zeros", {
  recent <- rep(1, 30)
  expected <- WCE::HR.WCE(wce, pad(recent), rep(0, 90))[1]

  expect_equal(HR(wce_gpu, recent, rep(0, 90))[[1, "HR"]], expected,
               tolerance = 1e-4)
  # a single value is a vector of length 1: padded too
  expect_equal(HR(wce_gpu, recent, 0)[[1, "HR"]], expected, tolerance = 1e-4)
  expect_equal(HR(wce_gpu, recent, 0)[[1, "HR"]],
               hr_by_hand(wce_gpu$WCEmat[1, ], pad(recent), rep(0, 90)))
})

test_that("HR - longer vectors are truncated to the cutoff", {
  expected <- hr_by_hand(wce_gpu$WCEmat[1, ], rep(1, 90), rep(0, 90))

  expect_equal(HR(wce_gpu, c(rep(1, 90), rep(5, 30)), rep(0, 150))[[1, "HR"]],
               expected)
  # exposures older than the cutoff have no effect
  expect_equal(HR(wce_gpu, c(rep(1, 90), rep(100, 30)), 0)[[1, "HR"]],
               expected)
})

test_that("HR - mixed lengths and fractional doses", {
  vecnum <- rep(1, 30)       # shorter than the cutoff
  vecdenom <- rep(0.5, 120)  # fractional doses, longer than the cutoff
  full_num <- pad(vecnum)
  full_den <- rep(0.5, 90)

  hr <- HR(wce_gpu, vecnum, vecdenom)[[1, "HR"]]
  expect_equal(hr, WCE::HR.WCE(wce, full_num, full_den)[1], tolerance = 1e-4)
  expect_equal(hr, hr_by_hand(wce_gpu$WCEmat[1, ], full_num, full_den))
})

test_that("HR - bootstrap confidence interval computed by hand", {
  delta <- pad(rep(1, 30))
  boot_hr <- exp(as.vector(wce_gpu_bootstrap$WCEmat_bootstrap %*% delta))

  hr <- HR(wce_gpu_bootstrap, rep(1, 30), 0)
  expect_equal(colnames(hr), c("HR", "CI 2.5%", "CI 97.5%"))
  expect_equal(hr[[1, "HR"]],
               hr_by_hand(wce_gpu_bootstrap$WCEmat[1, ], delta, rep(0, 90)))
  expect_equal(unname(hr[1, 2:3]), unname(quantile(boot_hr, c(0.025, 0.975))))

  hr90 <- HR(wce_gpu_bootstrap, rep(1, 30), 0, level = 0.9)
  expect_equal(colnames(hr90), c("HR", "CI 5%", "CI 95%"))
  expect_equal(unname(hr90[1, 2:3]), unname(quantile(boot_hr, c(0.05, 0.95))))
})

test_that("HR - rejects objects that are not wceGPU fits", {
  expect_error(HR(list(), 1, 0), "not a wceGPU object")
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
    constrained = FALSE, aic = FALSE, confint = 0.95
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
    constrained = FALSE, aic = FALSE, confint = 0.95
  )
  expect_true(wce_test$best.nknots %in% c(1, 2, 3))
  expect_equal(wce_test$nknots.grid, c(1, 2, 3))
})

test_that("WCE - summary(allres = TRUE) shows every candidate tried", {
  wce_test <- wceGPU(
    data = drugdata, nknots = 1:3, cutoff = 90, id = "Id",
    event = "Event", start = "Start", stop = "Stop",
    expos = "dose", covariates = c("age", "sex"),
    constrained = FALSE, aic = FALSE, confint = 0.95
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
    constrained = FALSE, aic = FALSE, confint = 0.95
  )
  expect_equal(wce_test$best.nknots, 1)
})

# Bootstrap + multi-knot: bootstrap() passes every nknots candidate back to
# Python, so each replicate independently selects its own best nknots. The
# warm start (init = TRUE) is skipped there, since the candidates' spline
# coefficients have different lengths.
test_that("WCE - bootstrap selects the best knot per replicate", {
  nbootstraps <- 20
  wce_fit <- wceGPU(
    data = drugdata, nknots = 1:3, cutoff = 90, id = "Id",
    event = "Event", start = "Start", stop = "Stop",
    expos = "dose", covariates = c("age", "sex"),
    constrained = FALSE, aic = FALSE, confint = 0.95, x = TRUE
  )
  wce_boot <- bootstrap(wce_fit, R = nbootstraps, batchsize = 10)

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

  # Spline coefficients have different lengths across candidates: no dense
  # bootstrap array for them.
  expect_null(wce_boot$bootstrap_est)
  expect_true(all(is.finite(wce_boot$coef_CI)))

  # The point estimate is left untouched by bootstrap().
  expect_equal(wce_boot$WCEmat, wce_fit$WCEmat)
  expect_equal(wce_boot$best.nknots, wce_fit$best.nknots)
})

test_that("WCE - bootstrap rejects a numeric init with several nknots", {
  wce_fit <- wceGPU(
    data = drugdata, nknots = 1:2, cutoff = 90, id = "Id",
    event = "Event", start = "Start", stop = "Stop",
    expos = "dose", covariates = c("age", "sex"), x = TRUE
  )
  expect_error(
    bootstrap(wce_fit, R = 2, init = rep(0, 6)),
    "single nknots value"
  )
})

test_that("WCE - x = TRUE stores the columns used by the model", {
  expect_equal(
    wce_gpu$data,
    drugdata[c("Id", "Event", "Start", "Stop", "dose", "age", "sex")]
  )

  wce_no_x <- wceGPU(
    data = drugdata, nknots = 1, cutoff = 90, id = "Id",
    event = "Event", start = "Start", stop = "Stop", expos = "dose"
  )
  expect_null(wce_no_x$data)
  expect_error(bootstrap(wce_no_x, R = 2), "x = TRUE")
})

test_that("WCE - bootstrap() does not take a data argument", {
  expect_error(
    bootstrap(wce_gpu, R = 2, data = drugdata),
    "doesn't take a data argument"
  )
})

test_that("WCE - wceGPU no longer takes the fit-time bootstrap arguments", {
  expect_error(
    wceGPU(
      data = drugdata, nknots = 1, cutoff = 90, id = "Id",
      event = "Event", start = "Start", stop = "Stop",
      expos = "dose", nbootstraps = 5
    ),
    "no longer takes nbootstraps"
  )
})

# The bootstrap arrays have one row per replicate and one column per
# coefficient: they must keep both dimensions when one of them has length 1.
test_that("WCE - bootstrap with a single replicate", {
  wce_test <- wceGPU(
    data = drugdata, nknots = 1, cutoff = 90, id = "Id",
    event = "Event", start = "Start", stop = "Stop",
    expos = "dose", covariates = c("age", "sex"),
    constrained = FALSE, aic = FALSE, confint = 0.95, x = TRUE
  )
  wce_test <- bootstrap(wce_test, R = 1)

  expect_equal(dim(wce_test$bootstrap_beta.hat.covariates), c(1, 2))
  expect_equal(dim(wce_test$bootstrap_est), c(1, ncol(wce_test$est)))
  expect_equal(dim(wce_test$WCEmat_bootstrap), c(1, 90))
})

test_that("WCE - bootstrap with a single covariate", {
  nbootstraps <- 5
  wce_test <- wceGPU(
    data = drugdata, nknots = 1, cutoff = 90, id = "Id",
    event = "Event", start = "Start", stop = "Stop",
    expos = "dose", covariates = "age",
    constrained = FALSE, aic = FALSE, confint = 0.95, x = TRUE
  )
  wce_test <- bootstrap(wce_test, R = nbootstraps)

  expect_equal(dim(wce_test$bootstrap_beta.hat.covariates), c(nbootstraps, 1))
  expect_equal(colnames(wce_test$bootstrap_beta.hat.covariates), "age")
  expect_equal(dim(wce_test$coef_CI), c(2, 1))
})

test_that("WCE - summary() of a bootstrap fit without covariates", {
  wce_test <- wceGPU(
    data = drugdata, nknots = 1, cutoff = 90, id = "Id",
    event = "Event", start = "Start", stop = "Stop",
    expos = "dose",
    constrained = FALSE, aic = FALSE, confint = 0.95, x = TRUE
  )
  wce_test <- bootstrap(wce_test, R = 5)

  out <- capture.output(summary(wce_test))
  expect_true(any(grepl("Partial log-likelihood", out)))
})
