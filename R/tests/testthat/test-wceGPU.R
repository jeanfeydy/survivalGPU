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


# snapshot
test_that("WCE - no covariates", {
  skip_on_ci()  # snapshot of GPU/CPU float output is environment-fragile; run locally only
  expect_snapshot({
    wceGPU(
      data = drugdata, nknots = 1, cutoff = 90, id = "Id",
      event = "Event", start = "Start", stop = "Stop",
      expos = "dose",
      constrained = FALSE, aic = FALSE, confint = 0.95,
      nbootstraps = 0, batchsize = 0
    )
  })
})

test_that("WCE - one covariate", {
  skip_on_ci()  # snapshot of GPU/CPU float output is environment-fragile; run locally only
  expect_snapshot({
    wceGPU(
      data = drugdata, nknots = 1, cutoff = 90, id = "Id",
      event = "Event", start = "Start", stop = "Stop",
      expos = "dose", covariates = c("age"),
      constrained = FALSE, aic = FALSE, confint = 0.95,
      nbootstraps = 0, batchsize = 0
    )
  })
})

test_that("WCE - two covariates", {
  skip_on_ci()  # snapshot of GPU/CPU float output is environment-fragile; run locally only
  expect_snapshot({
    wceGPU(
      data = drugdata, nknots = 1, cutoff = 90, id = "Id",
      event = "Event", start = "Start", stop = "Stop",
      expos = "dose", covariates = c("age","sex"),
      constrained = FALSE, aic = FALSE, confint = 0.95,
      nbootstraps = 0, batchsize = 0
    )
  })
})

test_that("WCE - AIC", {
  skip_on_ci()  # snapshot of GPU/CPU float output is environment-fragile; run locally only
  expect_snapshot({
    wceGPU(
      data = drugdata, nknots = 1, cutoff = 90, id = "Id",
      event = "Event", start = "Start", stop = "Stop",
      expos = "dose", covariates = c("age","sex"),
      constrained = FALSE, aic = TRUE, confint = 0.95,
      nbootstraps = 0, batchsize = 0
    )
  })
})

test_that("WCE - right constraint", {
  skip_on_ci()  # snapshot of GPU/CPU float output is environment-fragile; run locally only
  expect_snapshot({
    wceGPU(
      data = drugdata, nknots = 1, cutoff = 90, id = "Id",
      event = "Event", start = "Start", stop = "Stop",
      expos = "dose", covariates = c("age","sex"),
      constrained = "R", aic = FALSE, confint = 0.95,
      nbootstraps = 0, batchsize = 0
    )
  })
})

test_that("WCE - left constraint", {
  skip_on_ci()  # snapshot of GPU/CPU float output is environment-fragile; run locally only
  expect_snapshot({
    wceGPU(
      data = drugdata, nknots = 1, cutoff = 90, id = "Id",
      event = "Event", start = "Start", stop = "Stop",
      expos = "dose", covariates = c("age","sex"),
      constrained = "L", aic = FALSE, confint = 0.95,
      nbootstraps = 0, batchsize = 0
    )
  })
})

test_that("WCE - 3 knots", {
  skip_on_ci()  # snapshot of GPU/CPU float output is environment-fragile; run locally only
  expect_snapshot({
    wceGPU(
      data = drugdata, nknots = 3, cutoff = 90, id = "Id",
      event = "Event", start = "Start", stop = "Stop",
      expos = "dose", covariates = c("age","sex"),
      constrained = FALSE, aic = FALSE, confint = 0.95,
      nbootstraps = 0, batchsize = 0
    )
  })
})
