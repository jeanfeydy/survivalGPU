# Dataset
drugdata <- WCE::drugdata

# WCE models -------
# WCE GPU
wce_gpu <- wceGPU(
  data = drugdata, nknots = 1, cutoff = 90, id = "Id",
  event = "Event", start = "Start", stop = "Stop",
  expos = "dose", covariates = c("age", "sex"),
  constrained = FALSE, aic = FALSE, confint = 0.95,
  nbootstraps = 1, batchsize = 0
)

wce_gpu_bootstrap <- wceGPU(
  data = drugdata, nknots = 1, cutoff = 90, id = "Id",
  event = "Event", start = "Start", stop = "Stop",
  expos = "dose", covariates = c("age", "sex"),
  constrained = FALSE, aic = FALSE, confint = 0.95,
  nbootstraps = 15, batchsize = 0
)

# Original WCE
wce <- WCE::WCE(
  data = drugdata, analysis = "Cox", nknots = 1, cutoff = 90,
  id = "Id", event = "Event", start = "Start", stop = "Stop",
  expos = "dose", covariates = c("age", "sex"),
  constrained = FALSE, aic = FALSE
)


# Tests
# Check weight function between WCE GPU and original WCE
test_that("WCE mat", {
  expect_equal(
    round(as.vector(wce$WCEmat), 6),
    round(as.vector(wce_gpu$WCEmat), 6)
  )
})

# Check coefs between WCE GPU and original WCE
test_that("coef covariates", {
  expect_equal(
    round(as.vector(wce$beta.hat.covariates), 3),
    round(as.vector(wce_gpu$coef[, wce_gpu$covariates]), 3)
  )
})

# Check SE between WCE GPU and original WCE
test_that("SE covariates", {
  expect_equal(
    round(as.vector(wce$se.covariates), 3),
    round(as.vector(wce_gpu$SE[, wce_gpu$covariates]), 3)
  )
})

# Check covariance matrix between WCE GPU and original WCE
test_that("Vcovmat", {
  expect_equal(
    round(wce$vcovmat[[1]], 6),
    round(wce_gpu$vcovmat[[1]], 6)
  )
})

# Check ll between WCE GPU and original WCE
test_that("Partial ll", {
  expect_equal(
    round(as.vector(wce$loglik), 3),
    round(as.vector(wce_gpu$loglik), 3)
  )
})

# Check AIC/BIC between WCE GPU and original WCE
test_that("info.criterion", {
  expect_equal(
    round(as.numeric(wce$info.criterion), 0),
    round(as.numeric(wce_gpu$info.criterion), 0)
  )
})
