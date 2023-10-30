skip_if_no_python <- function() {
  have_python <- reticulate::py_available(initialize = TRUE)
  if(!have_python) testthat::skip("Python not available on system for testing")
}

skip_if_no_modules <- function() {
  survivalgpu_python_dep <- c("torch", "torch_scatter",
                              "pykeops", "matplotlib",
                              "beartype", "jaxtyping")

  py_module <- lapply(survivalgpu_python_dep, reticulate::py_module_available)
  if(any(py_module == FALSE)){
    testthat::skip("one or more modules not available for testing")
  }
}

test_that("survival-wceGPU", {
  skip_if_no_python()
  skip_if_no_modules()


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
      round(as.vector(wce$WCEmat), 1),
      round(as.vector(wce_gpu$WCEmat), 1)
    )
  })

  # Check coefs between WCE GPU and original WCE
  test_that("coef covariates", {
    expect_equal(
      round(as.vector(wce$beta.hat.covariates), 1),
      round(as.vector(wce_gpu$coef[, wce_gpu$covariates]), 1)
    )
  })

  # Check SE between WCE GPU and original WCE
  test_that("SE covariates", {
    expect_equal(
      round(as.vector(wce$se.covariates), 1),
      round(as.vector(wce_gpu$SE[, wce_gpu$covariates]), 1)
    )
  })

  # Check covariance matrix between WCE GPU and original WCE
  test_that("Vcovmat", {
    expect_equal(
      round(wce$vcovmat[[1]], 2),
      round(wce_gpu$vcovmat[[1]], 2)
    )
  })

  # Check ll between WCE GPU and original WCE
  test_that("Partial ll", {
    expect_equal(
      round(as.vector(wce$loglik), 1),
      round(as.vector(wce_gpu$loglik), 1)
    )
  })

  # Check AIC/BIC between WCE GPU and original WCE
  test_that("info.criterion", {
    expect_equal(
      round(as.numeric(wce$info.criterion), 0),
      round(as.numeric(wce_gpu$info.criterion), 0)
    )
  })


  # Check HR between WCE GPU and original WCE
  exposed   <- rep(1, 90)
  unexposed <- rep(0, 90)

  test_that("HR", {
    expect_equal(
      round(HR(wce_gpu_bootstrap, exposed, unexposed)[1], 1),
      round(WCE::HR.WCE(wce, exposed, unexposed)[1], 1)
    )
  })


  # snapshot
  test_that("WCE - no covariates", {
    expect_snapshot({
      wceGPU(
        data = drugdata, nknots = 1, cutoff = 90, id = "Id",
        event = "Event", start = "Start", stop = "Stop",
        expos = "dose",
        constrained = FALSE, aic = FALSE, confint = 0.95,
        nbootstraps = 1, batchsize = 0
      )
    })
  })

  test_that("WCE - one covariate", {
    expect_snapshot({
      wceGPU(
        data = drugdata, nknots = 1, cutoff = 90, id = "Id",
        event = "Event", start = "Start", stop = "Stop",
        expos = "dose", covariates = c("age"),
        constrained = FALSE, aic = FALSE, confint = 0.95,
        nbootstraps = 1, batchsize = 0
      )
    })
  })

  test_that("WCE - two covariates", {
    expect_snapshot({
      wceGPU(
        data = drugdata, nknots = 1, cutoff = 90, id = "Id",
        event = "Event", start = "Start", stop = "Stop",
        expos = "dose", covariates = c("age","sex"),
        constrained = FALSE, aic = FALSE, confint = 0.95,
        nbootstraps = 1, batchsize = 0
      )
    })
  })

  test_that("WCE - AIC", {
    expect_snapshot({
      wceGPU(
        data = drugdata, nknots = 1, cutoff = 90, id = "Id",
        event = "Event", start = "Start", stop = "Stop",
        expos = "dose", covariates = c("age","sex"),
        constrained = FALSE, aic = TRUE, confint = 0.95,
        nbootstraps = 1, batchsize = 0
      )
    })
  })

  test_that("WCE - right constraint", {
    expect_snapshot({
      wceGPU(
        data = drugdata, nknots = 1, cutoff = 90, id = "Id",
        event = "Event", start = "Start", stop = "Stop",
        expos = "dose", covariates = c("age","sex"),
        constrained = "R", aic = FALSE, confint = 0.95,
        nbootstraps = 1, batchsize = 0
      )
    })
  })

  test_that("WCE - left constraint", {
    expect_snapshot({
      wceGPU(
        data = drugdata, nknots = 1, cutoff = 90, id = "Id",
        event = "Event", start = "Start", stop = "Stop",
        expos = "dose", covariates = c("age","sex"),
        constrained = "L", aic = FALSE, confint = 0.95,
        nbootstraps = 1, batchsize = 0
      )
    })
  })

  test_that("WCE - 3 knots", {
    expect_snapshot({
      wceGPU(
        data = drugdata, nknots = 3, cutoff = 90, id = "Id",
        event = "Event", start = "Start", stop = "Stop",
        expos = "dose", covariates = c("age","sex"),
        constrained = FALSE, aic = FALSE, confint = 0.95,
        nbootstraps = 1, batchsize = 0
      )
    })
  })

})
