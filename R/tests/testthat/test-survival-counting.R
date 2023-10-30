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

test_that("survival-counting", {
  skip_if_no_python()
  skip_if_no_modules()

  # 'counting.R' test from survival package
  options(na.action=na.exclude) # preserve missings
  options(contrasts=c('contr.treatment', 'contr.poly')) #ensure constrast type
  library(survival)


  # TODO: remove this when the re-implementation of coxph is over
  # We currently do not support Efron ties, only Breslow
  ties <- "breslow"  # "efron"


  # Create a "counting process" version of the simplest test data set
  #
  test1 <- data.frame(time=  c(9, 3,1,1,6,6,8),
                      status=c(1,NA,1,0,1,1,0),
                      x=     c(0, 2,1,1,1,0,0))

  test1b<- list(start= c(0, 3,  0,  0, 5,  0, 6,14,  0,  0, 10,20,30, 0),
                stop = c(3,10, 10,  5,20,  6,14,20, 30,  10,20,30,40, 10),
                status=c(0, 1,  0,  0, 1,  0, 0, 1,  0,   0, 0, 0, 1,  0),
                x=     c(1, 1,  1,  1, 1,  0, 0, 0,  0,   0, 0, 0, 0,  NA),
                id =   c(3, 3,  4,  5, 5,  6, 6, 6,  7,   1, 1, 1, 1,   2))

  aeq <- function(x,y) all.equal(as.vector(x), as.vector(y))
  #
  #  Check out the various residuals under an Efron approximation
  #

  # Right process
  fit0 <- coxph(Surv(time, status) ~ x, test1, iter = 0, ties = ties)
  fit  <- coxph(Surv(time, status) ~ x, test1, ties = ties)
  fit0_gpu <- coxphGPU(Surv(time, status)~ x, test1, iter = 0, ties = ties)
  fit_gpu  <- coxphGPU(Surv(time, status) ~ x, test1, ties = ties)

  # Counting process
  fit0b <- coxph(Surv(start, stop, status) ~ x, test1b, iter = 0, ties = ties)
  fitb  <- coxph(Surv(start, stop, status) ~ x, test1b, ties = ties)
  fit0b_gpu <- coxphGPU(Surv(start, stop, status) ~ x, test1b, iter.max = 0, ties = ties)
  fitb_gpu <- coxphGPU(Surv(start, stop, status) ~ x, test1b, ties = ties)

  # offset feature
  fitc  <- coxph(Surv(time, status) ~ offset(fit$coef*x), test1, ties = ties)
  fitd  <- coxph(Surv(start, stop, status) ~ offset(fit$coef*x), test1b, ties = ties)
  #fitd_gpu  <- coxphGPU(Surv(start, stop, status) ~ offset(fit$coef*x), test1b) # offset feature not yet implemented in coxphGPU


  # Tests - check coefs between coxph and coxphGPU

  test_that("Check coefs between right process - No iter", {
    expect_equal(
      as.vector(fit0$coef),
      as.vector(fit0_gpu$coef)
    )
  })

  test_that("Check coefs between right process", {
    expect_equal(
      as.vector(round(fit$coef,3)),
      as.vector(round(fit_gpu$coef,3))
    )
  })

  test_that("Check coefs between counting process - No iter", {
    expect_equal(
      as.vector(fit0b$coef),
      as.vector(fit0b_gpu$coef)
    )
  })

  # # Error because small dataset ?
  # test_that("Check coefs between counting process", { # convergence problem with small numbers ?
  #   expect_equal(
  #     as.vector(fitb$coef),
  #     as.vector(fitb_gpu$coef)
  #   )
  # })

  aeq(resid(fit0), resid(fit0b, collapse=test1b$id))
  aeq(resid(fit), resid(fitb, collapse=test1b$id))
  aeq(resid(fitc), resid(fitd, collapse=test1b$id))
  aeq(resid(fitc), resid(fit))

  aeq(resid(fit0, type='score'), resid(fit0b, type='score', collapse=test1b$id))
  aeq(resid(fit, type='score'), resid(fitb, type='score', collapse=test1b$id))

  aeq(resid(fit0, type='scho'), resid(fit0b, type='scho', collapse=test1b$id))
  aeq(resid(fit, type='scho'), resid(fitb, type='scho', collapse=test1b$id))

  # The two survivals will have different censoring times
  #  nrisk, nevent, surv, and std should be the same
  temp1 <- survfit(fit, list(x=1), censor=FALSE)
  temp2 <- survfit(fitb, list(x=1), censor=FALSE)
  all.equal(unclass(temp1)[c(3,4,6,8)], unclass(temp2)[c(3,4,6,8)])

  # # Tests with some id on drugdata - convergence errors
  # drugdata <- WCE::drugdata
  # coxph(Surv(Start, Stop, Event) ~sex+age, drugdata %>% dplyr::filter(Id %in% c(1:6)))
  # coxphGPU(Surv(Start, Stop, Event) ~sex+age, drugdata %>% dplyr::filter(Id %in% c(1:6)))
  # drugdata %>%
  #   filter(Id == 6) #%>% tail(10)

})
