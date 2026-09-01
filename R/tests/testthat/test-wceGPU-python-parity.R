testthat::skip_if_not(reticulate::py_module_available("survivalgpu"))
skip_if_no_pykeops()

# This file directly compares the R package's own wceGPU()/bootstrap()
# pipeline against calling the underlying Python survivalgpu function
# (wce_R) via reticulate with independently, manually constructed inputs --
# i.e. it tests the correctness of the R-to-Python bridge/glue code itself,
# as opposed to the other tests in this directory, which validate numerical
# correctness against the reference WCE::WCE() implementation. See
# test-coxphGPU-python-parity.R for the analogous coxphGPU()/bootstrap()
# comparison.

drugdata <- WCE::drugdata
survivalgpu <- reticulate::import("survivalgpu")
torch <- reticulate::import("torch")

R_reps <- 20
seed <- 4242L

# --- Path 1: the R package's own wceGPU() + bootstrap() ---------------------

torch$manual_seed(seed)
fit <- wceGPU(
  data = drugdata, nknots = 1, cutoff = 90, id = "Id", event = "Event",
  start = "Start", stop = "Stop", expos = "dose", covariates = c("age", "sex")
)
fit_boot <- bootstrap(fit, R = R_reps, data = drugdata, batchsize = 10)

# --- Path 2: a hand-built, direct call into the Python backend --------------
# Independently reconstructs what wceGPU()/bootstrap() should be sending,
# without reusing any of the package's own R helper functions
# (.wceGPU_call_python).

torch$manual_seed(seed)
direct <- survivalgpu$wce_R(
  data = drugdata,
  ids = "Id",
  covars = c("age", "sex"),
  start = "Start",
  stop = "Stop",
  doses = "dose",
  events = "Event",
  cutoff = 90L,
  nknots = 1L,
  constrained = "None",
  aic = FALSE,
  bootstrap = R_reps,
  batchsize = 10L,
  init = as.numeric(fit$total_covariates), # matches bootstrap()'s default warm start
  device = NULL,
  double_precision = TRUE
)

test_that("wceGPU()'s point estimate matches a direct, hand-built Python call", {
  expect_equal(as.numeric(fit$beta.hat.covariates), as.numeric(direct$coef), tolerance = 1e-8)
  expect_equal(as.numeric(fit$loglik), as.numeric(direct$loglik), tolerance = 1e-8)
})

test_that("bootstrap()'s replicate coefficients match a direct Python call given the same torch seed", {
  expect_equal(
    as.numeric(fit_boot$bootstrap_beta.hat.covariates),
    as.numeric(drop(direct$bootstrap_coef)),
    tolerance = 1e-6
  )
})
