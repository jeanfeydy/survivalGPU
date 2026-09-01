testthat::skip_if_not(reticulate::py_module_available("survivalgpu"))

# This file directly compares the R package's own coxphGPU()/bootstrap()
# pipeline against calling the underlying Python survivalgpu functions via
# reticulate with independently, manually constructed inputs -- i.e. it
# tests the correctness of the R-to-Python bridge/glue code itself (argument
# marshalling, data alignment, dtype handling), as opposed to the other
# tests in this directory, which validate numerical correctness against
# survival::coxph(). If a future change to coxphGPU.R's internals silently
# scrambles what gets sent to Python (wrong column, wrong dtype, misaligned
# rows) while still producing plausible-looking output, this is the test
# meant to catch it.

drugdata <- WCE::drugdata
survivalgpu <- reticulate::import("survivalgpu")
torch <- reticulate::import("torch")

R_reps <- 25
seed <- 2024L

# --- Path 1: the R package's own coxphGPU() + bootstrap() -------------------

torch$manual_seed(seed)
fit <- coxphGPU(Surv(Start, Stop, Event) ~ sex + age, data = drugdata, x = TRUE)
fit_boot <- bootstrap(
  fit, R = R_reps, patient_id = "Id", data = drugdata, batchsize = 0
)

# --- Path 2: a hand-built, direct call into the Python backend --------------
# Independently reconstructs what coxphGPU()/bootstrap() should be sending,
# without reusing any of the package's own R helper functions
# (.coxphGPU_build_data_Y / .coxphGPU_call_python).

data_X <- as.matrix(drugdata[, c("sex", "age")])
storage.mode(data_X) <- "double"

data_Y <- data.table::data.table(
  Start = as.integer(drugdata$Start),
  Stop = as.integer(drugdata$Stop),
  Event = as.integer(drugdata$Event),
  Id = as.integer(drugdata$Id)
)

torch$manual_seed(seed)
direct <- survivalgpu$coxph_R(
  data_X = data_X,
  data_Y = data_Y,
  start = "Start",
  stop = "Stop",
  death = "Event",
  covars = c("sex", "age"),
  ties = "efron",
  strata = NULL,
  patient_id = "Id",
  bootstrap = R_reps,
  batchsize = 0,
  maxiter = 20,
  init = as.numeric(fit$coefficients), # matches bootstrap()'s default warm start
  device = NULL,
  double_precision = TRUE
)

test_that("coxphGPU()'s point estimate matches a direct, hand-built Python call", {
  expect_equal(unname(fit$coefficients), as.numeric(direct$coef), tolerance = 1e-8)
  expect_equal(fit$loglik[2], as.numeric(direct$loglik), tolerance = 1e-8)
})

test_that("bootstrap()'s replicate coefficients match a direct Python call given the same torch seed", {
  expect_equal(
    unname(fit_boot$coef_bootstrap),
    matrix(direct$bootstrap_coef, ncol = 2),
    tolerance = 1e-6
  )
})
