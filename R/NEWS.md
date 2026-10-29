# survivalGPU 0.1.0

* Initial CRAN submission.
* `coxphGPU()`: Cox proportional hazards regression, GPU-accelerated,
  with an interface modeled on `survival::coxph()`.
* `wceGPU()`: Weighted Cumulative Exposure (WCE) models, GPU-accelerated,
  with an interface modeled on `WCE::WCE()`.
* `bootstrap()`: adds bootstrap confidence intervals to a model fitted by
  `coxphGPU()` or `wceGPU()`, with patient-level resampling processed in
  batches on the GPU. The model must be fitted with `x = TRUE`, plus
  `patient_id` for `coxphGPU()`, so that the bootstrap resamples exactly the
  data the model was fitted on.
* `wceGPU()` accepts several candidate numbers of knots (e.g.
  `nknots = 1:3`) and selects the best one by AIC or BIC; `bootstrap()`
  repeats this selection in every replicate, so that the confidence
  intervals account for the uncertainty on the number of knots.
* `use_cuda()` to check whether CUDA drivers are detected; both models
  fall back to CPU automatically when they are not.
* The formula, model frame and martingale residuals of `coxphGPU()` are
  computed by `survival::coxph()` itself: the package has no compiled
  code.
