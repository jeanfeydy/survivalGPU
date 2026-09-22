# survivalGPU 0.1.0

* Initial CRAN submission.
* `coxphGPU()`: Cox proportional hazards regression, GPU-accelerated,
  with an interface modeled on `survival::coxph()`.
* `wceGPU()`: Weighted Cumulative Exposure (WCE) models, GPU-accelerated,
  with an interface modeled on `WCE::WCE()`.
* Bootstrap resampling support for confidence intervals in both models.
* `use_cuda()` to check whether CUDA drivers are detected; both models
  fall back to CPU automatically when they are not.
