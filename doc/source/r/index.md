# R

The `survivalGPU` R package performs survival analysis on GPU-accelerated
hardware. Currently, two models are implemented:

- **`coxphGPU()`** — Cox Proportional Hazards, built to take the same inputs
  as `survival::coxph()`.
- **`wceGPU()`** — Weighted Cumulative Exposure (WCE), built to take the same
  inputs as `WCE::WCE()`.

Both support bootstrap resampling, and both run on CPU if no CUDA-capable GPU
is available. Under the hood, the R package delegates its computations to the
same Python backend as the `survivalgpu` Python package, via
[reticulate](https://rstudio.github.io/reticulate/) — see
{doc}`installation` to set that up.

```{toctree}
:maxdepth: 2

installation
user_guide/index
```
