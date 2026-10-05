Python overview
================

The ``survivalgpu`` Python package performs survival analysis on
GPU-accelerated hardware, with `PyTorch <https://pytorch.org>`_. Currently,
two models are implemented, with an API loosely based on scikit-learn: build
the estimator, call ``.fit()``, then read the results off the fitted object's
attributes (all of which end with an underscore).

- :class:`~survivalgpu.CoxPHSurvivalAnalysis` — Cox Proportional Hazards, on
  ``(start, stop]`` intervals with fixed or time-varying covariates, with the
  Efron or Breslow handling of ties. Only requires PyTorch, and works on
  Linux, macOS and Windows.
- :class:`~survivalgpu.WCESurvivalAnalysis` — Weighted Cumulative Exposure
  (WCE), which models the cumulative effect of a time-varying exposure with
  B-splines. It also relies on `KeOps <https://www.kernel-operations.io>`_
  (the ``pykeops`` package), which is not available natively on Windows:
  use it through `WSL2 <https://learn.microsoft.com/windows/wsl/install>`_
  instead (see :doc:`installation`).

Both support bootstrap resampling, processed in parallel batches on the
device: this is where a GPU pays off most. For WCE models, the number of
knots of the splines can be selected by AIC or BIC among several candidates,
again in every bootstrap replicate. Both models also run on CPU if no
CUDA-capable GPU is available (see :func:`~survivalgpu.use_cuda`).

The package also provides simulation tools, such as
:func:`~survivalgpu.simulate_dataset`, to generate datasets with known
effects for constant, time-dependent and WCE covariates.

This is the same implementation that the ``survivalGPU`` R package calls
through `reticulate <https://rstudio.github.io/reticulate/>`_, so both
interfaces give the same results.
coxphGPU_bootstrap
Where to go next:

- :doc:`installation` — install the package, with or without the WCE model.
- :doc:`user_guide/index` — worked examples for both models.
- :doc:`api` — the full API reference.
