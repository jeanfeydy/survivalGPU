survivalGPU
============

**GPU-accelerated survival analysis** — Cox Proportional Hazards (CoxPH) and
Weighted Cumulative Exposure (WCE) models, built on `PyTorch
<https://pytorch.org>`_ (with `KeOps <https://www.kernel-operations.io>`_
additionally powering WCE). Classical survival models don't scale well to
large datasets or to heavy bootstrap resampling on CPU; survivalGPU runs the
core computations on the GPU to make both practical, while still working
without one (CPU fallback).

Two interfaces, one implementation
-----------------------------------

survivalGPU is usable from both **Python** and **R**, and both give you the
same models with the same results — there's a single canonical
implementation (in Python), not two separate codebases to keep in sync.

.. code-block:: python

   import numpy as np
   from survivalgpu import CoxPHSurvivalAnalysis

   stop = np.array([1, 1, 2], dtype=np.int64)
   event = np.array([0, 1, 1], dtype=np.int64)
   covariates = np.array([[1.0], [0.0], [4.0]])

   model = CoxPHSurvivalAnalysis(ties="efron")
   model.fit(covariates, stop, event=event)

   print(model.coef_)

.. code-block:: r

   library(survivalGPU)
   library(survival)

   data <- lung
   data$status <- lung$status - 1  # recode to 0/1

   fit <- coxphGPU(Surv(time, status) ~ age + sex, data = data)
   summary(fit)

``coxphGPU()``/``wceGPU()`` are built as drop-in companions to
`survival::coxph() <https://cran.r-project.org/package=survival>`_ and
`WCE::WCE() <https://cran.r-project.org/package=WCE>`_ respectively, so R
users familiar with those functions should feel at home.

The reticulate bridge
----------------------

The R package doesn't reimplement the models — ``coxphGPU()`` and
``wceGPU()`` are thin R wrappers that call directly into the same Python
classes shown above (``CoxPHSurvivalAnalysis``, ``WCESurvivalAnalysis``), via
`reticulate <https://rstudio.github.io/reticulate/>`_. Concretely: the R
package's source tree bundles the actual ``survivalgpu`` Python package (as a
copy of the same code, not a separate reimplementation), and on load, points
reticulate at it with a *delayed* import — so loading the R package itself
never requires Python, only actually *calling* ``coxphGPU()``/``wceGPU()``
does.

In practice, this means:

- A fix or new feature in the Python implementation is immediately available
  from R too, with no separate porting step.
- The R side needs a working Python environment (PyTorch, + KeOps for WCE)
  set up once — see :doc:`r/installation` — but nothing else Python-specific
  to think about afterwards.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   python/index
   r/index
