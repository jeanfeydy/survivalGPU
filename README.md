# survivalGPU

<!-- badges: start -->
[![R-CMD-check](https://github.com/jeanfeydy/survivalGPU/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/jeanfeydy/survivalGPU/actions/workflows/R-CMD-check.yaml)
[![Python tests](https://github.com/jeanfeydy/survivalGPU/actions/workflows/python-tests.yml/badge.svg)](https://github.com/jeanfeydy/survivalGPU/actions/workflows/python-tests.yml)
[![Codecov Status](https://codecov.io/gh/jeanfeydy/survivalGPU/graph/badge.svg)](https://codecov.io/gh/jeanfeydy/survivalGPU)
<!-- badges: end -->

**GPU-accelerated survival analysis** — Cox Proportional Hazards (CoxPH) and
Weighted Cumulative Exposure (WCE) models, built on [PyTorch](https://pytorch.org)
and [KeOps](https://www.kernel-operations.io). survivalGPU scales classical
survival models to large datasets and to heavy bootstrap resampling by running
the core computations on the GPU (with a CPU fallback). It's available as both
a **Python package** and an **R package**, sharing the same GPU backend.

This repository hosts both:

- **[`python/`](python/)** — the core implementation. See
  [python/README.md](python/README.md) for installation, quick start, and how
  to run the test suite. Available on PyPI: `pip install survivalgpu`.
- **[`R/`](R/)** — an R interface (`coxphGPU()`, `wceGPU()`) that calls into
  the same Python backend via [reticulate](https://rstudio.github.io/reticulate/),
  built as a drop-in companion to `survival::coxph()` and `WCE::WCE()`. See
  [R/README.md](R/README.md) for installation, quick start, and how to run the
  test suite. The R package is currently being prepared for CRAN submission;
  install the development version from GitHub in the meantime (see the R
  README for instructions).

## Citation

If you use survivalGPU in your research, please cite:

> *Accélération des calculs à l'aide de cartes graphiques pour la détection de
> signaux de pharmacovigilance sur le Système national des données de santé :
> le package survivalGPU*, A. Van Straaten, P. Sabatier, J. Feydy, A-S. Jannot,
> Revue d'Épidémiologie et de Santé Publique, Volume 71, Supplement 1, 2023,
> 101467, ISSN 0398-7620,
> [https://doi.org/10.1016/j.respe.2023.101467](https://doi.org/10.1016/j.respe.2023.101467).

## License

Distributed under the terms of the **LGPL-2.1-or-later** license. See
[LICENSE](LICENSE).
