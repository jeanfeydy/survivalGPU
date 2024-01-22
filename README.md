# survivalGPU

<!-- badges: start -->
[![R-CMD-check](https://github.com/jeanfeydy/survivalGPU/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/jeanfeydy/survivalGPU/actions/workflows/R-CMD-check.yaml)
[![Build Status](https://github.com/jeanfeydy/survivalGPU/actions/workflows/python-package.yml/badge.svg?branch=refactor_objects&event=push)](https://github.com/jeanfeydy/survivalGPU/actions)
[![Codecov Status](https://codecov.io/gh/jeanfeydy/survivalGPU/branch/refactor_objects/graph/badge.svg)](https://codecov.io/gh/jeanfeydy/survivalGPU)
<!-- badges: end -->

Fast implementation of survival analysis models (CoxPH, WCE...) with GPU support, for R and Python.
Please note that this package is still little more than a proof of concept: we are working to publish a first stable version by the summer of 2023. We have opened the code to get a first feed back from the community, but stress that our solver has not yet been tested thoroughly. The user interface of the Python and R packages are also likely to change over the next few months.

If you find this work useful, please cite:


*Accélération des calculs à l'aide de cartes graphiques pour la détection de signaux de pharmacovigilance sur le Système national des données de santé : le package survivalGPU*, A. Van Straaten, P. Sabatier, J. Feydy, A-S. Jannot, Revue d'Épidémiologie et de Santé Publique, Volume 71, Supplement 1, 2023, 101467, ISSN 0398-7620, https://doi.org/10.1016/j.respe.2023.101467.



## Run tests

For the Python `survivalgpu` package, go to the `survivalGPU/python` folder and run `pytest .`

For the R `survivalGPU` package, go to the `survivalGPU/R` folder. Then, launch an R interactive session and run:

```R
library(devtools)
load_all()
test()

# To render the documentation as a static website:
pkgdown::build_site()
```
