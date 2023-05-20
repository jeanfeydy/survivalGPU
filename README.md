# survivalGPU

[![Build Status](https://github.com/jeanfeydy/survivalGPU/actions/workflows/python-package.yml/badge.svg?branch=refactor_objects&event=push)](https://github.com/jeanfeydy/survivalGPU/actions)
[![Codecov Status](https://codecov.io/gh/jeanfeydy/survivalGPU/branch/refactor_objects/graph/badge.svg)](https://codecov.io/gh/jeanfeydy/survivalGPU)


Fast implementation of survival analysis models (CoxPH, WCE...) with GPU support, for R and Python.


## Run tests

For the Python `survivalgpu` package, go to the `survivalGPU/python` folder and run `pytest .`.

For the R `survivalGPU` package, go to the `survivalGPU/R` folder. Then, launch an R interactive session and run:

```R
library(devtools)
load_all()
test()
```
