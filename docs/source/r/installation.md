# Installation

survivalGPU is an R package built on top of a Python package, via the
[reticulate](https://rstudio.github.io/reticulate/) R package. To use it, you
need some Python libraries — such as `torch` — installed in your Python
environment. `pykeops` is only required if you want to use `wceGPU()` (the
WCE model); it is not available on Windows, so `coxphGPU()` works without it.

You may already have a suitable Python installation, but it's best to create
a dedicated virtualenv for this project. Here are the steps to configure a
Python virtual environment for `survivalGPU` in R.

```r
library(reticulate)

# Create the 'survivalGPU' environment:
virtualenv_create("survivalGPU")
virtualenv_list()
# You can check the Python libraries in this environment:
py_list_packages(envname = "survivalGPU")  # numpy by default

# Now, install all the survivalgpu Python dependencies.
# Add "pykeops" to this list if you need wceGPU() (not available on Windows):
virtualenv_install("survivalGPU", packages = c("torch", "matplotlib",
                                               "beartype", "jaxtyping"))
py_list_packages(envname = "survivalGPU")
```

```r
#  /!\ Restart your R session  /!\
library(reticulate)

# Check that it's the correct Python executable:
py_discover_config()
use_virtualenv(virtualenv = "survivalGPU")
py_config()
```

You're now connected to your `survivalGPU` Python environment. If you load
the `survivalGPU` R package after running
`reticulate::use_virtualenv(virtualenv = "survivalGPU")`, the package will use
this Python environment.

If you don't have the `reticulate` R package installed, install it with
`install.packages("reticulate")`. During installation, R may offer to install
Miniconda — if you accept, be careful, since your Python executable and
Python home will change to that new Miniconda environment. You can change
your Python environment again in a new R session.

See the [reticulate documentation](https://rstudio.github.io/reticulate/articles/versions.html)
for other ways to manage your Python configuration.
