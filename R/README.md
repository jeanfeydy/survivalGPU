
<!-- README.md is generated from README.Rmd. Please edit that file -->

# survivalGPU <img src="man/figures/logo.png" align="right" height="139" />

<!-- badges: start -->
<!-- badges: end -->

The survivalGPU library allows you to perform survival analyzes using
the resources of Graphic Processing Units (GPU) in order to accelerate
the speed of calculations. Currently, two models have been implemented :

- `coxphGPU()` for Cox[^1] [^2] model  
- `wceGPU()` for Weighted Cumulative Exposure[^3] model

It’s also possible to use the library without having Graphics Processing
Units (with CPU).

## Installation

### Requirements

Python packages :

- pytorch
- pytorch-scatter
- pykeops (for WCE model)

Actually, survivalGPU is not available for Windows.

survivalGPU require submodules : you can install the development version
of survivalGPU from [GitHub](https://github.com/) with
`install_git_with_submodule()`:

``` r
# install.packages("devtools")

install_git_with_submodule <- function(x, subdir) {
  install_dir <- tempfile()
  system(paste("git clone --recursive", shQuote(x), shQuote(install_dir)))
  
  # change name for windows install
  file.rename(file.path(install_dir,"R/inst/python/survivalgpu"),
              file.path(install_dir,"R/inst/python/survivalgpu_submodule"))
  file.copy(file.path(install_dir,"python/survivalgpu"),
            file.path(install_dir,"R/inst/python"), recursive=TRUE)
  
  devtools::install(file.path(file.path(install_dir,subdir)))
}

install_git_with_submodule("https://github.com/jeanfeydy/survivalGPU",
                           subdir="R")
```

> **Warning**: survivalGPU is a package dependant of python, and it’s
> necessary to have installed the `reticulate` R package. To manage your
> python or miniconda configuration, check vignette(“python_connect”).

## Examples

Let’s make an example with `drugdata` dataset from WCE package.

``` r
library(survivalGPU)
#> Please run `use_cuda()` to check CUDA drivers
library(survival)
library(WCE)
data(drugdata)
```

By default, functions run with GPU if detected. Then we specify the
number of bootstrap, and consequently the batchsize argument, according
to CUDA drivers detection.

``` r
if (use_cuda()) {
  n_bootstrap <- 1000
  batchsize <- 200
} else {
  n_bootstrap <- 50
  batchsize <- 10
}
```

### Cox

You can realize the Cox model with the `coxphGPU` function, which is
written in the same way as the `survival::coxph` function from survival
package, with a Surv object in the formula, containing Start, Stop and
Event variables.

``` r
coxphGPU_bootstrap <- coxphGPU(Surv(Start, Stop, Event) ~ sex + age,
  data = drugdata,
  bootstrap = n_bootstrap,
  batchsize = batchsize,
  ties = "efron"
)
```

You obtain with `summary` all results for initial model, and a
confidence interval for estimated coefficients with bootstrap.

``` r
summary(coxphGPU_bootstrap)
#> Call:
#> coxphGPU.default(formula = Surv(Start, Stop, Event) ~ sex + age, 
#>     data = drugdata, ties = "efron", bootstrap = n_bootstrap, 
#>     batchsize = batchsize)
#> 
#>   n= 77038, number of events= 383 
#> 
#>         coef exp(coef) se(coef)     z Pr(>|z|)    
#> sex 0.619201  1.857443 0.117770 5.258 1.46e-07 ***
#> age 0.010671  1.010728 0.003963 2.692  0.00709 ** 
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#>     exp(coef) exp(-coef) lower .95 upper .95
#> sex     1.857     0.5384     1.475     2.340
#> age     1.011     0.9894     1.003     1.019
#> 
#> Concordance= 0.59  (se = 0.018 )
#> Likelihood ratio test= 33.73  on 2 df,   p=5e-08
#> Wald test            = 36.1  on 2 df,   p=1e-08
#> Score (logrank) test = 37.09  on 2 df,   p=9e-09
#> 
#>  ---------------- 
#> Confidence interval with 50 bootstraps for exp(coef), conf.level = 0.95 :
#>        2.5%   97.5%
#> sex 1.48769 2.35963
#> age 1.00234 1.01867
```

#### Test the Proportional Hazards Assumption

You can test the proportional hazards assumption for a Cox regression
with `survival::cox.zph()`.

``` r
cox.zph(coxphGPU_bootstrap)
#>          chisq df    p
#> sex    0.00319  1 0.95
#> age    0.63109  1 0.43
#> GLOBAL 0.64378  2 0.72
```

Plot scaled Schoenfeld residuals against the time for each covariates
with `ggcoxzph()` function from `survminer` package.

``` r
survminer::ggcoxzph(cox.zph(coxphGPU_bootstrap))
```

<img src="man/figures/README-unnamed-chunk-6-1.png" width="70%" />

#### Forestplot

Also with the `survminer` package, you can use `ggforest()` function to
plot forestplot for Cox PH model. The confidence interval plotted is the
confidence interval by the normal distribution (not bootstrap)

``` r
survminer::ggforest(model = coxphGPU_bootstrap,
                    data = drugdata)
```

<img src="man/figures/README-unnamed-chunk-7-1.png" width="70%" />

#### Survival Curves

Is it possible to check Kaplan-Meier survival estimates with
`survival::survfit()` function. After you can use
`survminer::ggsurvplot()` to plot a survival curve.

### WCE

WCE allows modeling the cumulative effects of time-varying exposures,
weighted according to their relative proximity in time and represented
by time-dependent covariates. Currently, the weight function is
estimated by the Cox proportional hazards model. To build a model, you
can use the `wceGPU` function in the same way as the `WCE::WCE` function
from WCE package.

``` r
wce_gpu_bootstrap <- wceGPU(
  data = drugdata, nknots = 1, cutoff = 90, id = "Id",
  event = "Event", start = "Start", stop = "Stop",
  expos = "dose", covariates = c("age", "sex"),
  constrained = FALSE, aic = FALSE, confint = 0.95,
  nbootstraps = n_bootstrap, batchsize = batchsize
)
```

In the summary, there are estimated coefficients for the covariates with
his confidence interval, calculated with var-covariance matrix, and
information to see the significance of the covariates. It’s also
possible to have confidence interval with bootstrap.

``` r
summary(wce_gpu_bootstrap)
```

The risk function can be plot, and if you added bootstrap, confidence
band intervals will be visible.

``` r
plot(wce_gpu_bootstrap)
```

## References

[^1]: Andersen, P. and Gill, R. (1982). Cox’s regression model for
    counting processes, a large sample study. Annals of Statistics 10,
    1100-1120.

[^2]: Therneau, T., Grambsch, P., Modeling Survival Data: Extending the
    Cox Model. Springer-Verlag, 2000.

[^3]: Sylvestre MP, Abrahamowicz M. Flexible modeling of the cumulative
    effects of time-dependent exposures on the hazard. Stat Med. 2009
    Nov 30;28(27):3437-53.
