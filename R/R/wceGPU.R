#' Fast Weighted Cumulative Exposure Model
#'
#' @description Fits a weighted cumulative exposure (WCE) model on the GPU (or
#'   the CPU). Only point estimates are computed: call [bootstrap()] on the
#'   result for bootstrap confidence intervals, and inspect it with
#'   `summary()` and `plot()`.
#'
#' @usage
#' wceGPU(data, nknots, cutoff, constrained = FALSE, aic = FALSE, id,
#'        event, start, stop, expos, covariates = NULL,
#'        confint = 0.95, x = FALSE, ...)
#'
#' @param data Data frame in counting-process (long) format, with one row per
#'   subject and time unit.
#' @param nknots Number of interior knots of the cubic spline that models the
#'   weight function. With several values (e.g. `1:3`), one model is fitted
#'   per value and the one with the smallest information criterion (see
#'   `aic`) is kept; `best.nknots` and the `*.grid` components of the result
#'   describe all the candidates.
#' @param cutoff Length, in time units, of the exposure history that affects
#'   the hazard: the weight function is estimated for lags `0` to
#'   `cutoff - 1` (the `cutoff` columns of `WCEmat`), and older exposures have
#'   no effect.
#' @param constrained Shape constraint on the weight function. `FALSE`
#'   (default): none. `"Right"` (or `"R"`): it decreases smoothly to zero at
#'   the oldest lag. `"Left"` (or `"L"`): it is zero at lag 0, so the current
#'   exposure has no effect.
#' @param aic Logical. Information criterion used to choose among the
#'   `nknots` candidates and reported in `info.criterion`: the AIC if `TRUE`,
#'   the BIC if `FALSE` (default). Both equal
#'   \eqn{-2 \log(PL) + k p}{-2 log(PL) + k p}, with \eqn{PL} the partial
#'   likelihood and \eqn{p} the number of estimated parameters; \eqn{k = 2}
#'   for the AIC and \eqn{k = \log(d)}{k = log(d)}, with \eqn{d} the number
#'   of events, for the BIC (Volinsky and Raftery, 2000).
#' @param id Name of the column of `data` that identifies the subjects.
#' @param event Name of the column of `data` with the event indicator, coded
#'   `1` (event) or `0` (no event).
#' @param start Name of the column of `data` with the start time of each
#'   interval (the `time` argument of [survival::Surv()]).
#' @param stop Name of the column of `data` with the end time of each interval
#'   (the `time2` argument of [survival::Surv()]).
#' @param expos Name of the column of `data` with the exposure.
#' @param covariates Character vector with the names of the columns of `data`
#'   to adjust for. `NULL` (default): no adjustment.
#' @param confint Level of the bootstrap confidence intervals computed by
#'   [bootstrap()] (default `0.95`).
#' @param x For `wceGPU()`: logical. If `TRUE`, the columns of `data` used by
#'   the model (`id`, `event`, `start`, `stop`, `expos` and `covariates`) are
#'   stored in the result, as `data`. Required by [bootstrap()], which
#'   resamples exactly the data the model was fitted on. For `plot()`: a
#'   `wceGPU` object.
#' @param ... For `wceGPU()`: `device` (`"cpu"` or `"cuda"`; default: the GPU
#'   if one is available) and `double_precision` (`TRUE`, the default, for
#'   64-bit floats; `FALSE` for 32-bit floats, faster on most GPUs). For the
#'   methods: not used.
#'
#' @details At each time \eqn{t}, the exposure enters the Cox model through
#'   its weighted cumulative exposure over the last `cutoff` time units,
#'   \deqn{WCE(t) = \sum_{u=0}^{cutoff-1} w(u) X(t-u),}{WCE(t) = sum over u = 0, ..., cutoff - 1 of w(u) X(t - u),}
#'   where \eqn{X(t-u)} is the exposure \eqn{u} time units before \eqn{t}.
#'   The weight function \eqn{w} is a cubic spline with `nknots` interior
#'   knots. \eqn{WCE(t)} is therefore a linear combination of spline-weighted
#'   exposure histories, which enter a Cox proportional hazards model as
#'   time-dependent covariates, together with `covariates`. Their estimated
#'   coefficients give \eqn{w}, returned in `WCEmat`.
#'
#' @references Sylvestre MP, Abrahamowicz M. Flexible modeling of the cumulative
#'   effects of time-dependent exposures on the hazard. Stat Med. 2009 Nov
#'   30;28(27):3437-53. \doi{10.1002/sim.3701}
#'
#'   Sylvestre M (2025). _WCE: Weighted Cumulative Exposure Models_. R package
#'   version 1.0.4. \doi{10.32614/CRAN.package.WCE}. The reference
#'   implementation of the method: `wceGPU()` follows its interface, and its
#'   results are tested against it.
#'
#'   Volinsky CT, Raftery AE (2000). Bayesian information criterion for
#'   censored survival models. Biometrics, 56(1), 256-262.
#'   \doi{10.1111/j.0006-341X.2000.00256.x}
#'
#' @return `wceGPU()` returns an object of class `wceGPU`: a list that describes
#'   the selected model, i.e. the candidate in `nknots` that minimizes the
#'   information criterion. Its main components are:
#'   * `WCEmat`: matrix with one row and `cutoff` columns, the estimated weight
#'   function.
#'   * `beta.hat.covariates`, `se.covariates`: one-row matrices with the
#'   estimated coefficients of the covariates and their standard errors.
#'   * `est`, `SED`: one-row matrices with the estimated spline coefficients
#'   and their standard errors.
#'   * `vcovmat`: list holding the variance-covariance matrix of all the
#'   coefficients (covariates, then spline coefficients).
#'   * `knotsmat`: one-row matrix with the knots of the spline basis.
#'   * `loglik`, `info.criterion`, `nevents`: partial log-likelihood,
#'   information criterion (the BIC, or the AIC if `aic = TRUE`) and number of
#'   events.
#'   * `best.nknots`: the selected number of interior knots. `nknots.grid`,
#'   `info.criterion.grid` and `loglik.grid` give the candidates and the
#'   information criterion and partial log-likelihood obtained for each.
#'   * `data`: with `x = TRUE`, the columns of `data` used by the model.
#'
#'   After [bootstrap()], the object also contains:
#'   * `WCEmat_bootstrap`, `bootstrap_beta.hat.covariates`: matrices with one
#'   row per bootstrap replicate, for the weight function and for the
#'   coefficients of the covariates.
#'   * `WCEmat_CI`, `coef_CI`: two-row matrices with their lower and upper
#'   percentile confidence limits, at level `confint`.
#'   * `bootstrap.best.nknots`: the number of knots selected in each replicate.
#'   * `bootstrap_est`, `est_CI`: the same for the spline coefficients, only
#'   when a single `nknots` value is given.
#'
#'   `summary()` and `plot()` are called for their side effects (printing the
#'   model, plotting the weight function) and return `NULL` invisibly.
#'   `confint()` returns a matrix with one row per covariate and two columns:
#'   the lower and upper confidence limits of its coefficient, based on the
#'   normal approximation.
#' @export
#'
#' @seealso [bootstrap()]
#'
#' @examples
#' \dontrun{
#' # Dataset
#' drugdata <- WCE::drugdata
#'
#' # WCE model, point estimate only
#' wce_gpu <- wceGPU(data = drugdata, nknots = 1, cutoff = 90, id = "Id",
#'                   event = "Event", start = "Start", stop = "Stop",
#'                   expos = "dose", covariates = c("age", "sex"),
#'                   constrained = FALSE, aic = FALSE, confint = 0.95,
#'                   x = TRUE)
#'
#' # Results
#' wce_gpu
#' summary(wce_gpu)
#'
#' # See estimated weight function
#' plot(wce_gpu)
#'
#' # Bootstrap-based inference, as a separate step (example with 20
#' # replicates, but normally R > 500)
#' wce_gpu_bootstrap <- bootstrap(wce_gpu, R = 20)
#'
#' # See confidence bands for the estimated weight function due to bootstrap
#' plot(wce_gpu_bootstrap)
#'
#' # All estimated coefficients in bootstrap
#' coef(wce_gpu_bootstrap)
#'
#' # Estimate a HR (Exposed at a dose vs. unexposed)
#' exposed   <- rep(1, 90)
#' unexposed <- rep(0, 90)
#'
#' HR(wce_gpu_bootstrap, exposed, unexposed)
#' }
wceGPU <- function(data, nknots, cutoff, constrained = FALSE, aic = FALSE, id,
                   event, start, stop, expos, covariates = NULL,
                   confint = 0.95, x = FALSE, ...) {
  UseMethod("wceGPU")
}


#' @return \code{NULL}
#' @noRd
#' @method wceGPU default
#' @exportS3Method wceGPU default
wceGPU.default <- function(data, nknots, cutoff, constrained = FALSE,
                           aic = FALSE, id, event, start, stop, expos,
                           covariates = NULL,
                           confint = 0.95, x = FALSE,
                           device = NULL, double_precision = TRUE, ...) {
  # survivalgpu <- use_survivalGPU()

  # Bootstrap is no longer requested at fit time. Without this check, the old
  # arguments would be silently swallowed by `...` and the fit returned
  # without any bootstrap.
  removed <- intersect(names(list(...)), c("nbootstraps", "batchsize"))
  if (length(removed)) {
    stop("wceGPU() no longer takes ", paste(removed, collapse = ", "),
         ": fit the model with x = TRUE, then call bootstrap(fit, R = ).",
         call. = FALSE)
  }

  wce <- .wceGPU_call_python(
    data = data, id = id, event = event, start = start, stop = stop,
    expos = expos, covariates = covariates, nknots = nknots,
    constrained = constrained, cutoff = cutoff, aic = aic,
    bootstrap = 0, batchsize = 0, init = NULL,
    device = device, double_precision = double_precision
  )

  model_data <- if (isTRUE(x)) {
    data[unique(c(id, event, start, stop, expos, covariates))]
  }


  # call WCE outputs and rename them to follow R WCE convention

  # best.nknots is the candidate (out of possibly several) that Python
  # already selected by information criterion; every field below describes
  # that selected model, not the raw candidate list in `nknots`.
  best.nknots <- as.numeric(wce$best_nknots)[1]

  knotsmat <- matrix(c(wce$knotsmat), nrow = 1)
  rownames(knotsmat) <- paste(best.nknots, "knot(s)")


  WCEmat <- wce$risk_function
  colnames(WCEmat) <- paste0("t", 1:cutoff)

  beta.hat.covariates <- wce$coef
  colnames(beta.hat.covariates) <- covariates

  se.covariates <- wce$std
  colnames(se.covariates) <- covariates

  est <- wce$WCE_coef
  colnames(est) <- paste0("D", 1:(ncol(est)))

  SED = wce$SED
  colnames(SED) <- paste0("D", 1:(ncol(SED)))

  total_covariates <- c(beta.hat.covariates, est)
  names(total_covariates) <- c(covariates, paste0("D", 1:(ncol(est))))



  loglik <- c(wce$loglik)

  vcovmat <- list()

  vcovmat_knot <- drop(wce$imat)
  cov <- c(covariates, paste0("D", 1:(ncol(est))))
  rownames(vcovmat_knot) <- cov
  colnames(vcovmat_knot) <- cov

  vcovmat[[paste(best.nknots, "knot(s)")]] <- vcovmat_knot



  names(data)[names(data) == event] <- "Event"
  nevents <- length(data$Event[data$Event == 1])

  info_criterion <- c(wce$info_criterion)




  # List to return
  results <- list(
    knotsmat = knotsmat,
    beta.hat.covariates = beta.hat.covariates,
    se.covariates = se.covariates,
    est = est,
    SED = SED,
    total_covariates = total_covariates,
    WCEmat = WCEmat,
    vcovmat = vcovmat,
    covariates = covariates,
    loglik = loglik,
    constrained = constrained,
    nevents = nevents,
    aic = aic,
    info.criterion = info_criterion,
    nknots = nknots,
    best.nknots = best.nknots,
    nknots.grid = as.vector(wce$nknots_grid),
    info.criterion.grid = as.vector(wce$info_criterion_grid),
    loglik.grid = as.vector(wce$loglik_grid),
    cutoff = cutoff,
    confint = confint,
    nbootstraps = 0, # no bootstrap at fit time; see bootstrap()
    is_bootstraps = FALSE,
    # Stored so bootstrap() can be called later without re-supplying every
    # column name.
    id = id,
    event = event,
    start = start,
    stop = stop,
    expos = expos
  )

  results$analysis <- "Cox"
  results$data <- model_data

  # wceGPU object
  class(results) <- "wceGPU"
  return(results)
}


## Internal helpers, shared between wceGPU.default() and bootstrap.wceGPU()
## ------------------------------------------------------------------------

#' @noRd
.wceGPU_call_python <- function(data, id, event, start, stop, expos,
                                covariates, nknots, cutoff, constrained, aic,
                                bootstrap, batchsize, init, device,
                                double_precision) {
  wce_R <- tryCatch(
    survivalgpu$wce_R,
    error = function(e) survivalgpu_unavailable_error(e, need_keops = TRUE)
  )

  # Minor changes for python inputs
  if (isFALSE(constrained)) {
    py_constrained <- "None"
  } else {
    py_constrained <- constrained
  }

  if (length(covariates) < 2) {
    py_covariates <- as.list(covariates)
  } else {
    py_covariates <- covariates
  }

  # wce_R is resolved lazily even when pykeops (required only for WCE, not
  # for coxphGPU) isn't installed, so the informative error only surfaces
  # here, at call time, rather than at attribute-fetch time above.
  tryCatch(
    wce_R(
      data = data, ids = id, covars = py_covariates, start = start, stop = stop,
      doses = expos, events = event, nknots = nknots,
      constrained = py_constrained, cutoff = cutoff, aic = aic,
      bootstrap = bootstrap, batchsize = batchsize, init = init,
      device = device, double_precision = double_precision,
    ),
    error = function(e) survivalgpu_unavailable_error(e, need_keops = TRUE)
  )
}

#' @noRd
.wceGPU_extract_bootstrap <- function(wce, nbootstraps, cutoff, covariates,
                                      confint) {
  boot <- list(nbootstraps = nbootstraps, is_bootstraps = TRUE)

  # Which candidate won each replicate -- the actual measure of
  # knot-selection uncertainty when nknots has several candidates:
  boot$bootstrap.best.nknots <- as.vector(wce$bootstrap_best_nknots)

  # bootstrap_coef has shape (nbootstraps, 1, number of covariates): remove
  # the middle axis only. drop() would also collapse the other two when
  # there is a single replicate or a single covariate.
  bootstrap_beta.hat.covariates <- wce$bootstrap_coef
  dim(bootstrap_beta.hat.covariates) <- dim(wce$bootstrap_coef)[c(1, 3)]
  rownames(bootstrap_beta.hat.covariates) <- paste0("bootstrap", 1:nbootstraps)
  colnames(bootstrap_beta.hat.covariates) <- covariates
  boot$bootstrap_beta.hat.covariates <- bootstrap_beta.hat.covariates

  probs <- c((1 - confint) / 2, 1 - (1 - confint) / 2)

  # bootstrap_WCE_coef (the raw spline coefficients) is only present when a
  # single nknots candidate was given: with several candidates, different
  # replicates may have selected different-width spline coefficients, so
  # Python does not report a single dense array for it.
  if (!is.null(wce$bootstrap_WCE_coef)) {
    bootstrap_est <- wce$bootstrap_WCE_coef
    dim(bootstrap_est) <- dim(wce$bootstrap_WCE_coef)[c(1, 3)]
    rownames(bootstrap_est) <- paste0("bootstrap", 1:nbootstraps)
    colnames(bootstrap_est) <- paste0("D", 1:(ncol(bootstrap_est)))
    boot$bootstrap_est <- bootstrap_est
    # confidence Interval for coefficients (default 95%)
    boot$est_CI <- apply(bootstrap_est, 2, stats::quantile, p = probs)
  }

  WCEmat_bootstrap = wce$bootstrap_risk_functions
  rownames(WCEmat_bootstrap) <- paste0("bootstrap", 1:nbootstraps)
  colnames(WCEmat_bootstrap) <- paste0("t", 1:cutoff)
  boot$WCEmat_bootstrap <- WCEmat_bootstrap

  # confidence Interval for weights (default 95%)
  boot$WCEmat_CI <- apply(WCEmat_bootstrap, 2, stats::quantile, p = probs)

  # confidence Interval for coefficients (default 95%)
  boot$coef_CI <- apply(bootstrap_beta.hat.covariates, 2, stats::quantile, p = probs)

  boot
}


## Other functions ------------------------



## wceGPU Methods ------------------------


#' Print a wceGPU Model
#'
#' @param x A `wceGPU` object.
#' @param ... Not used.
#' @exportS3Method print wceGPU
#' @noRd
print.wceGPU <- function(x, ...) {
  object <- x
  if (object$constrained == FALSE) {
    cat_constrained <- "Unconstrained"
  } else if (object$constrained %in% c("R", "r", "RIGHT", "Right", "right")) {
    cat_constrained <- "Right constrained"
  } else if (object$constrained %in% c("L", "l", "LEFT", "Left", "left")) {
    cat_constrained <- "Left constrained"
  } else {
    cat_constrained <- "Constrained ?"
  }

  cat(paste(
    "------- ", cat_constrained, "model, with", object$best.nknots,
    ifelse(object$best.nknots > 1, "knots", "knot"), " -------\n"
  ))

  cat("Estimated WCE function :\n")
  print(object$WCEmat)

  if (object$is_bootstraps) {
    cat(paste("Number of bootstraps :", object$nbootstraps, "\n"))
    print(object$WCEmat_bootstrap)
  }



  cat("\n")
  cat(paste("Number of events :", object$nevents[1]),
      paste("Partial log-Likelihoods :", signif(object$loglik[1])),
      paste(
        ifelse(object$aic == TRUE, "AIC :", "BIC :"),
        signif(object$info.criterion[1])
      ),
      sep = "\n"
  )

  if (!is.null(object$beta.hat.covariates)) {
    cat(paste("\nCoefficients estimates for the covariates :"), sep = "\n")
    print(signif(object$beta.hat.covariates) )
  }

  # Display : first four et last four bootstrap
  # if(object$nbootstraps > 8){
  #
  #   WCEmat<-rbind(rbind(object$WCEmat[1:4,],
  #                       matrix(NA,nrow = 1,ncol = ncol(object$WCEmat))),
  #                 object$WCEmat[(object$nbootstraps-3):object$nbootstraps,])
  #
  #   WCEmat_char<-paste(capture.output(print(WCEmat)), collapse = "\n")
  #   cat(gsub("NA","..",WCEmat_char))
  #
  # }else{
  #
  #   print(object$WCEmat)
  #
  # }
  if (object$is_bootstraps) {
    cat("\n ---------------- \n")
    cat(paste0(
      "With bootstrap (", object$nbootstraps,
      " bootstraps), conf.level = ", object$confint, " :\n"
    ))
    print(object$WCEmat_CI)

    if (!is.null(object$covariates)) {
      cat(paste("\nConfidence Interval for covariates estimates :"), sep = "\n")
      print(signif(object$coef_CI[, object$covariates]))
    }
  }

  invisible(x)
}




#' Summary of a wceGPU Model
#'
#' @param object A `wceGPU` object.
#' @param allres Logical. If `TRUE`, `summary()` prints every candidate number
#'   of knots that was tried, with its partial log-likelihood and information
#'   criterion, instead of the summary of the selected model (default
#'   `FALSE`).
#' @exportS3Method summary wceGPU
#' @rdname wceGPU
summary.wceGPU <- function(object, allres = FALSE, ...) {

  objname <- deparse(substitute(object))

  if (isTRUE(allres)) {
    criterion <- if (isTRUE(object$aic)) "AIC" else "BIC"
    tab <- data.frame(
      nknots = object$nknots.grid,
      loglik = object$loglik.grid,
      info.criterion = object$info.criterion.grid
    )
    colnames(tab)[3] <- criterion

    cat("\nCandidate models tried:\n")
    print(tab)
    cat("\nBest model: ", object$best.nknots, " knot(s), by ", criterion,
        ".\n", sep = "")
    return(invisible(NULL))
  }

  shape <- if (isFALSE(object$constrained)) {
    "unconstrained"
  } else if (toupper(substr(object$constrained, 1, 1)) == "R") {
    "constrained to go smoothly to zero for exposures far in the past"
  } else {
    "constrained to start at zero for current exposures"
  }
  cat("Weighted cumulative exposure (WCE) in a Cox proportional hazards model\n")
  cat("Weight function: cubic spline with ", format(object$best.nknots),
      " interior knot(s), ", shape, "\n\n", sep = "")

  if (is.na(object$loglik)) {
    warning("The model did not converge: its results should not be used.",
            call. = FALSE)
  }
  zero_se <- colnames(object$SED)[which(object$SED[1, ] == 0)]
  if (length(zero_se) > 0) {
    warning("Standard error exactly zero for the spline coefficient(s) ",
            paste(zero_se, collapse = ", "),
            ": the model probably did not converge.", call. = FALSE)
  }

  if (!is.null(object$covariates)) {
    est <- object$beta.hat.covariates[1, ]
    se <- object$se.covariates[1, ]
    z <- est / se
    tab <- cbind(est, exp(est), se, z, 2 * stats::pnorm(-abs(z)))
    dimnames(tab) <- list(object$covariates,
                          c("coef", "exp(coef)", "se(coef)", "z", "Pr(>|z|)"))
    cat("Covariates:\n")
    stats::printCoefmat(tab, P.values = TRUE, has.Pvalue = TRUE)
    cat("\n")

    if (object$is_bootstraps) {
      cat("Bootstrap percentile intervals for the coefficients (",
          object$nbootstraps, " replicates, level ", format(object$confint),
          "):\n", sep = "")
      print(signif(t(object$coef_CI[, object$covariates, drop = FALSE]), 4))
      cat("\n")
    }
  }

  criterion <- if (isTRUE(object$aic)) "AIC" else "BIC"
  cat("Partial log-likelihood: ", format(object$loglik), ", ", criterion,
      ": ", format(object$info.criterion), "\n", sep = "")
  cat("Number of events: ", object$nevents, "\n", sep = "")
  cat("\nSee plot(", objname, ") for the estimated weight function.\n",
      sep = "")

  invisible(NULL)
}

#' Coefficients of a wceGPU Model
#'
#' @param object A `wceGPU` object.
#' @param ... Not used.
#' @exportS3Method coef wceGPU
#' @noRd
coef.wceGPU <- function(object, ...) {

  coefs <- list()

  coefs$est <- object$est

  if (object$is_bootstraps) {
    coefs$est_CI <- object$est_CI
  }

  if (!is.null(object$covariates)) {
    coefs$covariates <- object$beta.hat.covariates

    if (object$is_bootstraps) {
      coefs$coef_CI <- object$coef_CI
    }
  }

  coefs
}


#' Plot a wceGPU Model
#'
#' @param hist.covariates Logical. If `TRUE` and [bootstrap()] was called on
#'   the model, `plot()` also draws, for each covariate, the histogram of its
#'   bootstrap coefficients (default `FALSE`).
#' @importFrom graphics matplot
#' @importFrom graphics hist
#' @importFrom graphics title
#' @rdname wceGPU
#' @exportS3Method plot wceGPU
plot.wceGPU <- function(x, ..., hist.covariates = FALSE) {
  object <- x
  if (!object$is_bootstraps) {
    if (object$aic == TRUE) {
      info <- "AIC"
    } else {
      info <- "BIC"
    }
    bic_legend <- paste(info, "=", round(object$info.criterion, 2))

    graphics::matplot(t(object$WCEmat),
                      lty = 1, type = "l", ylab = "weights",
                      xlab = "Time elapsed"
    )
    graphics::title(paste("Estimated weight functions\n", bic_legend))
    graphics::matplot(t(object$WCEmat), pch = 1, add = TRUE)
  } else { # If bootstrap

    if (isTRUE(hist.covariates) & !is.null(object$covariates)) {
      for (i in object$covariates) {
        graphics::hist(object$bootstrap_beta.hat.covariates[, i],
                       main = paste0(
                         "Histogram of ", i, " coefficient with ",
                         object$nbootstraps,
                         " bootstraps\n (without bootstraps coef = ",
                         round(object$beta.hat.covariates[1, i], 2), ")"
                       ),
                       xlab = "Coefficient"
        )
      }
    }

    graphics::matplot((object$WCEmat[1,]),
                      lty = 1, type = "l", ylab = "weights",
                      xlab = "Time elapsed",
                      ylim = range(object$WCEmat[1,], object$WCEmat_CI)
    )
    graphics::title(paste0(
      "Estimated weight functions\n with confidence interval (",
      object$nbootstraps, " bootstraps)"
    ))
    graphics::matplot((object$WCEmat[1,]), pch = 1, add = TRUE)

    if (object$is_bootstraps)
    {

      graphics::matplot((object$WCEmat_CI[1,]),
                        type = c("l"), lty = 2, col = "red",
                        add = TRUE

      )
      graphics::matplot((object$WCEmat_CI[2,]),
                        type = c("l"), lty = 2, col = "red",
                        add = TRUE
    )
    }

  }
}


#' Confidence Intervals of a wceGPU Model
#'
#' @param parm Covariates whose coefficients get a confidence interval, as
#'   numbers or names. Missing (default): all of them.
#' @param level Confidence level of `confint()` (default `0.95`).
#' @param digits Number of significant digits of the levels in the column
#'   names of `confint()` (default `3`).
#' @exportS3Method confint wceGPU
#' @rdname wceGPU
confint.wceGPU <- function(object, parm, level = 0.95, ..., digits = 3) {

  cf <- object$beta.hat.covariates[1,]

  pnames <- names(cf)
  if (missing(parm)) {
    parm <- pnames
  } else if (is.numeric(parm)) {
    parm <- pnames[parm]
  }
  a <- (1 - level) / 2
  a <- c(a, 1 - a)
  pct <- paste(format(100 * a, trim = TRUE, scientific = FALSE, digits = digits), "%")
  fac <- qnorm(a)
  ci <- array(NA, dim = c(length(parm), 2L), dimnames = list(parm, pct))
  # vcovmat is a named list with a single element (the selected model's
  # covariance matrix) -- unwrap it before indexing into its diagonal.
  vcovmat <- object$vcovmat[[1]]
  ses <- sqrt(diag(vcovmat))[parm] # seems to be same thing as object$se.covariates
  ci[] <- cf[parm] + ses %o% fac
  ci
}


#' Hazard Ratio of a WCE Model
#'
#' Computes, from a model fitted by [wceGPU()], the hazard ratio between two
#' exposure histories.
#'
#' @param object A `wceGPU` object.
#' @param vecnum Numeric vector: the exposure history of interest (numerator
#'   of the HR), in reverse chronological order: the first value is the
#'   exposure at the current time, the second the exposure one time unit
#'   earlier, and so on. A vector longer than the `cutoff` of the model is
#'   truncated, and a shorter one is padded with zeros (no exposure further
#'   back in time).
#' @param vecdenom Numeric vector: the reference exposure history (denominator
#'   of the HR), in the same order and with the same length rules as
#'   `vecnum`.
#' @param level Level of the bootstrap confidence interval of the HR (default
#'   `0.95`).
#'
#' @export
#' @return A one-row matrix with the HR and, if [bootstrap()] was called on
#'   `object`, the lower and upper limits of its bootstrap percentile
#'   confidence interval.
#' @examples
#' \dontrun{
#' # Dataset
#' drugdata <- WCE::drugdata
#'
#' # WCE model with bootstrap (example with 20 bootstraps)
#' cutoff <- 90
#' wce_gpu <- wceGPU(data = drugdata, nknots = 1, cutoff = cutoff,
#'                   id = "Id", event = "Event", start = "Start",
#'                   stop = "Stop", expos = "dose",
#'                   covariates = c("age", "sex"),
#'                   constrained = FALSE, aic = FALSE, confint = 0.95,
#'                   x = TRUE)
#' wce_gpu_bootstrap <- bootstrap(wce_gpu, R = 20)
#'
#' # Exposed at a dose vs. unexposed
#' exposed   <- rep(1, cutoff)
#' unexposed <- rep(0, cutoff)
#'
#' HR(wce_gpu_bootstrap, exposed, unexposed)
#' }
HR <- function(object, vecnum, vecdenom, level = 0.95) {
  if (!inherits(object, "wceGPU")) stop("It's not a wceGPU object.")
  cutoff <- ncol(object$WCEmat)
    fit_length <- function(x, n) {
    c(head(x, n), rep(0, max(0, n - length(x))))
  }

  vecnum  <- fit_length(vecnum, cutoff)
  vecdenom <- fit_length(vecdenom, cutoff)

  hr <- exp(sum(object$WCEmat[1, ] * (vecnum - vecdenom)))

  if (object$is_bootstraps) {
    hr_boot <- as.vector(exp(object$WCEmat_bootstrap %*% (vecnum - vecdenom)))
    a <- (1 - level) / 2
    a <- c(a, 1 - a)
    ci <- quantile(hr_boot, p = a)
    pct <- paste0(format(100 * a, trim = TRUE, scientific = FALSE), "%")
    results <- matrix(c(hr, ci), nrow = 1L)
    colnames(results) <- c(
      "HR",
      paste("CI", pct[1]),
      paste("CI", pct[2])
    )
  } else {
    results <- matrix(hr, nrow = 1L)
    colnames(results) <- "HR"
  }

  return(results)

}


#' @rdname bootstrap
#' @exportS3Method bootstrap wceGPU
#' @examples
#' \dontrun{
#' drugdata <- WCE::drugdata
#' fit <- wceGPU(data = drugdata, nknots = 1:3, cutoff = 90, id = "Id",
#'               event = "Event", start = "Start", stop = "Stop",
#'               expos = "dose", covariates = c("age", "sex"), x = TRUE)
#' fit_boot <- bootstrap(fit, R = 500, batchsize = 100)
#' summary(fit_boot)
#' table(fit_boot$bootstrap.best.nknots)
#' }
bootstrap.wceGPU <- function(object, R, batchsize = 0, init = TRUE,
                             device = NULL, double_precision = TRUE, ...) {

  if (missing(R) || is.null(R) || R < 1) {
    stop("R (number of bootstrap replicates) must be a positive integer.")
  }
  if ("data" %in% names(list(...))) {
    stop("bootstrap() resamples the data stored by wceGPU(..., x = TRUE), so ",
         "that it always matches the fit: it doesn't take a data argument.",
         call. = FALSE)
  }
  if (is.null(object$data)) {
    stop("bootstrap() requires the data stored on the fitted object. Refit ",
         "with wceGPU(..., x = TRUE), then call bootstrap() again.",
         call. = FALSE)
  }

  # Python passes the same `init` to every nknots candidate, but their
  # spline coefficients have different lengths: a warm start is only
  # possible with a single candidate.
  single_candidate <- length(object$nknots.grid) == 1
  init_vec <- if (isTRUE(init)) {
    if (single_candidate) unname(object$total_covariates) else NULL
  } else if (isFALSE(init)) {
    NULL
  } else {
    if (!single_candidate) {
      stop("A numeric init can only be given for a model fitted with a ",
           "single nknots value.")
    }
    init
  }

  wce <- .wceGPU_call_python(
    data = object$data, id = object$id, event = object$event,
    start = object$start,
    stop = object$stop, expos = object$expos, covariates = object$covariates,
    nknots = object$nknots, constrained = object$constrained,
    cutoff = object$cutoff, aic = object$aic,
    bootstrap = R, batchsize = batchsize, init = init_vec,
    device = device, double_precision = double_precision
  )

  fit <- object
  boot_fields <- .wceGPU_extract_bootstrap(
    wce, nbootstraps = R, cutoff = object$cutoff,
    covariates = object$covariates, confint = object$confint
  )
  fit[names(boot_fields)] <- boot_fields
  fit
}
