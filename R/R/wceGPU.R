#' Fast WCE
#'
#' @description New implementation of the Weighted Cumulative Exposure model
#'   (see Details), compatible with GPU to accelerate calculation speed and
#'   work with large datasets. Fits a point estimate only; call [bootstrap()]
#'   on the result for bootstrap-based inference.
#'
#'   Use `summary()` and `plot()` methods to see results and risk function.
#'
#' @usage
#' wceGPU(data, nknots, cutoff, constrained = FALSE, aic = FALSE, id,
#'        event, start, stop, expos, covariates = NULL,
#'        confint = 0.95, controls = NULL, ...)
#'
#' @param data A data frame in an interval (long) format, in which each line
#'   corresponds to one and only one time unit for a given individual.
#' @param nknots Corresponds to the number(s) of interior knots for the cubic
#'   splines to estimate the weight function. For example, if nknots is set to
#'   2, then a model with two interior knots is fitted. If several values are
#'   given (e.g. `1:3`), one model is fitted per candidate and the one that
#'   minimizes the information criterion (see `aic`) is selected -- see
#'   `best.nknots` and the `*.grid` fields in the returned object.
#' @param cutoff Integer. Time window over which the WCE model is estimated.
#'   Corresponds to the length of the estimated weight function.
#' @param constrained Controls whether the weight function should be constrained
#'   to smoothly go to zero. Set to FALSE for unconstrained models, to 'Right'
#'   or 'R' to constrain the weight function to smoothly go to zero for exposure
#'   remote in time, and to 'Left' or 'L' to constrain the weight function to
#'   start a zero for the current values.
#' @param aic Logical. Controls which information criterion is reported in
#'   `info.criterion`: the AIC if TRUE, the BIC if FALSE (default). Note that
#'   the BIC implemented in WCE is the version suggested by Volinsky and
#'   Raftery in Biometrics (2000), which corresponds to
#'   BIC = -2 * log(PL) + p * log(d) where PL is the model's partial
#'   likelihood, p is the number of estimated parameters and d is the number of
#'   uncensored events; the AIC replaces the log(d) penalty with 2. See
#'   Sylvestre and Abrahamowicz (2009) for more details.
#' @param id Name of the variable in data corresponding to the identification of
#'   subjects.
#' @param event Name of the variable in data corresponding to event indicator.
#'   Must be coded 1 = event and 0 = no event.
#' @param start Name of the variable in data corresponding to the starting time
#'   for the interval. Corresponds to time argument in function Surv in the
#'   survival package.
#' @param stop Name of the variable in data corresponding to the ending time for
#'   the interval. Corresponds to time2 argument in function Surv in the
#'   survival package.
#' @param expos Name of the variable in data corresponding to the exposure
#'   variable.
#' @param covariates Optional. Vector of characters corresponding to the name(s)
#'   of the variable(s) in data corresponding to the covariate(s) to be included
#'   in the model. Default to NULL, which corresponds to fitting model(s)
#'   without covariates.
#' @param confint Level for confidence intervals. Default to 0.95.
#' @param controls List corresponding to the control parameters to be passed to
#'   the coxph function. See coxph.control for more details.
#' @param ... Optional; other parameters to be passed through to WCE
#'
#' @details WCE implements a flexible method for modeling cumulative effects of
#'   time-varying exposures, weighted according to their relative proximity in
#'   time, and represented by time-dependent covariates. The current
#'   implementation estimates the weight function in the Cox proportional
#'   hazards model. The function that assigns weights to doses taken in the past
#'   is estimated using cubic regression splines.
#'
#' @references Sylvestre MP, Abrahamowicz M. Flexible modeling of the cumulative
#'   effects of time-dependent exposures on the hazard. Stat Med. 2009 Nov
#'   30;28(27):3437-53.
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
#'                   constrained = FALSE, aic = FALSE, confint = 0.95)
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
#' wce_gpu_bootstrap <- bootstrap(wce_gpu, R = 20, data = drugdata)
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
                   confint = 0.95, controls = NULL, ...) {
  UseMethod("wceGPU")
}


#' @return \code{NULL}
#' @noRd
#' @method wceGPU default
#' @exportS3Method wceGPU default
wceGPU.default <- function(data, nknots, cutoff, constrained = FALSE,
                           aic = FALSE, id, event, start, stop, expos,
                           covariates = NULL,
                           confint = 0.95, controls = NULL, device = NULL, double_precision = TRUE, ...) {
  # survivalgpu <- use_survivalGPU()

  # Bootstrap is no longer requested at fit time. Without this check, the old
  # arguments would be silently swallowed by `...` and the fit returned
  # without any bootstrap.
  removed <- intersect(names(list(...)), c("nbootstraps", "batchsize"))
  if (length(removed)) {
    stop("wceGPU() no longer takes ", paste(removed, collapse = ", "),
         ": fit the model first, then call bootstrap(fit, R = , data = ).",
         call. = FALSE)
  }

  wce <- .wceGPU_call_python(
    data = data, id = id, event = event, start = start, stop = stop,
    expos = expos, covariates = covariates, nknots = nknots,
    constrained = constrained, cutoff = cutoff, aic = aic,
    bootstrap = 0, batchsize = 0, init = NULL,
    device = device, double_precision = double_precision
  )


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
    # Stored purely so bootstrap() can be called later without re-supplying
    # every column name -- these are already required at every wceGPU() fit,
    # unlike coxphGPU()'s patient_id which is only needed for bootstrap.
    id = id,
    event = event,
    start = start,
    stop = stop,
    expos = expos
  )

  results$analysis <- "Cox"

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


#' Print method for wceGPU
#'
#' @param x wceGPU object
#' @param ... additional argument(s) for methods.
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




#' Summary method for wceGPU object
#'
#' @param object wceGPU object
#' @param allres If TRUE, `summary()` prints every candidate number of knots
#'   that was tried, with its partial log-likelihood and information criterion,
#'   instead of the summary of the selected model.
#' @param ... additional argument(s) for methods.
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

#' Coef method for wceGPU object
#'
#' @param object wceGPU object.
#' @param ... additional argument(s) for methods.
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


#' Plot method for wceGPU
#'
#' @param x wceGPU object.
#' @param hist.covariates show histogram for each covariates if you use
#'   bootstrap.
#' @param ... additional argument(s) for methods.
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


#' Confint method for wceGPU
#'
#' @param object wceGPU object.
#' @param parm a specification of which parameters are to be given confidence
#'   intervals, either a vector of numbers or a vector of names. If missing, all
#'   parameters are considered.
#' @param level the confidence level required.
#' @param digits significant digits to print.
#' @param ... additional argument(s) for methods.
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


#' Hazard Ratio for WCE model
#'
#' Calculate the hazard ratio from a wceGPU object to compare two scenarios of
#' time-dependent exposures.
#'
#' @param object wceGPU object.
#' @param vecnum A vector of time-dependent exposures corresponding to a
#'   scenario of interest (numerator of the HR), in reverse chronological
#'   order: the first value is the exposure at the current time, the second
#'   the exposure one time unit earlier, and so on. A vector longer than the
#'   `cutoff` of the model is truncated, and a shorter one is padded with zeros
#'   (no exposure further back in time).
#' @param vecdenom A vector of time-dependent exposures corresponding to a
#'   scenario for the reference category (denominator of the HR), in the same
#'   order and with the same length rules as `vecnum`.
#' @param level the confidence level required for HR CI. Default to 0.95.
#'
#' @export
#' @return Returns a HR according to the scenarios. If bootstrap is present
#' in the wceGPU object, this function returns a confidence interval for the HR.
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
#'                   constrained = FALSE, aic = FALSE, confint = 0.95)
#' wce_gpu_bootstrap <- bootstrap(wce_gpu, R = 20, data = drugdata)
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
#'               expos = "dose", covariates = c("age", "sex"))
#' fit_boot <- bootstrap(fit, R = 500, data = drugdata, batchsize = 100)
#' summary(fit_boot)
#' table(fit_boot$bootstrap.best.nknots)
#' }
bootstrap.wceGPU <- function(object, R, data, batchsize = 0, init = TRUE,
                             device = NULL, double_precision = TRUE, ...) {

  if (missing(R) || is.null(R) || R < 1) {
    stop("R (number of bootstrap replicates) must be a positive integer.")
  }
  if (missing(data)) {
    stop("data is required: wceGPU() doesn't retain the expanded feature ",
         "matrix, so bootstrap() needs the original (or an equivalent) ",
         "data frame to rebuild it.")
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
    data = data, id = object$id, event = object$event, start = object$start,
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
