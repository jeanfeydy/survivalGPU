#' Fast WCE
#'
#' @description New implementation of the Weighted Cumulative Exposure model
#'   (see @details), compatible with GPU to accelerate calculation speed and
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
#'   2, then a model with two interior knots is fitted.
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
#' @return WCE results
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
#' # Bootstrap-based inference, as a separate step (normally R > 500)
#' wce_gpu_bootstrap <- bootstrap(wce_gpu, R = 20, data = drugdata, batchsize = 0)
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

  wce <- .wceGPU_call_python(
    data = data, id = id, event = event, start = start, stop = stop,
    expos = expos, covariates = covariates, nknots = nknots,
    constrained = constrained, cutoff = cutoff, aic = aic,
    bootstrap = 0, batchsize = 0, init = NULL,
    device = device, double_precision = double_precision
  )

  nbootstraps <- 0
  is_bootstraps <- FALSE

  # call WCE outputs and rename them to follow R WCE convention

  knotsmat <- matrix(c(wce$knotsmat), nrow = 1)
  rownames(knotsmat) <- paste(nknots, "knot(s)")


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

  vcovmat <- wce$imat   # (1, n, m) array — will need to adapt it if we do with mode than one nknots


  vcovmat <- list()

  vcovmat_knot <- drop(wce$imat)
  cov <- c(covariates, paste0("D", 1:(ncol(est))))
  rownames(vcovmat_knot) <- cov
  colnames(vcovmat_knot) <- cov

  vcovmat[[paste(nknots, "knot(s)")]] <- vcovmat_knot



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
    cutoff = cutoff,
    confint = confint,
    nbootstraps = nbootstraps,
    is_bootstraps = is_bootstraps,
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
  wce_R <- tryCatch(survivalgpu$wce_R, error = survivalgpu_unavailable_error)

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
    error = survivalgpu_unavailable_error
  )
}

#' @noRd
.wceGPU_extract_bootstrap <- function(wce, nbootstraps, nknots, cutoff,
                                      covariates, confint) {
  bootstrap_beta.hat.covariates <- drop(wce$bootstrap_coef)
  rownames(bootstrap_beta.hat.covariates) <- paste0("bootstrap", 1:nbootstraps)
  colnames(bootstrap_beta.hat.covariates) <- covariates

  bootstrap_est <- drop(wce$bootstrap_WCE_coef)
  rownames(bootstrap_est) <- paste0("bootstrap", 1:nbootstraps)
  colnames(bootstrap_est) <- paste0("D", 1:(ncol(bootstrap_est)))

  WCEmat_bootstrap <- wce$bootstrap_risk_functions
  rownames(WCEmat_bootstrap) <- paste0("bootstrap", 1:nbootstraps)
  colnames(WCEmat_bootstrap) <- paste0("t", 1:cutoff)

  probs <- c((1 - confint) / 2, 1 - (1 - confint) / 2)

  list(
    nbootstraps = nbootstraps,
    is_bootstraps = TRUE,
    bootstrap_beta.hat.covariates = bootstrap_beta.hat.covariates,
    bootstrap_est = bootstrap_est,
    WCEmat_bootstrap = WCEmat_bootstrap,
    # confidence Interval for weights (default 95%)
    WCEmat_CI = apply(WCEmat_bootstrap, 2, stats::quantile, p = probs),
    # confidence Interval for coefficients (default 95%)
    coef_CI = apply(bootstrap_beta.hat.covariates, 2, stats::quantile, p = probs),
    est_CI = apply(bootstrap_est, 2, stats::quantile, p = probs)
  )
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
    "------- ", cat_constrained, "model, with", object$nknots,
    ifelse(object$nknots > 1, "knots", "knot"), " -------\n"
  ))

  print("Estimated WCE function\n:")
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
}


#' Summary method for wceGPU object
#'
#' @param object wceGPU object
#' @param allres Post-processing calculations. If TRUE, returns
#'   linear predictors, wald.test, concordance for all bootstraps.
#' @param ... additional argument(s) for methods.
#' @exportS3Method summary wceGPU
#' @rdname wceGPU
summary.wceGPU <- function(object, allres = FALSE, ...) {

  objname <- deparse(substitute(object))

  if (allres == FALSE) {
    sumWCEall(object, objname, ...)
  } else {
    print("The model with more than one number of knots is not yet implemented in wceGPU.")
  }
}

#' For the moment there is only the poissibility to use 1 knot
#' In the future it will be possible to select for several knots
#' Then we will have to do a summary_best and a summary_all, use a parameter
#' @noRd
sumWCEall <- function(object, objname, ...) {

  best <- which.min(object$info.criterion)

  if (is.na(object$loglik[best]) == TRUE) {cat('Warning : the model did not converge, and no \npartial log-likelihood was produced. Results \nfor this model should be ignored.\n\n')}
  if (sum(object$SED[[best]]==0) >0) {cat('Warning : some of the SE for the spline \nvariables in the model are exactlty zero, probably \nbecause the model did not converge. Variable(s)',  names(which(object$SED[[1]]==0)), ' \nhad SE=0. Consider re-parametrizing or increasing \nthe number of iterations\n\n')}

  if (object$analysis == 'Cox') lab <- 'Proportional hazards model'

  nknots <-  length(object$knotsmat)

  if (nknots == 1) {
    sub <- "A single model with 1 knot was estimated.\n\n"} else {
      sub3 <- paste("\nThe best-fitting estimated weight function has ", length(get_interior(object$knotsmat[[best]])), 'knots(s).\n\n', sep ='')
    }

  if (object$constrained == 'Left') {
    cat("\n*** Left-constrained estimated WCE function (",lab ,").***\n", sep='')}
  if (object$constrained == 'Right') {
    cat("\n*** Right-constrained estimated WCE function  (",lab ,").***\n", sep='')}
  if (object$constrained == FALSE) {
    cat("\nUnconstrained estimated WCE function (",lab ,").***\n", sep='')}
  if (object$aic == FALSE) {criterion <- "BIC: "} else {criterion <- "AIC: "}
  if (is.null(object$covariates[1]) == FALSE){
    cat("\nEstimated coefficients for the covariates: \n")
    bhat <- unlist(object$beta.hat.covariates[best,])
    s_hat <- unlist(object$se.covariates[best,])
    coefmat <- data.frame(cbind(bhat, exp(bhat), s_hat, unlist(bhat/s_hat),  2*pnorm(-abs(unlist(bhat/s_hat)))))
    rownames(coefmat) <-  object$covariates
    colnames(coefmat) <- c("coef", "exp(coef)", "se(coef)", "z","p")
    print(round(coefmat, 4))
    cat('\n')
  }

  if (object$is_bootstraps) {
    # cat("\n ---------------- \n")
    cat(paste0(
      "With bootstrap (", object$nbootstraps,
      " bootstraps), conf.level = ", object$confint, " :\n"
    ))
    cat("\nCI of estimates :\n")
    print(t(signif(object$coef_CI[, object$covariates])))
    cat('\n')
    # cat("\n ---------------- \n")
    # cat("\n")
    # cat("Quantile Partial log-Likelihoods :\n")
    # print(quantile(object$loglik))
    # cat("\n")
    # cat(paste("Quantile", ifelse(object$aic==TRUE,"AIC :","BIC :")),
    #     sep = "\n")
    # print(quantile(object$info.criterion))
  }




  objname <- deparse(substitute(object))

  cat("Partial log-likelihood: ", object$loglik[which.min(object$info.criterion)], "  ", criterion, min(object$info.criterion), "\n\n", sep='')
  cat("Number of events: ", object$nevents, "\n\n", sep='')
  cat("Use plot(", objname , ') to see the estimated weight function corresponding to this model.\n', sep="")



}

get_interior <- function(g){
  g <- unlist(g)
  g <- g[5:length(g)]
  g[1:(length(g) - 4)]
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
  if (object$nbootstraps == 1) {
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
        graphics::hist(object$beta.hat.covariate[, i],
                       main = paste0(
                         "Histogram of ", i, " coefficient with ",
                         object$nbootstraps,
                         " bootstraps\n (without bootstraps coef = ",
                         round(object$beta.hat.covariate[1, i], 2), ")"
                       ),
                       xlab = "Coefficient"
        )
      }
    }

    graphics::matplot((object$WCEmat[1,]),
                      lty = 1, type = "l", ylab = "weights",
                      xlab = "Time elapsed"
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
  ses <- sqrt(diag(object$vcovmat))[parm] # seems to be same thing as object$se.covariates
  ci[] <- cf[parm] + ses %o% fac
  ci
}


#' Hazard Ratio for WCE model
#'
#' Calculate the hazard ratio from a wceGPU object to compare two scenarios of
#' time-dependent exposures.
#'
#' @param object wceGPU object.
#' @param vecnum 	A vector of time-dependent exposures corresponding to a
#'   scenario of interest (numerator of the HR).
#' @param vecdenom A vector of time-dependent exposures corresponding to a
#'   scenario for the reference category (denominator of the HR).
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
#' wce_gpu_bootstrap <- wceGPU(data = drugdata, nknots = 1, cutoff = cutoff,
#'                             id = "Id", event = "Event", start = "Start",
#'                             stop = "Stop", expos = "dose",
#'                             covariates = c("age", "sex"),
#'                             constrained = FALSE, aic = FALSE, confint = 0.95,
#'                             nbootstraps = 20, batchsize = 0)
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
  if (length(vecnum) != cutoff | length(vecdenom) != cutoff) stop("At least one of the vector provided as the numerator or denominator is not of proper length.")

  hr <- exp(object$WCEmat[1, ] %*% vecnum) / exp(object$WCEmat[1, ] %*% vecdenom)

 if (object$is_bootstraps) {
   hr_boot <- apply(object$WCEmat_bootstrap, 1, function(x) exp(x %*% vecnum) / exp(x %*% vecdenom))
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


#' Bootstrap-based inference for an already-fitted wceGPU model
#'
#' Adds bootstrap-based confidence intervals to a model already fit by
#' [wceGPU()], without recomputing knot placement from scratch by hand --
#' it reruns the fit with the same knots/cutoff/constraint configuration
#' stored on `object`, plus `R` bootstrap replicates, as a single batched
#' GPU call to the Python backend.
#'
#' @param object a wceGPU object.
#' @param R number of bootstrap replicates.
#' @param data the data frame used in the original [wceGPU()] call (or an
#'   equivalent one) -- required, since `wceGPU()` doesn't retain the
#'   expanded spline-basis feature matrix (unlike coxphGPU()'s `x = TRUE`),
#'   so it's rebuilt here from `data` using the same knot configuration.
#' @param batchsize number of bootstrap copies handled at a time; see
#'   [wceGPU()].
#' @param init starting coefficients for the GPU refits. `TRUE` (the
#'   default) reuses `object`'s own fitted coefficients (covariates + spline
#'   coefficients) as a warm start; `FALSE` starts from zero; or supply a
#'   numeric vector directly.
#' @param device,double_precision see [wceGPU()].
#' @param ... additional argument(s) for methods.
#'
#' @return A copy of `object` with the bootstrap fields
#'   (`bootstrap_beta.hat.covariates`, `bootstrap_est`, `WCEmat_bootstrap`,
#'   `WCEmat_CI`, `coef_CI`, `est_CI`, `nbootstraps`, `is_bootstraps`)
#'   updated from the new bootstrap run; every point-estimate field is left
#'   untouched, so [summary.wceGPU()]/[plot.wceGPU()]/[confint.wceGPU()]
#'   work exactly as they do on a model fit with bootstrap directly.
#'
#' @rdname bootstrap
#' @exportS3Method bootstrap wceGPU
#' @examples
#' \dontrun{
#' drugdata <- WCE::drugdata
#' fit <- wceGPU(data = drugdata, nknots = 1, cutoff = 90, id = "Id",
#'               event = "Event", start = "Start", stop = "Stop",
#'               expos = "dose", covariates = c("age", "sex"))
#' fit_boot <- bootstrap(fit, R = 500, data = drugdata, batchsize = 100)
#' summary(fit_boot)
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

  init_vec <- if (isTRUE(init)) {
    unname(object$total_covariates)
  } else if (isFALSE(init)) {
    NULL
  } else {
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
    wce, nbootstraps = R, nknots = object$nknots, cutoff = object$cutoff,
    covariates = object$covariates, confint = object$confint
  )
  fit[names(boot_fields)] <- boot_fields
  fit
}
