#' Fast WCE
#'
#' @description New implementation of the Weighted Cumulative Exposure model
#'   (see @details), compatible with GPU to accelerate calculation speed and
#'   work with large datasets.
#'
#'   Use `summary()` and `plot()` methods to see results and risk function.
#'
#' @usage
#' wceGPU(data, nknots, cutoff, constrained = FALSE, aic = FALSE, id,
#'        event, start, stop, expos, covariates = NULL, nbootstraps = 1,
#'        batchsize = 0, confint = 0.95, controls = NULL, ...)
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
#' @param aic Logical. If TRUE, then the AIC is used to select the best fitting
#'   model among those estimated for the different numbers of interior knots
#'   requested with nknots. If FALSE, then the BIC is used instead of the AIC.
#'   Default to FALSE (BIC). Note that the BIC implemented in WCE is the version
#'   suggested by Volinsky and Raftery in Biometrics (2000), which corresponds
#'   to BIC = 2 * log(PL) + p * log(d) where PL is the model's partial
#'   likelihood, p is the number of estimated parameters and d is the number of
#'   uncensored events. See Sylvestre and Abrahamowicz (2009) for more details.
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
#' @param nbootstraps Number of repeats for the bootstrap cross-validation.
#' @param batchsize Number of bootstrap copies that should be handled at a time.
#'   Defaults to 0, which means that we handle all copies at once. If you run
#'   into out of memory errors, please consider using batchsize=100, 10 or 1.
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
#' @examples
#' \dontrun{
#' # Dataset
#' drugdata <- WCE::drugdata
#'
#' # WCE model
#' wce_gpu <- wceGPU(data = drugdata, nknots = 1, cutoff = 90, id = "Id",
#'                   event = "Event", start = "Start", stop = "Stop",
#'                   expos = "dose", covariates = c("age", "sex"),
#'                   constrained = FALSE, aic = FALSE, confint = 0.95,
#'                   nbootstraps = 1, batchsize = 0)
#'
#' # Results
#' wce_gpu
#' summary(wce_gpu)
#'
#' # See estimated weight function
#' plot(wce_gpu)
#'
#' # WCE model with bootstrap (example with 20 bootstraps, but normally
#' # nbootstraps > 500)
#' wce_gpu_bootstrap <- wceGPU(data = drugdata, nknots = 1, cutoff = 90,
#'                             id = "Id", event = "Event", start = "Start",
#'                             stop = "Stop", expos = "dose",
#'                             covariates = c("age", "sex"),
#'                             constrained = FALSE, aic = FALSE, confint = 0.95,
#'                             nbootstraps = 20, batchsize = 0)
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
                   nbootstraps = 0, batchsize = 0, confint = 0.95,
                   controls = NULL, ...) {
  UseMethod("wceGPU")
}


#' @return \code{NULL}
#' @noRd
#' @method wceGPU default
#' @exportS3Method wceGPU default
wceGPU.default <- function(data, nknots, cutoff, constrained = FALSE,
                           aic = FALSE, id, event, start, stop, expos,
                           covariates = NULL, nbootstraps = 0, batchsize = 0,
                           confint = 0.95, controls = NULL, device = NULL, ...) {
  # survivalgpu <- use_survivalGPU()

  wce_R <- survivalgpu$wce_R


  # Minor changes for python inputs
  if (constrained == FALSE) {
    py_constrained <- "None"
  } else {
    py_constrained <- constrained
  }


  if (length(covariates) < 2) {
    py_covariates <- as.list(covariates)
  } else {
    py_covariates <- covariates
  }




  wce <- wce_R(
    data = data, ids = id, covars = py_covariates, stop = stop,
    doses = expos, events = event, n_knots = nknots,
    constrained = py_constrained, cutoff = cutoff,
    bootstrap = nbootstraps, batchsize = batchsize, device = device
  )


  if (is.null(nbootstraps)) {
    is_bootstraps <- FALSE
  }
  else if (nbootstraps == 0) {
     is_bootstraps <- FALSE
  }
  else if (nbootstraps > 0) {
     is_bootstraps <- TRUE
  }
  else {
      stop(sprintf("Invalid value for nbootstraps: %s. Expect NULL or an positive integer", deparse(nbootstraps)))
  }







  # print("risk function")
  # print(wce$risk_function)
  # print("ending risk function")
  # print(wce$bootstrap_risk_functions)


  # --- outputs of wce_R :
  # hessian
  # coef
  # loglik
  # u
  # imat
  # means
  # knotsmat
  # std
  # SED
  # WCEmat
  # est
  # vcovmat

  # print(wce$covars) # covars very weird


  # get all relevant outputs and rename them to follow the R WCE
  # R WCE naming convention

  # beta.hat.covariates <- wce$coef
  # est <- wce$WCE_coef
  # SED <- wce$SED

  # loglik <- wce$loglik
  # vcovmat <- wce$imat


  # bootstrap_WCE_mat <- wce$bootstrap_risk_functions
  # bootstrap_beta.hat.covariates <- wce$bootstrap_beta.hat.covariates
  # bootstrap_est <- wce$bootstrap_WCE_coef


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

  BIC <- sapply(wce$loglik, BIC_for_wce,
                n.events = nevents, n.knots = nknots,
                cons = constrained, aic = aic, covariates = covariates
  )




  # List to return
  results <- list(
    knotsmat = knotsmat,
    beta.hat.covariates = beta.hat.covariates,
    se.covariates = se.covariates,
    est = est,
    SED = SED,
    WCEmat = WCEmat,
    vcovmat = vcovmat,
    covariates = covariates,
    loglik = loglik,
    constrained = constrained,
    nevents = nevents,
    aic = aic,
    info.criterion = BIC,
    nknots = nknots,
    confint = confint,
    nbootstraps = nbootstraps,
    is_bootstraps = is_bootstraps
  )


  if (is_bootstraps) {

    bootstrap_beta.hat.covariates <- drop(wce$bootstrap_coef)
    rownames(bootstrap_beta.hat.covariates) <- paste0("bootstrap", 1:nbootstraps)
    colnames(bootstrap_beta.hat.covariates) <- covariates
    results$bootstrap_beta.hat.covariates <- bootstrap_beta.hat.covariates

    bootstrap_est <- drop(wce$bootstrap_WCE_coef)
    rownames(bootstrap_est) <- paste0("bootstrap", 1:nbootstraps)
    colnames(bootstrap_est) <- paste0("D", 1:(ncol(bootstrap_est)))
    results$bootstrap_est <- bootstrap_est


    WCEmat_bootstrap = wce$bootstrap_risk_functions
    rownames(WCEmat_bootstrap) <- paste0("bootstrap", 1:nbootstraps)
    colnames(WCEmat_bootstrap) <- paste0("t", 1:cutoff)
    results$WCEmat_bootstrap <- WCEmat_bootstrap


    probs <- c((1 - confint) / 2, 1 - (1 - confint) / 2)
    # confidence Interval for weights (default 95%)
    results$WCEmat_CI <- apply(WCEmat_bootstrap, 2, stats::quantile, p = probs)


    # confidence Interval for coefficients (default 95%)
    results$coef_CI  <- apply(bootstrap_beta.hat.covariates, 2, stats::quantile, p = probs)
    results$est_CI  <- apply(bootstrap_est, 2, stats::quantile, p = probs)
  }

  results$analysis <- "Cox"

  # wceGPU object
  class(results) <- "wceGPU"
  return(results)
}


## Other functions ------------------------

# Estimate BIC for different models
BIC_for_wce <- function(PL, n.events, n.knots, cons = F, aic = FALSE, covariates) {
  if (is.null(covariates == T)) {
    if (cons == FALSE) {
      if (aic == TRUE) {
        bic <- -2 * PL + (n.knots + 4) * 2
      } else {
        bic <- -2 * PL + (n.knots + 4) * log(n.events)
      }
    } else {
      if (aic == TRUE) {
        bic <- -2 * PL + (n.knots + 2) * 2
      } else {
        bic <- -2 * PL + (n.knots + 2) * log(n.events)
      }
    }
  } else {
    pp <- length(covariates)
    if (cons == FALSE) {
      if (aic == TRUE) {
        bic <- -2 * PL + (n.knots + 4 + pp) * 2
      } else {
        bic <- -2 * PL + (n.knots + 4 + pp) * log(n.events)
      }
    } else {
      if (aic == TRUE) {
        bic <- -2 * PL + (n.knots + 2 + pp) * 2
      } else {
        bic <- -2 * PL + (n.knots + 2 + pp) * log(n.events)
      }
    }
  }
  return(bic)
}



## wceGPU Methods ------------------------


#' Print method for wceGPU
#'
#' @param object wceGPU object
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


summary.wceGPU <- function(object, allres = FALSE, ...) {

  objname <- deparse(substitute(object))

  if (allres == FALSE) {
    sumWCEall(object, objname, ...)
  } else {
    print("The model with more than one number of knots is not yet implemented in wceGPU.")
  }
}

#' Summary method for wceGPU object
#'
#' For the moment there is only the poissibility to use 1 knot
#' In the future it will be possible to select for several knots
#' Then we will have to do a summary_best and a summary_all, use a parameter
sumWCEall <- function(object, objname, ...) {

  best <- which.min(object$info.criterion)

  if (is.na(object$loglik[best]) == T) {cat('Warning : the model did not converge, and no \npartial log-likelihood was produced. Results \nfor this model should be ignored.\n\n')}
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
  if (object$aic == F) {criterion <- "BIC: "} else {criterion <- "AIC: "}
  if (is.null(object$covariates[1]) == F){
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

#' Summary method for wceGPU object
#'
#' @param object wceGPU object
#' @param ... additional argument(s) for methods.
#' @exportS3Method summary wceGPU
#' @rdname wceGPU


# #' Coef method for wceGPU object
# #'
# #' @param object wceGPU object.
# #' @param ... additional argument(s) for methods.
# #' @exportS3Method coef wceGPU
# #' @noRd
# coef.wceGPU <- function(object, ...) {
#   if (is.null(object$beta.hat.covariate)) {
#     list(WCEest = object$est)
#   } else {
#     if (object$nbootstraps == 1) {
#       list(
#         WCEest = object$,
#         covariates = object$coef[, object$covariates]
#       )
#     } else {
#       list(
#         coef = list(
#           WCEest = object$coef[, !colnames(object$coef) %in% object$covariates],
#           covariates = object$coef[, object$covariates]
#         ),
#         CI = list(
#           WCEest = object$coef_CI[, !colnames(object$coef) %in% object$covariates],
#           covariates = object$coef_CI[, object$covariates]
#         )
#       )
#     }
#   }
# }


coef.wceGPU <- function(object, ...) {

  ceofs <- list()

  coefs$est <- object$est

  is_bootstraps <- FALSE


  if (n_bootstraps > 1) {
    results$est_CI <- object$est_CI
  }

  if (!is.null(object$covariates)) {
    coefs$covariates <- object$beta.hat.covariates

    if (n_bootstraps > 1) {
      results$coef_CI <- object$coef_CI
    }
  }




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
#' Calcul the hazard ratio from a wceGPU object to compare two scenarios of
#' time-dependant exposures.
#'
#' @param object wceGPU object.
#' @param vecnum 	A vector of time-dependent exposures corresponding to a
#'   scenario of interest (numerator of the HR).
#' @param vecdenom A vector of time-dependent exposures corresponding to a
#'   scenario for the reference category (denominator of the HR).
#' @param level the confidence level required for HR CI. Default to 0.95.
#' @param without_bootstrap Gaussian approximation for confidence interval.
#'
#' @export
#' @return Returns a HR according to the scenarios. If bootstrap is present
#' (or without_bootstrap = TRUE) in wceGPU object, this function returns
#' confidence interval for the HR.
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
#'
#' # Confidence interval with Gaussian approximation when no bootstrap
#'  wce_gpu <- wceGPU(data = drugdata, nknots = 1, cutoff = cutoff, id = "Id",
#'                    event = "Event", start = "Start", stop = "Stop",
#'                    expos = "dose", covariates = c("age", "sex"),
#'                    constrained = FALSE, aic = FALSE, confint = 0.95,
#'                    batchsize = 0)
#'
#' HR(wce_gpu_bootstrap, exposed, unexposed, without_bootstrap = TRUE)
#' }
HR <- function(object, vecnum, vecdenom, level = 0.95) {

  if (!inherits(object, "wceGPU")) stop("It's not a wceGPU object.")
  cutoff <- ncol(object$WCEmat)
  if (length(vecnum) != cutoff | length(vecdenom) != cutoff) stop("At least one of the vector provided as the numerator or denominator is not of proper length.")

  hr <- apply(object$WCEmat, 1, function(x) exp(x %*% vecnum) / exp(x %*% vecdenom), simplify = TRUE)

 if (object$is_bootstraps) {
   a <- (1 - level) / 2
   a <- c(a, 1 - a)
   ci <- quantile(hr, p = a)
   pct <- paste0(format(100 * a, trim = TRUE, scientific = FALSE), "%")
   results <- matrix(c(hr[1], ci), nrow = 1L)
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
