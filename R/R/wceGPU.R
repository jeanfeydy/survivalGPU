#' Fast WCE
#'
#' @description New implementation of the Weighted Cumulative Exposure model
#'   (see @details), compatible with GPU to accelerate calculation speed and
#'   work with large datasets.
#'
#'   Use `summary()` and `plot()` methods to see results and risk function.
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
#' wce_gpu <- wceGPU(
#'   data = drugdata, nknots = 1, cutoff = 90, id = "Id",
#'   event = "Event", start = "Start", stop = "Stop", expos = "dose",
#'   covariates = c("age", "sex"), constrained = FALSE, aic = FALSE,
#'   confint = 0.95, nbootstraps = 1, batchsize = 0
#' )
#'
#' # Results
#' wce_gpu
#' summary(wce_gpu)
#'
#' # See estimated weight function
#' plot(wce_gpu)
#'
#' # WCE model with bootstrap (example with 20 bootstraps)
#' wce_gpu_bootstrap <- wceGPU(
#'   data = drugdata, nknots = 1, cutoff = 90, id = "Id",
#'   event = "Event", start = "Start", stop = "Stop", expos = "dose",
#'   covariates = c("age", "sex"), constrained = FALSE, aic = FALSE,
#'   confint = 0.95, nbootstraps = 20, batchsize = 0
#' )
#'
#' # See confidence bands for the estimated weight function due to bootstrap
#' plot(wce_gpu_bootstrap)
#'
#' # All estimated coefficients in bootstrap
#' coef(wce_gpu_bootstrap)
#' }
wceGPU <- function(data, nknots, cutoff, constrained = FALSE, aic = FALSE, id,
                   event, start, stop, expos, covariates = NULL,
                   nbootstraps = 1, batchsize = 0, confint = 0.95,
                   controls = NULL, ...) {
  UseMethod("wceGPU")
}


#' @return \code{NULL}
#' @noRd
#' @method wceGPU default
#' @exportS3Method wceGPU default
wceGPU.default <- function(data, nknots, cutoff, constrained = FALSE,
                           aic = FALSE, id, event, start, stop, expos,
                           covariates = NULL, nbootstraps = 1, batchsize = 0,
                           confint = 0.95, controls = NULL, ...) {
  survivalgpu <- use_survivalGPU()
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
    doses = expos, events = event, nknots = nknots,
    constrained = py_constrained, cutoff = cutoff,
    bootstrap = nbootstraps, batchsize = batchsize
  )

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


  # Outputs post processing
  knotsmat <- matrix(c(wce$knotsmat), nrow = 1)
  rownames(knotsmat) <- paste(nknots, "knot(s)")

  WCEmat <- wce$WCEmat
  colnames(WCEmat) <- paste0("t", 1:cutoff)
  rownames(WCEmat) <- paste0("bootstrap", 1:nbootstraps)

  coef <- wce$coef
  cov <- c(covariates, paste0("D", 1:(ncol(coef) - length(covariates))))
  colnames(coef) <- cov
  rownames(coef) <- paste0("bootstrap", 1:nbootstraps)

  vcovmat <- lapply(c(1:nbootstraps), function(x) wce$imat[x, , ])
  vcovmat <- lapply(vcovmat, "colnames<-", cov)
  vcovmat <- lapply(vcovmat, "rownames<-", cov)
  names(vcovmat) <- paste0("bootstrap", 1:nbootstraps)

  SE <- cbind(wce$std, wce$SED)
  colnames(SE) <- cov
  rownames(SE) <- paste0("bootstrap", 1:nbootstraps)

  names(data)[names(data) == event] <- "Event"
  nevents <- length(data$Event[data$Event == 1])

  BIC <- sapply(wce$loglik, BIC_for_wce,
    n.events = nevents, n.knots = nknots,
    cons = constrained, aic = aic, covariates = covariates
  )

  # List to return
  results <- list(
    knotsmat = knotsmat,
    WCEmat = WCEmat,
    loglik = c(wce$loglik),
    coef = coef,
    vcovmat = vcovmat,
    SE = SE,
    covariates = covariates,
    constrained = constrained,
    nevents = nevents,
    aic = aic,
    info.criterion = BIC,
    nknots = nknots,
    confint = confint,
    nbootstraps = nbootstraps
  )

  if (nbootstraps > 1) {
    probs <- c((1 - confint) / 2, 1 - (1 - confint) / 2)
    # confidence Interval for weights (default 95%)
    results$WCEmat_CI <- apply(WCEmat, 2, stats::quantile, p = probs)

    # confidence Interval for coefficients (default 95%)
    results$coef_CI <- apply(coef, 2, stats::quantile, p = probs)
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

  rownames(object$WCEmat) <- rep(" ", object$nbootstraps)
  print(object$WCEmat[1, ])

  cat("\n")
  cat(paste("Number of events :", object$nevents[1]),
    paste("Partial log-Likelihoods :", signif(object$loglik[1])),
    paste(
      ifelse(object$aic == TRUE, "AIC :", "BIC :"),
      signif(object$info.criterion[1])
    ),
    sep = "\n"
  )

  if (!is.null(object$covariates)) {
    cat(paste("\nCoefficients estimates for the covariates :"), sep = "\n")
    print(signif(object$coef[1, object$covariates]))
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
  if (object$nbootstraps > 1) {
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
#' @param ... additional argument(s) for methods.
#' @exportS3Method summary wceGPU
#' @rdname wceGPU
summary.wceGPU <- function(object, ...) {
  estimates <- object$coef[1, object$covariates]
  se_estimates <- object$SE[1, object$covariates]
  z <- estimates / se_estimates
  p <- 2 * pnorm(-abs(z))
  conf.int <- confint(object, level = object$confint, parm = object$covariates)

  coef_mat <- data.frame(
    coef = estimates,
    ci_inf = conf.int[, 1],
    ci_sup = conf.int[, 2],
    exp_coef = exp(estimates),
    se_coef = se_estimates,
    z = z,
    p = p
  )

  colnames(coef_mat) <- c(
    "coef", paste("CI", colnames(conf.int)), "exp(coef)",
    "se(coef)", "z", "p"
  )

  cat("Estimated coefficients for the covariates :", sep = "\n")
  stats::printCoefmat(coef_mat,
    digits = 2,
    P.values = TRUE,
    has.Pvalue = TRUE
  )
  cat("\n")
  cat(paste("Number of events :", object$nevents[1]),
    paste("Partial log-Likelihoods :", signif(object$loglik[1])),
    paste(
      ifelse(object$aic == TRUE, "AIC :", "BIC :"),
      signif(object$info.criterion[1])
    ),
    sep = "\n"
  )
  if (object$nbootstraps > 1) {
    cat("\n ---------------- \n")
    cat(paste0(
      "With bootstrap (", object$nbootstraps,
      " bootstraps), conf.level = ", object$confint, " :\n"
    ))
    cat("\nCI of estimates :\n")
    print(t(signif(object$coef_CI[, object$covariates])))
    # cat("\n")
    # cat("Quantile Partial log-Likelihoods :\n")
    # print(quantile(object$loglik))
    # cat("\n")
    # cat(paste("Quantile", ifelse(object$aic==TRUE,"AIC :","BIC :")),
    #     sep = "\n")
    # print(quantile(object$info.criterion))
  }
}


#' Coef method for wceGPU object
#'
#' @param object wceGPU object.
#' @param ... additional argument(s) for methods.
#' @exportS3Method coef wceGPU
#' @noRd
coef.wceGPU <- function(object, ...) {
  if (is.null(object$covariates)) {
    list(WCEest = object$coef)
  } else {
    if (object$nbootstraps == 1) {
      list(
        WCEest = object$coef[, !colnames(object$coef) %in% object$covariates],
        covariates = object$coef[, object$covariates]
      )
    } else {
      list(
        coef = list(
          WCEest = object$coef[, !colnames(object$coef) %in% object$covariates],
          covariates = object$coef[, object$covariates]
        ),
        CI = list(
          WCEest = object$coef_CI[, !colnames(object$coef) %in% object$covariates],
          covariates = object$coef_CI[, object$covariates]
        )
      )
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
        graphics::hist(object$coef[, i],
          main = paste0(
            "Histogram of ", i, " coefficient with ",
            object$nbootstraps,
            " bootstraps\n (without bootstraps coef = ",
            round(object$coef[1, i], 2), ")"
          ),
          xlab = "Coefficient"
        )
      }
    }

    graphics::matplot((object$WCEmat[1, ]),
      lty = 1, type = "l", ylab = "weights",
      xlab = "Time elapsed"
    )
    graphics::title(paste0(
      "Estimated weight functions\n with confidence interval (",
      object$nbootstraps, " bootstraps)"
    ))
    graphics::matplot((object$WCEmat[1, ]), pch = 1, add = TRUE)
    graphics::matplot((object$WCEmat_CI[1, ]),
      type = c("l"), lty = 2, col = "red",
      add = TRUE
    )
    graphics::matplot((object$WCEmat_CI[2, ]),
      type = c("l"), lty = 2, col = "red",
      add = TRUE
    )
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
  cf <- object$coef[1, ]
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
  ses <- sqrt(diag(object$vcovmat[[1]]))[parm]
  ci[] <- cf[parm] + ses %o% fac
  ci
}
