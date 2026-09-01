#' Fast Cox Proportional Hazards Regression Model
#'
#' @description Fits a Cox proportional hazards regression model. An extension
#'   to use (or not) your GPU to speed up calculations. Fits a point estimate
#'   only; call [bootstrap()] on the result for bootstrap-based inference.
#'
#' @usage coxphGPU(formula, data, ties = c("efron", "breslow"), init, control,
#'          singular.ok = TRUE, model = FALSE, x = FALSE, y = TRUE, ...)
#'
#' @inheritParams survival::coxph
#' @param formula a formula object, with the response on the left of a ~
#'   operator, and the terms on the right. The response must be a survival
#'   object as returned by the Surv function.
#' @param ... Other arguments for methods.
#'
#' @import stats
#' @import survival
#' @importFrom utils methods
#' @importFrom utils head
#' @importFrom data.table data.table
#'
#' @return A coxphGPU object representing the fit.
#' @export
#'
#' @references Therneau T (2021). _A Package for Survival Analysis in R_. R
#'   package version 3.2-13
#'
#' @seealso [survival::coxph()], [bootstrap()]
#'
#' @examples
#' \dontrun{
#' library(survival)
#' library(WCE)
#' data(drugdata)
#'
#' ## Check CUDA drivers (if FALSE you use CPU)
#' use_cuda()
#'
#' ## Cox Proportional Hazards point estimate; x = TRUE is required if you
#' ## plan to call bootstrap() on the result afterwards.
#' fit <- coxphGPU(Surv(Start, Stop, Event) ~ sex + age,
#'                 data = drugdata, x = TRUE)
#'
#' ## Bootstrap-based inference, as a separate step
#' fit_boot <- bootstrap(fit, R = 1000, patient_id = "Id", data = drugdata,
#'                       batchsize = if (use_cuda()) 200 else 10)
#'
#' summary(fit_boot)
#' }
coxphGPU <- function(formula, data, ties = c("efron", "breslow"), init, control,
                     singular.ok = TRUE, model = FALSE, x = FALSE, y = TRUE,
                     ...) {
  UseMethod("coxphGPU")
}

#' @return \code{NULL}
#' @noRd
#' @method coxphGPU default
#' @exportS3Method coxphGPU default
coxphGPU.default <- function(formula, data, ties = c("efron", "breslow"), init,
                             control, singular.ok = TRUE,
                             model = FALSE, x = FALSE, y = TRUE, ..., weights,
                             subset, na.action, robust, tt, method = ties, id,
                             cluster, istate, statedata,
                             nocenter = c(-1, 0, 1), device = NULL, double_precision = TRUE) {

  if (!missing(weights)) stop("weights are not yet implemented in coxphGPU")
  if (!missing(tt)) stop("tt process is not yet implemented in coxphGPU")
  if (is.list(formula)) stop("multistate models not yet implemented on coxphGPU")

  ##############################################################################
  ##############################################################################
  #
  # Pre-processing of survival::coxph
  #
  ##############################################################################
  ##############################################################################

  ties <- match.arg(ties)

  # To save in memory all coxph inputs
  Call <- match.call()
  ## We want to pass any ... args to coxph.control, but not pass things
  ##  like "dats=mydata" where someone just made a typo.  The use of ...
  ##  is simply to allow things like "eps=1e6" with easier typing
  # We save in a list all arguments which are not default
  extraArgs <- list(...)
  if (length(extraArgs)) { # Condition if more options
    controlargs <- names(formals(coxph.control)) # legal arg names
    indx <- pmatch(names(extraArgs), controlargs, nomatch = 0L)
    if (any(indx == 0L)) {
      stop(gettextf(
        "Argument %s not matched",
        names(extraArgs)[indx == 0L]
      ), domain = NA)
    }
  }

  # if no 'control' in input, we take the coxph.control() values
  if (missing(control)) control <- coxph.control(...)

  if (missing(formula)) stop("a formula argument is required")

  ##############################################################################
  # Delegate formula parsing, cluster()/strata() handling, and model.frame /
  # model.matrix construction to survival::coxph() itself, through its public
  # API, instead of vendoring a modified copy of that pipeline. iter.max = 0
  # means no Newton iteration runs here -- `prep` is only used to obtain the
  # design matrix, response, strata, terms/contrasts/xlevels metadata and
  # covariate means that coxph() already knows how to build correctly. The
  # numerical fit still happens below, via the Python/GPU backend.
  ##############################################################################

  prep_control <- control
  prep_control$iter.max <- 0

  prep_call <- Call
  prep_call[[1L]] <- quote(survival::coxph)
  prep_call$ties <- ties
  prep_call$control <- prep_control
  prep_call$model <- model
  prep_call$x <- TRUE
  prep_call$y <- TRUE
  # Drop coxphGPU-only arguments that survival::coxph() doesn't accept.
  # (a call object errors on `[[<- NULL` for a name it doesn't already
  #  have, unlike a list, so only touch names actually present.)
  for (nm in c("device", "double_precision", "singular.ok")) {
    if (nm %in% names(prep_call)) prep_call[[nm]] <- NULL
  }

  prep <- eval(prep_call, parent.frame())

  # survival::coxph() sets naive.var only when it decided (from cluster/id/
  # weights, or an explicit robust=TRUE) that a robust sandwich variance was
  # needed; reuse that decision instead of re-deriving it ourselves.
  if (!is.null(prep$naive.var)) {
    stop("robust variance is not yet implemented in coxphGPU")
  }

  Y <- prep$y
  type <- attr(Y, "type")
  if (type != "right" && type != "counting") {
    stop(paste("Cox model doesn't support \"", type,
               "\" survival data",
               sep = ""
    ))
  }
  data.n <- nrow(Y) # remember this before any time transforms

  if (!is.null(prep$offset) && any(prep$offset != 0)) {
    stop("offset() is not yet implemented in coxphGPU")
  }

  if (prep$nevent == 0) {
    # No events in the data! Mirrors survival::coxph()'s own early return.
    class(prep) <- "coxph"
    prep$x <- NULL
    return(prep)
  }

  X <- prep$x
  Terms <- prep$terms
  assign <- prep$assign
  xlevels <- prep$xlevels
  contr.save <- prep$contrasts
  weights <- prep$weights # NULL unless the (currently blocked) weights arg is used
  offset <- rep(0.0, nrow(Y)) # always all-zero: any nonzero offset already stopped above
  istrat <- if (!is.null(prep$strata)) as.integer(prep$strata) else NULL
  # Below, a few dead-but-harmless branches inherited from survival::coxph()
  # (robust/cluster, tt, and x/strata attachment) are kept as-is for fidelity
  # with survival's own code; these three stand in for variables that used to
  # come from the vendored formula/model.frame parsing this delegates away.
  robust <- FALSE # any combination that would set this TRUE already stopped above
  cluster <- NULL # ditto -- cluster()/robust always errors out before this point
  timetrans <- NULL # tt() always errors out before this point (guard at the top)
  strats <- prep$strata
  strata.keep <- prep$strata

  # infinite covariates are not screened out by the na.omit routines
  if (!all(is.finite(X))) {
    stop("data contains an infinite predictor")
  }

  # init is checked after the final X matrix has been made
  if (missing(init)) {
    init <- NULL
  } else {
    if (length(init) != ncol(X)) stop("wrong length for init argument")
    temp <- X %*% init - sum(colMeans(X) * init) + offset
    # it's okay to have a few underflows, but if all of them are too
    #   small we get all zeros
    if (any(exp(temp) > .Machine$double.xmax) || all(exp(temp) == 0)) {
      stop("initial values lead to overflow or underflow of the exp function")
    }
  }

  # TRUE if coxph.penalty class
  pterms <- rep(FALSE, length(attr(Terms, "term.labels")))

  rname <- rownames(X)

  # if (type == "right") stop("right Surv not yet implemented in coxphGPU.
  #                          Please use `Surv(time1,time2,event)` in formula")

  if(type == "counting"){

  # from agreg.fit.R (survival) / for counting type Surv object
  nvar <- ncol(X)
  event <- Y[, 3]

  if (all(event == 0)) stop("Can't fit a Cox model with 0 failures")

  if (missing(offset) || is.null(offset)) offset <- rep(0.0, nrow(Y))
  if (missing(weights) || is.null(weights)) {
    weights <- rep(1.0, nrow(Y))
  } else if (any(weights <= 0)) {
    stop("Invalid weights, must be >0")
  } else {
    weights <- as.vector(weights)
  }

  # Find rows to be ignored.  We have to match within strata: a
  #  value that spans a death in another stratum, but not it its
  #  own, should be removed.  Hence the per stratum delta
  if (length(istrat) == 0) {
    y1 <- Y[, 1]
    y2 <- Y[, 2]
    strata <- NULL
  } else {
    if (is.numeric(istrat)) {
      strata <- as.integer(istrat)
    } else {
      strata <- as.integer(as.factor(istrat))
    }
    delta <- strata * (1 + max(Y[, 2]) - min(Y[, 1]))
    y1 <- Y[, 1] + delta
    y2 <- Y[, 2] + delta
  }
  event <- Y[, 3] > 0
  dtime <- sort(unique(y2[event]))
  indx1 <- findInterval(y1, dtime)
  indx2 <- findInterval(y2, dtime)
  # indx1 != indx2 for any obs that spans an event time
  ignore <- (indx1 == indx2)
  nused <- sum(!ignore)

  # Sort the data (or rather, get a list of sorted indices)
  #  For both stop and start times, the indices go from last to first
  if (length(strata) == 0) {
    sort.end <- order(ignore, -Y[, 2]) - 1L # indices start at 0 for C code
    sort.start <- order(ignore, -Y[, 1]) - 1L
    strata <- rep(0L, nrow(Y))
  } else {
    sort.end <- order(ignore, strata, -Y[, 2]) - 1L
    sort.start <- order(ignore, strata, -Y[, 1]) - 1L
  }

  if (is.null(nvar) || nvar == 0) {
    # A special case: Null model.  Just return obvious stuff
    #  To keep the C code to a small set, we call the usual routines, but
    #  with a dummy X matrix and 0 iterations
    nvar <- 1
    x <- matrix(as.double(1:nrow(Y)), ncol = 1) # keep the .C call happy
    maxiter <- 0
    nullmodel <- TRUE
    if (length(init) != 0) stop("Wrong length for initial values")
    init <- 0.0 # dummy value to keep a .C call happy (doesn't like 0 length)
  } else {
    nullmodel <- FALSE
    maxiter <- control$iter.max

    # In commentary below because Null value for coxph_R
    #if (is.null(init)) init <- rep(0., nvar)
    #if (length(init) != nvar) stop("Wrong length for initial values")
  }

  # 2021 change: pass in per covariate centering.  This gives
  #  us more freedom to experiment.  Default is to leave 0/1 variables alone

  # It seems y,
  if (is.null(nocenter)) zero.one <- rep(FALSE, ncol(X))
  # zero.one <- apply(X, 2, function(z) all(z %in% nocenter))
  mat <- as.matrix(X)
  zero.one <- colSums(!array(mat %in% nocenter, dim(mat))) == 0

  # the returned value of agfit$coef starts as a copy of init, so make sure
  #  is is a vector and not a matrix; as.double suffices.
  # Solidify the storage mode of other arguments
  storage.mode(Y) <- storage.mode(X) <- "double"
  storage.mode(offset) <- storage.mode(weights) <- "double"

  # # survival routine
  # agfit <- .Call("agfit4", nused,
  #                Y, X, strata, weights,
  #                offset,
  #                as.double(init),
  #                sort.start, sort.end,
  #                as.integer(method=="efron"),
  #                as.integer(maxiter),
  #                as.double(control$eps),
  #                as.double(control$toler.chol),
  #                ifelse(zero.one, 0L, 1L))

  # agfit4 centers variables within strata, so does not return a vector
  #  of means.  Use a fill in consistent with other coxph routines

  agmeans <- ifelse(zero.one, 0, colMeans(X))

  }else if(type == "right"){

    n <-  nrow(Y)
    if (is.matrix(X)) nvar <- ncol(X)
    else {
      if (length(X)==0) nvar <-0
      else nvar <-1
    }
    time <- Y[,1]
    status <- Y[,2]

    # Sort the data (or rather, get a list of sorted indices)
    if (length(istrat)==0) {
      sorted <- order(time)
      strata <- NULL
      newstrat <- as.integer(rep(0,n))
    }
    else {
      sorted <- order(istrat, time)
      strata <- istrat[sorted]
      newstrat <- as.integer(c(1*(diff(as.numeric(strata))!=0), 1))
    }
    if (missing(offset) || is.null(offset)) offset <- rep(0,n)
    if (missing(weights)|| is.null(weights))weights<- rep(1,n)
    else {
      if (any(weights<=0)) stop("Invalid weights, must be >0")
      weights <- weights[sorted]
    }
    stime <- as.double(time[sorted])
    sstat <- as.integer(status[sorted])

    if (nvar==0) {
      # A special case: Null model.
      #  (This is why I need the rownames arg- can't use x' names)
      # Set things up for 0 iterations on a dummy variable
      x <- as.matrix(rep(1.0, n))
      nullmodel <- TRUE
      nvar <- 1
      init <- 0
      maxiter <- 0
    }
    else {
      nullmodel <- FALSE
      maxiter <- control$iter.max
      if (!missing(init) && length(init)>0) {
        if (length(init) != nvar) stop("Wrong length for initial values")
      }
      else init <- rep(0,nvar)
    }

    # 2012 change: individually choose which variable to rescale
    # default: leave 0/1 variables alone
    if (is.null(nocenter)) zero.one <- rep(FALSE, ncol(X))
    else zero.one <- apply(X, 2, function(z) all(z %in% nocenter))

    storage.mode(weights) <- storage.mode(init) <- "double"

  }


  ##############################################################################
  ##############################################################################
  #
  # End of pre-processing of survival::coxph
  #
  # Cox model with python function coxph_R()
  #
  ##############################################################################
  ##############################################################################

  # if (!is.null(istrat)) {
  #   stop("Stratification is not implemented yet in coxphGPU")
  # }

  # if(robust == TRUE)
  #   stop("Robust variance is not implemented yet in coxphGPU")

  db <- .coxphGPU_build_data_Y(formula, type, y1 = y1, y2 = y2, Y = Y,
                               time = time, status = status)
  start <- db$start
  stop <- db$stop
  event <- db$event
  data_Y <- db$data_Y

  # data_Y <- cbind(data_Y,X)

  # return(list(y1=y1,
  #             y=y2,
  #             x = X,
  #             event = Y[,3]))

  # Covariables
  #covar <- assign
  covar <- colnames(X)


  # # remove NA for coxph_R
  # #keep_col <- c(stop, event, colnames(X))
  # keep_col <- c(stop, event, names(covar))
  # data_real <- as.data.frame(data)[,keep_col]


  # return(list(data_real = data_real,
  #             data= data))
  # return(list(dataset = as.data.frame(data),
  #             y = Y,
  #             stime=stime,
  #             sstat=sstat,
  #             x = X))

  # data <- quote(options()$na.action)
  data <- na.omit(data)

  coxfit <- .coxphGPU_call_python(
    data_Y = data_Y, data_X = X, start = start, stop = stop, death = event,
    covars = covar, ties = ties, strata = strata, patient_id = NULL,
    bootstrap = 0, batchsize = 0, maxiter = maxiter,
    init = init, device = device, double_precision = double_precision
  )

  if (is.character(coxfit)) {
    fit <- list(fail = coxfit)
    class(fit) <- "coxphGPU"
    return(fit)
  }

  ##############################################################################
  #
  # Post processing of survival::coxph
  #
  ##############################################################################

  if (is.matrix(X)) {
    nvar <- ncol(X)
  } else {
    if (length(X) == 0) {
      nvar <- 0
    } else {
      nvar <- 1
    }
  }

  coef <- c(coxfit$coef)
  names(coef) <- dimnames(X)[[2]]

  # now only one imat, hessian
  # var <- lapply(c(1:bootstrap), function(x) matrix(coxfit$imat[x, , ],
  #                                                  ncol = ncol(coef)))

  # list ?
  # var <- lapply(c(1:1), function(x) matrix(coxfit$imat[x, ,],
  #                                                  ncol = ncol(coef)))

  # matrix array format
  var <- matrix(coxfit$imat,
                ncol = length(coef))

  # # fit, object to return
  # if (bootstrap > 1 & !isTRUE(all.results)) {
  #   all.results <- FALSE
  #   fit <- list(
  #     coefficients = utils::head(coef, 1), # coef[1,],
  #     var = utils::head(var, 1), # var[[1]],
  #     loglik = coxfit$loglik[1],
  #     loglik_init = coxfit$`loglik init`[1],
  #     score = coxfit$`sctest_init`[1],
  #     means = coxfit$means
  #   )
  # } else {
  #   all.results <- TRUE
  #   fit <- list(
  #     coefficients = coef,
  #     var = var,
  #     loglik = coxfit$loglik,
  #     loglik_init = coxfit$`loglik init`,
  #     score = coxfit$`sctest_init`,
  #     means =coxfit$means
  #   )
  # }

  fit <- list(
    coefficients = coef,
    var = var,
    loglik = c(coxfit$`loglik init`, coxfit$loglik),
    # loglik = c(coxfit$loglik),
    # loglik_init = c(coxfit$`loglik init`),
    score = c(coxfit$`sctest_init`),
    means = c(coxfit$means),
    iter = c(coxfit$iter)
  )

  fit$method <- method
  fit$nbootstraps <- 0 # no bootstrap at fit time; see bootstrap()

  # structure(
  #   fit,
  #   ...,
  #   class = c(class, "coxph")
  # )
  #fit$class <- c(class, "coxph")
  #fit$class <- "coxphGPU"
  #fit$class <- "coxph"
  fit$class <- c("coxphGPU", "coxph")

  if(type == "counting"){

  # return to agreg.fit.R
  lp <- apply(matrix(fit$coefficients, ncol = length(covar)), 1, function(x) c(X %*% x) + offset - sum(x * agmeans))
  if (any(lp > log(.Machine$double.xmax))) {
    # prevent a failure message due to overflow
    #  this occurs with near-infinite coefficients
    temp <- lp + log(.Machine$double.xmax) - (1 + max(lp))
    score <- exp(temp)
  } else {
    score <- exp(lp)
  }


  residuals <- .Call(
    "agmart3", nused,
    Y, score, weights,
    strata,
    sort.start, sort.end,
    as.integer(method == "efron")
  )

  }else if(type == "right"){

    # coef <- coxfit$coef
    lp <- apply(matrix(fit$coefficients, ncol = length(covar)), 1, function(x) c(X %*% x) + offset - sum(x * coxfit$means))
  #  lp <- c(X %*% matrix(fit$coefficients, ncol = length(covar))) + offset - sum(coef * coxfit$means)
   # var <- matrix(coxfit$imat, nvar, nvar)

    # if (coxfit$flag < nvar) which.sing <- diag(var)==0
    # else which.sing <- rep(FALSE,nvar)

    infs <- abs(coxfit$u %*% var)
    # if (maxiter >1) {
    #   if (coxfit$flag == 1000) {
    #     warning("Ran out of iterations and did not converge")
    #     if (max(lp) > 500 || any(!is.finite(infs)))
    #       warning("one or more coefficients may be infinite")
    #   }
    #   else {
    #     infs <- (!is.finite(coxfit$u) |
    #                ((infs > control$eps) &
    #                   infs > control$toler.inf*abs(coef)))
    #     if (any(infs))
    #       warning(paste("Loglik converged before variable ",
    #                     paste((1:nvar)[infs],collapse=","),
    #                     "; coefficient may be infinite. "))
    #   }
    # }
    # if (maxiter > 0) coef[which.sing] <- NA  #leave it be if iter=0 is set

    temp <- lp[sorted]
    if (any(temp > log(.Machine$double.xmax))) {
      # prevent a failure message due to overflow
      #  this occurs with near-infinite coefficients
      temp <- temp + log(.Machine$double.xmax) - (1 + max(temp))
    }
    score <- exp(temp)
    coxres <- .C("coxmart", as.integer(n),
                 as.integer(method=='efron'),
                 stime,
                 sstat,
                 newstrat,
                 as.double(score),
                 as.double(weights),
                 resid=double(n))
    residuals <- double(n)
    residuals[sorted] <- coxres$resid

  }

  names(residuals) <- rname

  fit$linear.predictors <- c(lp)
  fit$residuals <- residuals

  if (is.character(fit)) {
    fit <- list(fail = fit)
    class(fit) <- "coxphGPU"
  } else {
    if (!is.null(fit$coefficients) && any(is.na(fit$coefficients))) {
      vars <- (1:length(fit$coefficients))[is.na(fit$coefficients)]
      msg <- paste(
        "X matrix deemed to be singular; variable",
        paste(vars, collapse = " ")
      )
      if (!singular.ok) stop(msg)
      # else warning(msg)  # stop being chatty
    }
    fit$n <- data.n
    fit$nevent <- sum(Y[, ncol(Y)])
    fit$terms <- Terms
    fit$assign <- assign
    class(fit) <- fit$class
    #     fit$class <- NULL
    #
    # don't compute a robust variance if there are no coefficients
    if (robust && !is.null(fit$coefficients) && !all(is.na(fit$coefficients))) {
      fit$naive.var <- fit$var # fit$var[[1]]
      # a little sneaky here: by calling resid before adding the
      #   na.action method, I avoid having missing re-inserted
      # I also make sure that it doesn't have to reconstruct X and Y
      fit2 <- c(fit, list(x = X, y = Y, weights = weights))
      # fit2$coefficients <- c(utils::head(fit$coefficients,1))
      # fit2$var <- fit$var[[1]]
      # fit2$linear.predictors <- c(fit$linear.predictors)
      class(fit2) <- "coxph"
      if (length(istrat)) fit2$strata <- istrat
      if (length(cluster)) {
        temp <- residuals(fit2,
                          type = "dfbeta", collapse = cluster,
                          weighted = TRUE
        )
        # get score for null model
        if (is.null(init)) {
          fit2$linear.predictors <- 0 * fit$linear.predictors
        } else {
          fit2$linear.predictors <- c(X %*% init)
        }
        temp0 <- residuals(fit2,
                           type = "score", collapse = cluster,
                           weighted = TRUE
        )
      } else {
        temp <- residuals(fit2, type = "dfbeta", weighted = TRUE)
        fit2$linear.predictors <- 0 * fit$linear.predictors
        temp0 <- residuals(fit2, type = "score", weighted = TRUE)
      }
      fit$var <- t(temp) %*% temp
      u <- apply(as.matrix(temp0), 2, sum)
      fit$rscore <- coxph.wtest(t(temp0) %*% temp0, u, control$toler.chol)$test
    }

    # multiple Wald tests needed? requires the variance-covariance matrix for all bootstraps

    # # Wald test
    # if (length(fit$coefficients) && is.null(fit$wald.test)) {
    #   # not for intercept only models, or if test is already done
    #   nabeta <- !is.na(fit$coefficients)
    #   # The init vector might be longer than the betas, for a sparse term
    #   if (is.null(init)) {
    #     temp <- fit$coefficients[nabeta]
    #   } else {
    #     temp <- (fit$coefficients -
    #                init[1:ncol(fit$coefficients)])[nabeta]
    #   }
    #
    #   n_wald.test <- nrow(fit$coefficients)
    #   temp <- matrix(temp, nrow = n_wald.test)
    #
    #   wald.test <- rep(NA, n_wald.test)
    #   for (n_wt in 1:n_wald.test) {
    #     wald.test[n_wt] <- coxph.wtest(
    #       fit$var[[n_wt]][nabeta[n_wt, ], nabeta[n_wt, ]], temp[n_wt, ],
    #       control$toler.chol
    #     )$test
    #   }
    #   fit$wald.test <- wald.test
    # }

    #Wald test


    if (length(fit$coefficients) && is.null(fit$wald.test)) {
      #not for intercept only models, or if test is already done
      nabeta <- !is.na(fit$coefficients)
      # The init vector might be longer than the betas, for a sparse term
      if (is.null(init)) temp <- fit$coefficients[nabeta]
      else temp <- (fit$coefficients -
                      init[1:length(fit$coefficients)])[nabeta]
      fit$wald.test <-  coxph.wtest(fit$var[nabeta,nabeta], temp,
                                    control$toler.chol)$test
    }

    # Concordance.  Done here so that we can use cluster if it is present
    # The returned value is a subset of the full result, partly because it
    #  is all we need, but more for backward compatibility with survConcordance.fit

    # if (length(cluster)) {
    #   temp <- apply(fit$linear.predictors, 2, concordancefit,
    #                 y = Y,
    #                 strata = istrat, weights = weights, cluster = cluster,
    #                 reverse = TRUE, timefix = FALSE
    #   )
    # } else {
    #   temp <- apply(fit$linear.predictors, 2, concordancefit,
    #                 y = Y,
    #                 strata = istrat, weights = weights,
    #                 reverse = TRUE, timefix = FALSE
    #   )
    # }
    #
    # if (is.matrix(temp$count)) {
    #   fit$concordance <- lapply(temp, function(x) {
    #     c(colSums(x$count),
    #       concordance = x$concordance,
    #       std = sqrt(x$var)
    #     )
    #   })
    # } else {
    #   fit$concordance <- lapply(temp, function(x) {
    #     c(x$count,
    #       concordance = x$concordance,
    #       std = sqrt(x$var)
    #     )
    #   })
    # }

    if (length(cluster))
      temp <- concordancefit(Y, fit$linear.predictors, istrat, weights,
                             cluster=cluster, reverse=TRUE,
                             timefix= FALSE)
    else temp <- concordancefit(Y, fit$linear.predictors, istrat, weights,
                                reverse=TRUE, timefix= FALSE)
    if (is.matrix(temp$count))
      fit$concordance <- c(colSums(temp$count), concordance=temp$concordance,
                           std=sqrt(temp$var))
    else fit$concordance <- c(temp$count, concordance=temp$concordance,
                              std=sqrt(temp$var))

    na.action <- prep$na.action
    if (length(na.action)) fit$na.action <- na.action
    if (model) {
      if (length(timetrans)) {
        stop("'model=TRUE' not supported for models with tt terms")
      }
      fit$model <- prep$model
    }
    if (x) {
      fit$x <- X
      if (length(timetrans)) {
        fit$strata <- istrat
      } else if (length(strats)) fit$strata <- strata.keep
    }
    if (y) fit$y <- Y
    fit$timefix <- control$timefix # remember this option
    fit$control <- control # so bootstrap.coxphGPU() can default maxiter later
  }

  if (!is.null(weights) && any(weights != 1)) fit$weights <- weights
  names(fit$means) <- names(fit$coefficients)

  fit$formula <- formula(Terms)
  if (length(xlevels) > 0) fit$xlevels <- xlevels
  fit$contrasts <- contr.save
  if (any(offset != 0)) fit$offset <- offset

  fit$call <- Call
  fit$pterms <- pterms

  return(fit)
}


################################################################################

## Internal helpers, shared between coxphGPU.default() and bootstrap.coxphGPU()
## ------------------------------------------------------------------------

#' @noRd
.coxphGPU_build_data_Y <- function(formula, type, y1 = NULL, y2 = NULL, Y,
                                   time = NULL, status = NULL) {
  # ytemp gives us the original variable names used inside Surv(...) in the
  # formula (e.g. "Start","Stop","Event"), reused below purely as data.table
  # column labels for the bridge to Python.
  ytemp <- all.vars(formula[1:2])
  suppressWarnings(z <- as.numeric(ytemp)) # are any of the elements numeric?
  ytemp <- ytemp[is.na(z)] # toss numerics, e.g. Surv(t, 1-s)

  if (type == "counting") { # if Surv object is counting type
    start <- ytemp[1]
    stop <- ytemp[2]
    event <- ytemp[3]

    data_Y <- data.table(start = y1, stop = y2, status = Y[, 3])
    names(data_Y)[1] <- start
    names(data_Y)[2] <- stop
    names(data_Y)[3] <- event
  } else { # if Surv object is right (Without Start in Surv)
    start <- NULL
    stop <- ytemp[1]
    event <- ytemp[2]

    data_Y <- data.table(stop = time, status = status)
    names(data_Y)[1] <- stop
    names(data_Y)[2] <- event
  }

  list(data_Y = data_Y, start = start, stop = stop, event = event)
}

#' @noRd
.coxphGPU_call_python <- function(data_Y, data_X, start, stop, death, covars,
                                  ties, strata, patient_id, bootstrap,
                                  batchsize, maxiter, init, device,
                                  double_precision) {
  coxph_R <- tryCatch(survivalgpu$coxph_R, error = survivalgpu_unavailable_error)

  coxph_R(
    data_Y = data_Y,
    data_X = data_X,
    start = start,
    stop = stop,
    death = death,
    covars = covars,
    ties = ties,
    strata = strata,
    patient_id = patient_id,
    bootstrap = bootstrap,
    batchsize = batchsize,
    maxiter = maxiter,
    init = init,
    device = device,
    double_precision = double_precision
  )
}

#' @noRd
.coxphGPU_extract_bootstrap_coef <- function(coxfit, coef_names, ncoef) {
  coef_bootstrap <- matrix(coxfit$`bootstrap_coef`, ncol = ncoef)
  colnames(coef_bootstrap) <- coef_names
  coef_bootstrap
}


################################################################################

## coxphGPU Methods ------------------------


#' Print method for coxphGPU object
#'
#' @param x a coxphGPU object
#' @param digits significant digits to print
#' @param signif.stars show stars to highlight small p-values
#' @param ... additional argument(s) for methods.
#'
#' @exportS3Method print coxphGPU
#' @inherit survival::print.coxph return references
print.coxphGPU <- function(x, ..., digits = max(1L, getOption("digits") - 3L),
                           signif.stars = FALSE) {

  NextMethod("print", x)

  if (x$nbootstraps > 0) {
        cat("\n--- Other results with bootstrap with summary() ---")
      }
}


#' Summary method for coxphGPU object
#'
#' Use `summary()` method to see confidence interval for covariates with two
#' process : normal distribution and bootstrap (if [bootstrap()] was called
#' on the object first).
#' @inheritParams survival::summary.coxph
#' @param object a coxphGPU object
#'
#' @return With `summary()` :
#' * `conf.int`:                  a matrix with one row for each coefficient,
#' containing the confidence limits for exp(coef).
#' * `conf.int_bootstrap`:        confidence limits for exp(coef) determined by
#' bootstrap
#' * `logtest, sctest, waldtest`: the overall likelihood ratio, score, and Wald
#' test statistics for the model
#' * `concordance`:               the concordance statistic and its standard
#' error
#' * `rsq`:                       an approximate R^2 based on Nagelkirke
#' (Biometrika 1991).
#' @exportS3Method summary coxphGPU
#' @rdname coxphGPU
summary.coxphGPU <- function(object, ..., conf.int = 0.95, scale = 1) {

  survival_summary <- NextMethod("summary", object)
  survival_summary$nbootstraps <- object$nbootstraps
  survival_summary$conf.int_level = conf.int

    if (object$nbootstraps > 0) {
      probs <- c((1 - conf.int) / 2, 1 - (1 - conf.int) / 2)

      # confidence Interval for coefficients (default 95%)
      survival_summary$conf.int_bootstrap <- apply(object$coef_bootstrap, 2, stats::quantile, p = probs)

    }

  class(survival_summary) <- c("summary.coxphGPU", "summary.coxph")
  return(survival_summary)

}


#' Print summary for coxphGPU object
#'
#' @param x summary.coxphGPU object
#' @param digits significant digits to print
#' @param signif.stars show stars to highlight small p-values
#' @param ... additional argument(s) for methods.
#'
#' @exportS3Method print summary.coxphGPU
#' @noRd
print.summary.coxphGPU <- function(x, ...,
                                   digits = max(getOption("digits") - 3, 3),
                                   signif.stars = getOption("show.signif.stars")) {

  NextMethod("print", x)

  if (x$nbootstraps > 1) {
    cat(" ---------------- \n")
    cat(paste0(
      "Confidence interval with ",
      x$nbootstraps,
      " bootstraps for exp(coef), conf.level = ",
      x$conf.int_level, " :\n"
    ))
    print(signif(t(exp(x$conf.int_bootstrap))))
  }

}


#' Coef method for coxphGPU object
#'
#' @param object coxphGPU object.
#' @param ... additional argument(s) for methods.
#' @exportS3Method coef coxphGPU
#' @noRd
coef.coxphGPU <- function(object, ...) {
  object$coefficients
}


#' Residuals method for coxphGPU
#'
#' @inherit survival::residuals.coxph description references
#' @inheritParams survival::residuals.coxph
#'
#' @seealso [survival::residuals.coxph()]
#'
#' @exportS3Method residuals coxphGPU
#' @examples
#' \dontrun{
#' library(survival)
#' fit <- coxphGPU(Surv(start, stop, event) ~ age + surgery,
#'                 data = heart)
#'
#' # Martingale residuals
#' mresid <- resid(fit, collapse = heart$id)
#' }
residuals.coxphGPU <- function(object, ...,
                               type = c("martingale", "deviance", "score",
                                        "schoenfeld", "dfbeta", "dfbetas",
                                        "scaledsch", "partial"),
                               collapse = FALSE,
                               weighted = (type %in% c("dfbeta", "dfbetas"))){

  NextMethod("residuals", object)

}


#' Predict method for coxphGPU
#'
#' Compute fitted values and regression terms for a model fitted by coxphGPU
#'
#' @inherit survival::predict.coxph references
#' @inheritParams survival::predict.coxph
#' @param object the results of a coxphGPU fit.
#'
#' @seealso [survival::predict.coxph()]
#'
#' @exportS3Method predict coxphGPU
#' @examples
#' \dontrun{
#' library(survival)
#' options(na.action = na.exclude) # retain NA in predictions
#' fit <- coxphGPU(Surv(time, status) ~ age + ph.ecog + strata(inst), lung)
#' predict(fit, type = "lp")
#' predict(fit, type = "expected")
#' predict(fit, type = "risk", se.fit = TRUE)
#' predict(fit, type = "terms", se.fit = TRUE)
#' }
predict.coxphGPU <- function(object, newdata,
                             type = c("lp", "risk", "expected", "terms", "survival"),
                             se.fit = FALSE, na.action = na.pass,
                             terms = names(object$assign), collapse,
                             reference = c("strata", "sample", "zero"), ...) {

  NextMethod("predict", object)

}


#' Bootstrap-based inference for an already-fitted coxphGPU model
#'
#' Adds bootstrap-based confidence intervals to a model already fit by
#' [coxphGPU()], without refitting from the formula. This lets you fit once
#' and decide about bootstrap inference later, while still running all `R`
#' replicate refits as a single batched GPU call to the Python backend.
#'
#' @param object a coxphGPU object.
#' @param ... additional argument(s) for methods.
#'
#' @export
bootstrap <- function(object, ...) {
  UseMethod("bootstrap")
}

#' @param R number of bootstrap replicates.
#' @param patient_id name of the column in `data` that identifies each
#'   patient (subject), so that resampling is performed at the patient
#'   level rather than at the row level. Only relevant to bootstrap, so
#'   (unlike [coxphGPU()]) it's an argument here rather than at the
#'   original fit.
#' @param data the data frame used in the original [coxphGPU()] call (or a
#'   copy with the same row names — see Details).
#' @param batchsize number of bootstrap copies handled at a time; see
#'   [coxphGPU()].
#' @param init starting coefficients for the GPU refits. `TRUE` (the
#'   default) reuses `object`'s own fitted coefficients as a warm start;
#'   `FALSE` starts from zero; or supply a numeric vector directly.
#' @param control a [survival::coxph.control()] object; defaults to the one
#'   used at the original fit (stored on `object`).
#' @param device,double_precision see [coxphGPU()].
#'
#' @details
#' `object` must have been fit with `x = TRUE` (`coxphGPU(..., x = TRUE)`),
#' so its design matrix is available to reuse without re-parsing the
#' formula. `data` must have the same row names as the data frame used in
#' the original call — the ordinary R default, unless explicitly reset —
#' so that `patient_id`'s values can be correctly realigned to the stored
#' design matrix even if the original fit used `subset =` or dropped rows
#' to missing values.
#'
#' @return A copy of `object` with `coef_bootstrap` and `nbootstraps`
#'   updated from the new bootstrap run; every other field (coefficients,
#'   variance, residuals, ...) is left untouched, so [summary.coxphGPU()]
#'   picks up the bootstrap confidence intervals automatically.
#'
#' @rdname bootstrap
#' @exportS3Method bootstrap coxphGPU
#' @examples
#' \dontrun{
#' library(survival)
#' fit <- coxphGPU(Surv(Start, Stop, Event) ~ sex + age,
#'                 data = drugdata, x = TRUE)
#' fit_boot <- bootstrap(fit, R = 1000, patient_id = "Id", data = drugdata,
#'                       batchsize = 200)
#' summary(fit_boot)
#' }
bootstrap.coxphGPU <- function(object, R, patient_id, data, batchsize = 0,
                               init = TRUE, control,
                               device = NULL, double_precision = TRUE, ...) {

  if (is.null(object$x)) {
    stop("bootstrap() requires the design matrix stored on the fitted ",
         "object. Refit with coxphGPU(..., x = TRUE), then call ",
         "bootstrap() again.")
  }
  if (missing(R) || is.null(R) || R < 1) {
    stop("R (number of bootstrap replicates) must be a positive integer.")
  }
  if (missing(patient_id) || missing(data)) {
    stop("patient_id and data are required: patient_id is only needed for ",
         "bootstrap resampling, so it isn't captured at the original ",
         "coxphGPU() call.")
  }

  pid_values <- data[rownames(object$x), patient_id]
  if (anyNA(pid_values)) {
    stop("data does not have the same row names as the fitted object; ",
         "pass the same data frame used in the original coxphGPU() call.")
  }

  Y <- object$y
  type <- attr(Y, "type")
  X <- object$x
  covar <- colnames(X)
  istrat <- if (!is.null(object$strata)) as.integer(object$strata) else NULL

  if (type == "counting") {
    # Mirrors coxphGPU.default()'s own counting-type response construction
    # (the per-stratum time delta that keeps risk sets from spanning
    # strata boundaries), so resampled data is grouped identically.
    if (length(istrat) == 0) {
      y1 <- Y[, 1]
      y2 <- Y[, 2]
      strata <- rep(0L, nrow(Y))
    } else {
      strata <- istrat
      delta <- strata * (1 + max(Y[, 2]) - min(Y[, 1]))
      y1 <- Y[, 1] + delta
      y2 <- Y[, 2] + delta
    }
    storage.mode(X) <- "double"
    db <- .coxphGPU_build_data_Y(object$formula, type, y1 = y1, y2 = y2, Y = Y)
  } else {
    strata <- istrat
    db <- .coxphGPU_build_data_Y(object$formula, type, Y = Y,
                                 time = Y[, 1], status = Y[, 2])
  }

  pid_df <- data.frame(pid_values)
  names(pid_df) <- patient_id
  data_Y <- cbind(pid_df, db$data_Y)

  if (missing(control)) {
    control <- if (!is.null(object$control)) object$control else coxph.control()
  }

  init_vec <- if (isTRUE(init)) {
    unname(object$coefficients)
  } else if (isFALSE(init)) {
    NULL
  } else {
    init
  }

  coxfit <- .coxphGPU_call_python(
    data_Y = data_Y, data_X = X, start = db$start, stop = db$stop,
    death = db$event, covars = covar, ties = object$method, strata = strata,
    patient_id = patient_id, bootstrap = R, batchsize = batchsize,
    maxiter = control$iter.max, init = init_vec,
    device = device, double_precision = double_precision
  )

  if (is.character(coxfit)) {
    stop("bootstrap() failed: ", coxfit)
  }

  fit <- object
  fit$coef_bootstrap <- .coxphGPU_extract_bootstrap_coef(
    coxfit, names(object$coefficients), length(object$coefficients)
  )
  fit$nbootstraps <- R
  fit
}
