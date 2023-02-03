#' Fast Cox Proportional Hazards Regression Model
#'
#' @description Fits a Cox proportional hazards regression model. An extension
#'   to use (or not) your GPU to speed up calculations, in particular for
#'   bootstrap.
#'
#' @param formula a formula object, with the response on the left of a ~
#'   operator, and the terms on the right. The response must be a survival
#'   object as returned by the `survival::Surv` function. For the moment,
#'   `coxphGPU()` only manages the counting type for `Surv` object, i.e. two
#'   `time` argument in the function (Surv(time = start, time2 = stop, event)
#'   for example)
#' @param data a data.frame in which to interpret the variables named in the
#'   formula.
#' @param ties a character string specifying the method for tie handling. If
#'   there are no tied death times all the methods are equivalent. Nearly all
#'   Cox regression programs use the Breslow method by default, but not this
#'   one. The Efron approximation is used as the default here, it is more
#'   accurate when dealing with tied death times, and is as efficient
#'   computationally.
#' @param bootstrap Number of repeats for the bootstrap cross-validation.
#' @param batchsize Number of bootstrap copies that should be handled at a time.
#'   Defaults to 0, which means that we handle all copies at once. If you run
#'   into out of memory errors, please consider using batchsize=100, 10 or 1.
#' @param all.results Post-processing calculations. If TRUE, coxphGPU returns
#'   linears.predictors, wald.test, concordance for all bootstraps. Default to
#'   FALSE if bootstrap.
#' @param na.action a missing-data filter function. This is applied to the
#'   model.frame after any subset argument has been used. Default is
#'   options()\$na.action.
#' @param control Object of class coxph.control specifying iteration limit and
#'   other control options. Default is coxph.control(...).
#' @param singular.ok logical value indicating how to handle collinearity in the
#'   model matrix. If TRUE, the program will automatically skip over columns of
#'   the X matrix that are linear combinations of earlier columns. In this case
#'   the coefficients for such columns will be NA, and the variance matrix will
#'   contain zeros. For ancillary calculations, such as the linear predictor,
#'   the missing coefficients are treated as zeros.
#' @param model logical value: if TRUE, the model frame is returned in component
#'   model.
#' @param x logical value: if TRUE, the x matrix is returned in component x.
#' @param y logical value: if TRUE, the response vector is returned in component
#'   y.
#' @param ... Other arguments for methods.
#'
#' @import stats
#' @import survival
#' @importFrom utils methods
#' @importFrom utils head
#'
#' @return A coxphGPU object representing the fit.
#' @export
#'
#' @references Therneau T (2021). _A Package for Survival Analysis in R_. R
#'   package version 3.2-13
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
#' ## Cox Proportional Hazards without bootstrap
#' coxphGPU(Surv(Start, Stop, Event) ~ sex + age,
#'          data = drugdata,
#'          bootstrap = 1)
#'
#' ## Cox Proportional Hazards with bootstrap
#'
#'  if(use_cuda()){
#'      n_bootstrap <- 1000; batchsize <- 200
#'  }else{
#'      n_bootstrap <- 50  ; batchsize <- 10}
#'
#' coxph_bootstrap <- coxphGPU(Surv(Start, Stop, Event) ~ sex + age,
#'                             data = drugdata,
#'                             bootstrap = n_bootstrap,
#'                             batchsize = batchsize)
#' summary(coxph_bootstrap)
#' }
coxphGPU <- function(formula, data, ties = c("efron", "breslow"), bootstrap = 1,
                     batchsize = 0, all.results = FALSE, na.action, control,
                     singular.ok = TRUE, model = FALSE, x = FALSE, y = TRUE,
                     ...) {
  UseMethod("coxphGPU")
}

#' @return \code{NULL}
#' @noRd
#' @method coxphGPU default
#' @exportS3Method coxphGPU default
coxphGPU.default <- function(formula, data, ties = c("efron", "breslow"),
                             bootstrap = 1, batchsize = 0, all.results = FALSE,
                             na.action, control, singular.ok = TRUE,
                             model = FALSE, x = FALSE, y = TRUE, ...,  weights,
                             subset, init, robust, tt, method = ties, id,
                             cluster, istate, statedata, nocenter=c(-1, 0, 1)){

  if(!missing(weights)) stop("weights are not yet implemented in coxphGPU")
  if(!missing(init)) stop("init is not yet implemented in coxphGPU")
  if(!missing(tt)) stop("tt process is not yet implemented in coxphGPU")

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
    controlargs <- names(formals(coxph.control)) #legal arg names
    indx <- pmatch(names(extraArgs), controlargs, nomatch=0L)
    if (any(indx==0L))
      stop(gettextf("Argument %s not matched",
                    names(extraArgs)[indx==0L]), domain = NA)
  }

  # if no 'control' in input, we take the coxph.control() values
  if (missing(control)) control <- coxph.control(...)

  # Move any cluster() term out of the formula, and make it an argument
  #  instead.  This makes everything easier.  But, I can only do that with
  #  a local copy, doing otherwise messes up future use of update() on
  #  the model object for a user stuck in "+ cluster()" mode.

  if (missing(formula)) stop("a formula argument is required")

  ss <- "cluster"

  # Condition if formula is a list of formula
  if (is.list(formula))

    # We create a term object
    # We take the first formula of the list if is list of formula
    Terms <- if (missing(data)) terms(formula[[1]], specials=ss) else
      terms(formula[[1]], specials=ss, data=data)
  else Terms <- if (missing(data)) terms(formula, specials=ss) else
    terms(formula, specials=ss, data=data)

  ##############################################################################
  # Pre processing if presence of cluster/specials
  ##############################################################################

  # We detect presence of cluster in the formula
  tcl <- attr(Terms, 'specials')$cluster
  if (length(tcl) > 1) stop("a formula cannot have multiple cluster terms")

  if (length(tcl) > 0) {
    factors <- attr(Terms, 'factors')
    if (any(factors[tcl,] >1)) stop("cluster() cannot be in an interaction")
    if (attr(Terms, "response") ==0)
      stop("formula must have a Surv response")

    # Process if cluster in formula and input function
    # if no cluster in input, we take cluster of the formula, else we take cluster of input
    if (is.null(Call$cluster))
      Call$cluster <- attr(Terms, "variables")[[1+tcl]][[2]]
    else warning("cluster appears both in a formula and as an argument, formula term ignored")

    # [.terms is broken at least through R 4.1; use our
    #  local drop.special() function instead.
    terms_v1<-Terms
    Terms <- drop.special(Terms, tcl) # drop.terms() / remove cluster information in Terms
    formula <- Call$formula <- formula(Terms) # change formula in the Call (because Cluster is now in input)
  }

  # ----- end of cluster process -----------------------------------------------

  # create a call to model.frame() that contains the formula (required)
  #  and any other of the relevant optional arguments
  #  but don't evaluate it just yet
  indx <- match(c("formula", "data", "weights", "subset", "na.action",
                  "cluster", "id", "istate"),
                names(Call), nomatch=0)
  if (indx[1] ==0) stop("A formula argument is required")
  tform <- Call[c(1,indx)]
  tform[[1L]] <- quote(stats::model.frame)


  # if the formula is a list, do the first level of processing on it.

  # If formula is a list
  if (is.list(formula)) {
    multiform <- TRUE
    stop("multistate models not yet implemented on coxphGPU")
    dformula <- formula[[1]]   # the default formula for transitions
    if (missing(statedata)) covlist <- parsecovar1(formula[-1])
    else {
      if (!inherits(statedata, "data.frame"))
        stop("statedata must be a data frame")
      if (is.null(statedata$state))
        stop("statedata data frame must contain a 'state' variable")
      covlist <- parsecovar1(formula[-1], names(statedata))
    }

    # create the master formula, used for model.frame
    # the term.labels + reformulate + environment trio is used in [.terms;
    #  if it's good enough for base R it's good enough for me
    tlab <- unlist(lapply(covlist$rhs, function(x)
      attr(terms.formula(x$formula), "term.labels")))
    tlab <- c(attr(terms.formula(dformula), "term.labels"), tlab)
    newform <- reformulate(tlab, dformula[[2]])
    environment(newform) <- environment(dformula)
    formula <- newform
    tform$na.action <- na.pass  # defer any missing value work to later
  }

  # If only one formula
  else {
    multiform <- FALSE   # formula is not a list of expressions
    covlist <- NULL
    dformula <- formula
  }

  # add specials to the formula
  special <- c("strata", "tt", "frailty", "ridge", "pspline")
  tform$formula <- if(missing(data)) terms(formula, special) else
    terms(formula, special, data=data)

  # Make "tt" visible for coxph formulas, without making it visible elsewhere

  # If a tt is specified, we create a new environment and a tt() function
  if (!is.null(attr(tform$formula, "specials")$tt)) {

    coxenv <- new.env(parent= environment(formula))
    assign("tt", function(x) x, envir=coxenv)
    environment(tform$formula) <- coxenv

  }

  # okay, now evaluate the formula
  mf <- eval(tform, parent.frame())
  Terms <- terms(mf)

  # Grab the response variable, and deal with Surv2 objects
  n <- nrow(mf)
  Y <- model.response(mf)
  Y_temp<-Y

  # Process if Y is a Surv2 object
  isSurv2 <- inherits(Y, "Surv2")
  if (isSurv2) {
    # this is Surv2 style data
    # if there were any obs removed due to missing, remake the model frame
    if (length(attr(mf, "na.action"))) {
      tform$na.action <- na.pass
      mf <- eval.parent(tform)
    }
    if (!is.null(attr(Terms, "specials")$cluster))
      stop("cluster() cannot appear in the model statement")
    new <- surv2data(mf)
    mf <- new$mf
    istate <- new$istate
    id <- new$id
    Y <- new$y
    n <- nrow(mf)
  }

  # Process if Y is not a Surv2 object
  else {
    if (!is.Surv(Y)) stop("Response must be a survival object")
    id <- model.extract(mf, "id")
    istate <- model.extract(mf, "istate")
  }

  if (n==0) stop("No (non-missing) observations")

  type <- attr(Y, "type")
  # several types : right, left, counting, etc...

  multi <- FALSE
  if (type=="mright" || type == "mcounting")
    multi <- TRUE

  else if (type!='right' && type!='counting')
    stop(paste("Cox model doesn't support \"", type,
               "\" survival data", sep=''))
  data.n <- nrow(Y)   #remember this before any time transforms

  if (!multi && multiform)
    stop("formula is a list but the response is not multi-state")
  if (multi && length(attr(Terms, "specials")$frailty) >0)
    stop("multi-state models do not currently support frailty terms")
  if (multi && length(attr(Terms, "specials")$pspline) >0)
    stop("multi-state models do not currently support pspline terms")
  if (multi && length(attr(Terms, "specials")$ridge) >0)
    stop("multi-state models do not currently support ridge penalties")

  if (control$timefix) Y <- aeqSurv(Y) # timefix == TRUE by default in coxph.control()
  # aeqSurv : deal with the issue of ties that get incorrectly broken due to floating point imprecision

  # Formula check
  if (length(attr(Terms, 'variables')) > 2) { # a ~1 formula has length 2
    ytemp <- terms.inner(formula[1:2])
    suppressWarnings(z <- as.numeric(ytemp)) # are any of the elements numeric?
    ytemp <- ytemp[is.na(z)]  # toss numerics, e.g. Surv(t, 1-s)
    xtemp <- terms.inner(formula[-2])
    if (any(!is.na(match(xtemp, ytemp))))
      warning("a variable appears on both the left and right sides of the formula")
  }

  # The time transform will expand the data frame mf.  To do this
  #  it needs Y and the strata.  Everything else (cluster, offset, weights)
  #  should be extracted after the transform

  # tt (time transform)
  strats <- attr(Terms, "specials")$strata
  hasinteractions <- FALSE
  dropterms <- NULL
  if (length(strats)) { # if at least one strata
    stemp <- untangle.specials(Terms, 'strata', 1) # detect strata and index in terms matrix
    if (length(stemp$vars)==1) strata.keep <- mf[[stemp$vars]] # if only one stratification, strata.keep worth strat levels
    else strata.keep <- strata(mf[,stemp$vars], shortlabel=TRUE) # if several strat, as many levels as combinations of strat
    istrat <- as.integer(strata.keep)
    # istrat : vector to identify stratification

    for (i in stemp$vars) {  #multiple strata terms are allowed
      # The factors attr has one row for each variable in the frame, one
      #   col for each term in the model.  Pick rows for each strata
      #   var, and find if it participates in any interactions.
      if (any(attr(Terms, 'order')[attr(Terms, "factors")[i,] >0] >1))
        hasinteractions <- TRUE
    }
    if (!hasinteractions) dropterms <- stemp$terms
  } else istrat <- NULL
  # istrat = NULL if no stratification

  if (hasinteractions && multi)
    stop("multi-state coxph does not support strata*covariate interactions")

  ##############################################################################
  # tt process
  ##############################################################################

  timetrans <- attr(Terms, "specials")$tt
  if (missing(tt)) tt <- NULL
  if (length(timetrans)) {
    if (multi || isSurv2) stop("the tt() transform is not implemented for multi-state or Surv2 models")
    timetrans <- untangle.specials(Terms, 'tt') # detect tt index in terms matrix
    ntrans <- length(timetrans$terms) # number of tt()

    if (is.null(tt)) {
      tt <- function(x, time, riskset, weights){ #default to O'Brien's logit rank
        obrien <- function(x) {
          r <- rank(x)
          (r-.5)/(.5+length(r)-r)
        }
        unlist(tapply(x, riskset, obrien))
      }
    }
    if (is.function(tt)) tt <- list(tt)  #single function becomes a list

    if (is.list(tt)) {
      if (any(!sapply(tt, is.function)))
        stop("The tt argument must contain function or list of functions")
      if (length(tt) != ntrans) {
        if (length(tt) ==1) {
          temp <- vector("list", ntrans)
          for (i in 1:ntrans) temp[[i]] <- tt[[1]]
          tt <- temp
        }
        else stop("Wrong length for tt argument")
      }
    }
    else stop("The tt argument must contain a function or list of functions")

    if (ncol(Y)==2) {
      if (length(strats)==0) {
        sorted <- order(-Y[,1], Y[,2])
        newstrat <- rep.int(0L, nrow(Y))
        newstrat[1] <- 1L
      }
      else {
        sorted <- order(istrat, -Y[,1], Y[,2])
        #newstrat marks the first obs of each strata
        newstrat <-  as.integer(c(1, 1*(diff(istrat[sorted])!=0)))
      }
      if (storage.mode(Y) != "double") storage.mode(Y) <- "double"
      counts <- .Call("coxcount1", Y[sorted,],
                      as.integer(newstrat))
      tindex <- sorted[counts$index]
    }
    else {
      if (length(strats)==0) {
        sort.end  <- order(-Y[,2], Y[,3])
        sort.start<- order(-Y[,1])
        newstrat  <- c(1L, rep(0, nrow(Y) -1))
      }
      else {
        sort.end  <- order(istrat, -Y[,2], Y[,3])
        sort.start<- order(istrat, -Y[,1])
        newstrat  <- c(1L, as.integer(diff(istrat[sort.end])!=0))
      }
      if (storage.mode(Y) != "double") storage.mode(Y) <- "double"
      counts <- .Call("coxcount2", Y,
                      as.integer(sort.start -1L),
                      as.integer(sort.end -1L),
                      as.integer(newstrat))
      tindex <- counts$index
    }
    Y <- Surv(rep(counts$time, counts$nrisk), counts$status)
    type <- 'right'  # new Y is right censored, even if the old was (start, stop]

    mf <- mf[tindex,]
    istrat <- rep(1:length(counts$nrisk), counts$nrisk)
    weights <- model.weights(mf)
    if (!is.null(weights) && any(!is.finite(weights)))
      stop("weights must be finite")

    tcall <- attr(Terms, 'variables')[timetrans$terms+2]
    pvars <- attr(Terms, 'predvars')
    pmethod <- sub("makepredictcall.", "", as.vector(methods("makepredictcall")))
    for (i in 1:ntrans) {
      newtt <- (tt[[i]])(mf[[timetrans$var[i]]], Y[,1], istrat, weights)
      mf[[timetrans$var[i]]] <- newtt
      nclass <- class(newtt)
      if (any(nclass %in% pmethod)) { # It has a makepredictcall method
        dummy <- as.call(list(as.name(class(newtt)[1]), tcall[[i]][[2]]))
        ptemp <- makepredictcall(newtt, dummy)
        pvars[[timetrans$terms[i]+2]] <- ptemp
      }
    }
    attr(Terms, "predvars") <- pvars
  }

  # ------- end of tt process ----------------------------------------


  ##############################################################################
  # Routines depending on id, cluster, weights et robust presence
  ##############################################################################

  # Return a list with levels/labels of factor covariates
  xlevels <- .getXlevels(Terms, mf)

  # grab the cluster, if present.  Using cluster() in a formula is no
  #  longer encouraged
  cluster <- model.extract(mf, "cluster") # cluster vector
  weights <- model.weights(mf) # weight vector

  # The user can call with cluster, id, robust, or any combination
  # Default for robust: if cluster or any id with > 1 event or
  #  any weights that are not 0 or 1, then TRUE
  # If only id, treat it as the cluster too
  has.cluster <- !(missing(cluster) || length(cluster)==0)
  has.id <-      !(missing(id) || length(id)==0)
  has.rwt<-      (!is.null(weights) && any(weights != floor(weights)))
  #has.rwt<- FALSE  # we are rethinking this
  has.robust <-  (!missing(robust) && !is.null(robust))  # arg present
  if (has.id) id <- as.factor(id)

  if (missing(robust) || is.null(robust)) {
    if (has.cluster || has.rwt ||
        (has.id && (multi || anyDuplicated(id[Y[,ncol(Y)]==1])))){
      robust <- TRUE
    }else{
      robust <- FALSE
    }
  }
  if(robust == TRUE) stop("robust variance is not yet implemented in coxphGPU")

  if (!is.logical(robust)) stop("robust must be TRUE/FALSE")

  if (has.cluster) {
    if (!robust) {
      warning("cluster specified with robust=FALSE, cluster ignored")
      ncluster <- 0
      clname <- NULL
    }
    else {
      if (is.factor(cluster)) {
        clname <- levels(cluster)
        cluster <- as.integer(cluster)
      } else {
        clname  <- sort(unique(cluster))
        cluster <- match(cluster, clname)
      }
      ncluster <- length(clname)
    }
  } else {
    if (robust && has.id) {
      # treat the id as both identifier and clustering
      clname <- levels(id)
      cluster <- as.integer(id)
      ncluster <- length(clname)
    }
    else {
      ncluster <- 0  # has neither
    }
  }

  # if the user said "robust", (time1,time2) data, and no cluster or
  #  id, complain about it
  if (robust && is.null(cluster)) {
    if (ncol(Y) ==2 || !has.robust) cluster <- seq.int(1, nrow(mf))
    else stop("one of cluster or id is needed")
  }

  contrast.arg <- NULL  #due to shared code with model.matrix.coxph
  attr(Terms, "intercept") <- 1  # always have a baseline hazard

  # ---- end of routine over id, cluster,and robust ----------------------------

  ##############################################################################
  # Conditions if multistate model
  ##############################################################################

  if (multi) {
    # check for consistency of the states, and create a transition
    #  matrix
    if (length(id)==0)
      stop("an id statement is required for multi-state models")

    mcheck <- survcheck2(Y, id, istate)
    # error messages here
    if (mcheck$flag["overlap"] > 0)
      stop("data set has overlapping intervals for one or more subjects")

    transitions <- mcheck$transitions
    istate <- mcheck$istate
    states <- mcheck$states

    #  build tmap, which has one row per term, one column per transition
    if (missing(statedata))
      covlist2 <- parsecovar2(covlist, NULL, dformula= dformula,
                              Terms, transitions, states)
    else covlist2 <- parsecovar2(covlist, statedata, dformula= dformula,
                                 Terms, transitions, states)
    tmap <- covlist2$tmap
    if (!is.null(covlist)) {
      # first vector will be true if there is at least 1 transition for which all
      #  covariates are present, second if there is at least 1 for which some are not
      good.tran <- bad.tran <- rep(FALSE, nrow(Y))
      # We don't need to check interaction terms
      termname <- rownames(attr(Terms, 'factors'))
      trow <- (!is.na(match(rownames(tmap), termname)))

      # create a missing indicator for each term
      termiss <- matrix(0L, nrow(mf), ncol(mf))
      for (i in 1:ncol(mf)) {
        xx <- is.na(mf[[i]])
        if (is.matrix(xx)) termiss[,i] <- apply(xx, 1, any)
        else termiss[,i] <- xx
      }

      for (i in levels(istate)) {
        rindex <- which(istate ==i)
        j <- which(covlist2$mapid[,1] == match(i, states))  #possible transitions
        for (jcol in j) {
          k <- which(trow & tmap[,jcol] > 0)  # the terms involved in that
          bad.tran[rindex] <- (bad.tran[rindex] |
                                 apply(termiss[rindex, k, drop=FALSE], 1, any))
          good.tran[rindex] <- (good.tran[rindex] |
                                  apply(!termiss[rindex, k, drop=FALSE], 1, all))
        }
      }
      n.partially.used <- sum(good.tran & bad.tran & !is.na(Y))
      omit <- (!good.tran & bad.tran) | is.na(Y)
      if (all(omit)) stop("all observations deleted due to missing values")
      temp <- setNames(seq(omit)[omit], attr(mf, "row.names")[omit])
      attr(temp, "class") <- "omit"
      mf <- mf[!omit,, drop=FALSE]
      attr(mf, "na.action") <- temp
      Y <- Y[!omit]
      id <- id[!omit]
      if (length(istate)) istate <- istate[!omit]  # istate can be NULL
    }
  }
  # ----- end of conditions multistate model -----------------------------------



  ##############################################################################
  # Routines over strata, offset, and weights
  ##############################################################################

  # Routine strata

  # dropterms if stratification : remove strata attr in terms, and remove strata
  # in X matrix
  if (length(dropterms)) {
    Terms2 <- Terms[-dropterms]
    X <- model.matrix(Terms2, mf, constrasts.arg=contrast.arg)
    # we want to number the terms wrt the original model matrix
    temp <- attr(X, "assign")
    shift <- sort(dropterms)
    for (i in seq(along.with=shift))
      temp <- temp + 1*(shift[i] <= temp)
    attr(X, "assign") <- temp
  }
  else X <- model.matrix(Terms, mf, contrasts.arg=contrast.arg)


  # drop the intercept after the fact, and also drop strata if necessary
  Xatt <- attributes(X)
  if (hasinteractions) adrop <- c(0, untangle.specials(Terms, "strata")$terms)
  else adrop <- 0
  xdrop <- Xatt$assign %in% adrop  #columns to drop (always the intercept)
  X <- X[, !xdrop, drop=FALSE]
  attr(X, "assign") <- Xatt$assign[!xdrop]
  attr(X, "contrasts") <- Xatt$contrasts

  # offset routine
  offset <- model.offset(mf)
  if (is.null(offset) | all(offset==0)) offset <- rep(0., nrow(mf))
  else if (any(!is.finite(offset) | !is.finite(exp(offset))))
    stop("offsets must lead to a finite risk score")

  # weights routine
  weights <- model.weights(mf)
  if (!is.null(weights) && any(!is.finite(weights)))
    stop("weights must be finite")

  # ---- end of routines -------------------------------------------------------

  assign <- attrassign(X, Terms)
  contr.save <- attr(X, "contrasts")

  # if no event, return result with NA / 0
  if (sum(Y[, ncol(Y)]) == 0) {
    # No events in the data!
    ncoef <- ncol(X)
    ctemp <- rep(NA, ncoef)
    names(ctemp) <- colnames(X)
    concordance= c(concordant=0, discordant=0, tied.x=0, tied.y=0, tied.xy=0,
                   concordance=NA, std=NA, timefix=FALSE)
    rval <- list(coefficients= ctemp,
                 var = matrix(0.0, ncoef, ncoef),
                 loglik=c(0,0),
                 score =0,
                 iter =0,
                 linear.predictors = offset,
                 residuals = rep(0.0, data.n),
                 means = colMeans(X), method=method,
                 n = data.n, nevent=0, terms=Terms, assign=assign,
                 concordance=concordance,  wald.test=0.0,
                 y = Y, call=Call)
    class(rval) <- "coxph"
    return(rval)
  }

  ##############################################################################
  # Conditions if multi = TRUE
  ##############################################################################

  if (multi) {
    if (length(strats) >0) {
      stratum_map <- tmap[c(1L, strats),] # strats includes Y, + tmap has an extra row
      stratum_map[-1,] <- ifelse(stratum_map[-1,] >0, 1L, 0L)
      if (nrow(stratum_map) > 2) {
        temp <- stratum_map[-1,]
        if (!all(apply(temp, 2, function(x) all(x==0) || all(x==1)))) {
          # the hard case: some transitions use one strata variable, some
          #  transitions use another.  We need to keep them separate
          strata.keep <- mf[,strats]  # this will be a data frame
          istrat <- sapply(strata.keep, as.numeric)
        }
      }
    }
    else stratum_map <- tmap[1,,drop=FALSE]
    cmap <- parsecovar3(tmap, colnames(X), attr(X, "assign"), covlist2$phbaseline)
    xstack <- stacker(cmap, stratum_map, as.integer(istate), X, Y, strata=istrat,
                      states=states)

    rkeep <- unique(xstack$rindex)
    transitions <- survcheck2(Y[rkeep,], id[rkeep], istate[rkeep])$transitions

    X <- xstack$X
    Y <- xstack$Y
    istrat <- xstack$strata
    if (length(offset)) offset <- offset[xstack$rindex]
    if (length(weights)) weights <- weights[xstack$rindex]
    if (length(cluster)) cluster <- cluster[xstack$rindex]
    t2 <- tmap[-c(1, strats),,drop=FALSE]
    r2 <- row(t2)[!duplicated(as.vector(t2)) & t2 !=0]
    c2 <- col(t2)[!duplicated(as.vector(t2)) & t2 !=0]
    a2 <- lapply(seq(along.with=r2), function(i) {cmap[assign[[r2[i]]], c2[i]]})
    tab <- table(r2)
    count <- tab[r2]
    names(a2) <- ifelse(count==1, row.names(t2)[r2],
                        paste(row.names(t2)[r2], colnames(cmap)[c2], sep="_"))
    assign <- a2
  }

  # ----- end of conditions if multi = TRUE ------------------------------------


  # infinite covariates are not screened out by the na.omit routines
  #  But this needs to be done after the multi-X part
  if (!all(is.finite(X)))
    stop("data contains an infinite predictor")

  # init is checked after the final X matrix has been made
  if (missing(init)) init <- NULL
  else {
    if (length(init) != ncol(X)) stop("wrong length for init argument")
    temp <- X %*% init - sum(colMeans(X) * init) + offset
    # it's okay to have a few underflows, but if all of them are too
    #   small we get all zeros
    if (any(exp(temp) > .Machine$double.xmax) || all(exp(temp)==0))
      stop("initial values lead to overflow or underflow of the exp function")
  }

  # TRUE if coxph.penalty class
  pterms <- sapply(mf, inherits, 'coxph.penalty')

  rname <- row.names(mf)

  if(type == "right") stop("right Surv not yet implemented in coxphGPU.
                           Please use `Surv(time1,time2,event)` in formaula")

  # from agreg.fit.R (survival) / for counting type Surv object
  nvar <- ncol(X)
  event <- Y[,3]
  if (all(event==0)) stop("Can't fit a Cox model with 0 failures")

  if (missing(offset) || is.null(offset)) offset <- rep(0.0, nrow(Y))
  if (missing(weights)|| is.null(weights))weights<- rep(1.0, nrow(Y))
  else if (any(weights<=0)) stop("Invalid weights, must be >0")
  else weights <- as.vector(weights)

  # Find rows to be ignored.  We have to match within strata: a
  #  value that spans a death in another stratum, but not it its
  #  own, should be removed.  Hence the per stratum delta
  if (length(istrat) ==0) {y1 <- Y[,1]; y2 <- Y[,2]; strata = NULL}
  else  {
    if (is.numeric(istrat)) strata <- as.integer(istrat)
    else strata <- as.integer(as.factor(istrat))
    delta  <-  strata* (1+ max(Y[,2]) - min(Y[,1]))
    y1 <- Y[,1] + delta
    y2 <- Y[,2] + delta
  }
  event <- Y[,3] > 0
  dtime <- sort(unique(y2[event]))
  indx1 <- findInterval(y1, dtime)
  indx2 <- findInterval(y2, dtime)
  # indx1 != indx2 for any obs that spans an event time
  ignore <- (indx1 == indx2)
  nused  <- sum(!ignore)

  # Sort the data (or rather, get a list of sorted indices)
  #  For both stop and start times, the indices go from last to first
  if (length(strata)==0) {
    sort.end  <- order(ignore, -Y[,2]) -1L #indices start at 0 for C code
    sort.start<- order(ignore, -Y[,1]) -1L
    strata <- rep(0L, nrow(Y))
  }
  else {
    sort.end  <- order(ignore, strata, -Y[,2]) -1L
    sort.start<- order(ignore, strata, -Y[,1]) -1L
  }

  if (is.null(nvar) || nvar==0) {
    # A special case: Null model.  Just return obvious stuff
    #  To keep the C code to a small set, we call the usual routines, but
    #  with a dummy X matrix and 0 iterations
    nvar <- 1
    x <- matrix(as.double(1:nrow(Y)), ncol=1)  #keep the .C call happy
    maxiter <- 0
    nullmodel <- TRUE
    if (length(init) !=0) stop("Wrong length for inital values")
    init <- 0.0  #dummy value to keep a .C call happy (doesn't like 0 length)
  }
  else {
    nullmodel <- FALSE
    maxiter <- control$iter.max

    if (is.null(init)) init <- rep(0., nvar)
    if (length(init) != nvar) stop("Wrong length for inital values")
  }

  # 2021 change: pass in per covariate centering.  This gives
  #  us more freedom to experiment.  Default is to leave 0/1 variables alone
  if (is.null(nocenter)) zero.one <- rep(FALSE, ncol(X))
  zero.one <- apply(X, 2, function(z) all(z %in% nocenter))

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

  ##############################################################################
  ##############################################################################
  #
  # End of pre-processing of survival::coxph
  #
  # Cox model with python function coxph_R()
  #
  ##############################################################################
  ##############################################################################

  if(!is.null(istrat))
    stop("Stratification is not implemented yet in coxphGPU")

  # if(robust == TRUE)
  #   stop("Robust variance is not implemented yet in coxphGPU")

  # Variable 'Stop' and 'Event' for coxph_R
  if(type == "counting"){ # if Surv object is counting type
    stop = ytemp[2]
    event = ytemp[3]
  } else { # if Surv object is right (Without Start in Surv)
    stop = ytemp[1]
    event = ytemp[2]
  }

  # Covariables
  covar<-assign

  # Python coxph
  survivalgpu <- use_survivalGPU()
  coxph_R <- survivalgpu$coxph_R
  coxfit <- coxph_R(data,
                    stop,
                    event,
                    covar,
                    ties=ties,
                    bootstrap=bootstrap,
                    batchsize=batchsize)
  # maxiter = maxiter (add maxiter argument in coxph_R)
  # doscale


  ##############################################################################
  #
  # Post processing of survival::coxph
  #
  ##############################################################################

  if (is.matrix(X))
    nvar <- ncol(X)
  else {
    if (length(X)==0)
      nvar <-0
    else
      nvar <-1
  }

  coef           = coxfit$coef
  colnames(coef) = dimnames(X)[[2]]
  var            = lapply(c(1:bootstrap),function(x) coxfit$imat[x,,])

  # fit, object to return
  if(bootstrap > 1 & !isTRUE(all.results)){

    all.results = FALSE
    fit <- list(coefficients  = utils::head(coef,1), # coef[1,],
                var           = utils::head(var,1), # var[[1]],
                loglik        = coxfit$loglik[1],
                loglik_init   = coxfit$`loglik init`[1],
                score         = coxfit$`sctest init`[1],
                means         = coxfit$means)
  }else{

    all.results = TRUE
    fit <- list(coefficients  = coef,
                var           = var,
                loglik        = coxfit$loglik,
                loglik_init   = coxfit$`loglik init`,
                score         = coxfit$`sctest init`,
                means         = coxfit$means)
  }

  fit$method      = method
  fit$nbootstraps = bootstrap
  if(bootstrap > 1) fit$coef_bootstrap = coef
  fit$class       = 'coxphGPU'


  # return to agreg.fit.R
  lp <- apply(matrix(fit$coefficients, ncol = length(covar)),1,function(x) c(X %*% x) + offset - sum(x*agmeans))
  if (any(lp > log(.Machine$double.xmax))) {
    # prevent a failure message due to overflow
    #  this occurs with near-infinite coefficients
    temp <- lp + log(.Machine$double.xmax) - (1 + max(lp))
    score <- exp(temp)
  } else score <- exp(lp)

  residuals <- .Call("agmart3", nused,
                     Y, score, weights,
                     strata,
                     sort.start, sort.end,
                     as.integer(method=='efron'))

  names(residuals) <- rname

  fit$linear.predictors = lp
  fit$residuals         = residuals

  if (is.character(fit)) {
    fit <- list(fail = fit)
    class(fit) <- 'coxphGPU'
  }
  else {
    if (!is.null(fit$coefficients) && any(is.na(fit$coefficients))) {
      vars <- (1:length(fit$coefficients))[is.na(fit$coefficients)]
      msg <-paste("X matrix deemed to be singular; variable",
                  paste(vars, collapse=" "))
      if (!singular.ok) stop(msg)
      # else warning(msg)  # stop being chatty
    }
    fit$n      = data.n
    fit$nevent = sum(Y[,ncol(Y)])
    fit$terms  = Terms
    fit$assign = assign
    class(fit) = fit$class
    #     fit$class <- NULL
    #
    # don't compute a robust variance if there are no coefficients
    if (robust && !is.null(fit$coefficients) && !all(is.na(fit$coefficients))) {
      fit$naive.var <- fit$var # fit$var[[1]]
      # a little sneaky here: by calling resid before adding the
      #   na.action method, I avoid having missings re-inserted
      # I also make sure that it doesn't have to reconstruct X and Y
      fit2 <- c(fit, list(x=X, y=Y, weights=weights))
      # fit2$coefficients <- c(utils::head(fit$coefficients,1))
      # fit2$var <- fit$var[[1]]
      # fit2$linear.predictors <- c(fit$linear.predictors)
      class(fit2)<-'coxph'
      if (length(istrat)) fit2$strata <- istrat
      if (length(cluster)) {
        temp <- residuals(fit2, type='dfbeta', collapse=cluster,
                          weighted=TRUE)
        # get score for null model
        if (is.null(init))
          fit2$linear.predictors <- 0*fit$linear.predictors
        else fit2$linear.predictors <- c(X %*% init)
        temp0 <- residuals(fit2, type='score', collapse=cluster,
                           weighted=TRUE)
      }
      else {
        temp <- residuals(fit2, type='dfbeta', weighted=TRUE)
        fit2$linear.predictors <- 0*fit$linear.predictors
        temp0 <- residuals(fit2, type='score', weighted=TRUE)
      }
      fit$var <- t(temp) %*% temp
      u <- apply(as.matrix(temp0), 2, sum)
      fit$rscore <- coxph.wtest(t(temp0)%*%temp0, u, control$toler.chol)$test
    }

    #Wald test
    if (length(fit$coefficients) && is.null(fit$wald.test)) {
      #not for intercept only models, or if test is already done
      nabeta <- !is.na(fit$coefficients)
      # The init vector might be longer than the betas, for a sparse term
      if (is.null(init)) temp <- fit$coefficients[nabeta]
      else temp <- (fit$coefficients -
                      init[1:ncol(fit$coefficients)])[nabeta]

      n_wald.test <- nrow(fit$coefficients)
      temp <- matrix(temp, nrow = n_wald.test)

        wald.test <- rep(NA,n_wald.test)
        for (n_wt in 1:n_wald.test) {
          wald.test[n_wt] <-  coxph.wtest(fit$var[[n_wt]][nabeta[n_wt,],nabeta[n_wt,]], temp[n_wt,],
                                          control$toler.chol)$test
        }
        fit$wald.test <- wald.test
    }

    # Concordance.  Done here so that we can use cluster if it is present
    # The returned value is a subset of the full result, partly because it
    #  is all we need, but more for backward compatability with survConcordance.fit
    if (length(cluster))
      temp <- apply(fit$linear.predictors, 2, concordancefit,  y = Y,
                    strata = istrat, weights = weights, cluster = cluster,
                    reverse = TRUE, timefix = FALSE)

    else temp <- apply(fit$linear.predictors, 2, concordancefit, y = Y,
                       strata = istrat, weights = weights,
                       reverse = TRUE, timefix = FALSE)

    if (is.matrix(temp$count))
      fit$concordance <- lapply(temp, function(x) c(colSums(x$count),
                                                    concordance = x$concordance,
                                                    std = sqrt(x$var)))

    else
      fit$concordance <- lapply(temp, function(x) c(x$count,
                                                    concordance = x$concordance,
                                                    std = sqrt(x$var)))

    na.action <- attr(mf, "na.action")
    if (length(na.action)) fit$na.action <- na.action
    if (model) {
      if (length(timetrans)) {
        stop("'model=TRUE' not supported for models with tt terms")
      }
      fit$model <- mf
    }
    if (x)  {
      fit$x <- X
      if (length(timetrans)) fit$strata <- istrat
      else if (length(strats)) fit$strata <- strata.keep
    }
    if (y)  fit$y <- Y
    fit$timefix <- control$timefix  # remember this option
  }

  if (!is.null(weights) && any(weights!=1)) fit$weights <- weights
  if (multi) {
    fit$transitions <- transitions
    fit$states <- states
    fit$cmap <- cmap
    fit$stratum_map <- stratum_map   # why not 'stratamap'?  Confusion with fit$strata
    fit$resid <- rowsum(fit$resid, xstack$rindex)
    # add a suffix to each coefficent name.  Those that map to multiple transitions
    #  get the first transition they map to
    single <- apply(cmap, 1, function(x) all(x %in% c(0, max(x)))) #only 1 coef
    cindx <- col(cmap)[match(1:length(fit$coefficients), cmap)]
    rindx <- row(cmap)[match(1:length(fit$coefficients), cmap)]
    suffix <- ifelse(single[rindx], "", paste0("_", colnames(cmap)[cindx]))
    newname <- paste0(names(fit$coefficients), suffix)
    if (any(covlist2$phbaseline > 0)) {
      # for proporional baselines, use a better name
      base  <- colnames(tmap)[covlist2$phbaseline]
      child <- colnames(tmap)[which(covlist2$phbaseline >0)]
      indx <- 1 + length(newname) - length(base):1 # coefs are the last ones
      newname[indx] <-  paste0("ph(", child, "/", base, ")")
    }
    names(fit$coefficients) <- newname
    if (x) fit$strata <- istrat  # save the expanded strata
    class(fit) <- c("coxphms", class(fit))
  }
  names(fit$means) <- names(fit$coefficients)

  fit$formula <- formula(Terms)
  if (length(xlevels) >0) fit$xlevels <- xlevels
  fit$contrasts <- contr.save
  if (any(offset !=0)) fit$offset <- offset

  fit$all.results = all.results
  fit$call        = Call
  fit$pterms      = pterms

  return(fit)
}


################################################################################

## coxphGPU Methods ------------------------

#' Print method for coxphGPU object
#'
#' @param object coxphGPU object
#' @param digits digits
#' @param signif.stars Stars to see signif
#' @param ... additional argument(s) for methods.
#' @exportS3Method print coxphGPU
#' @noRd
print.coxphGPU <- function(x, ..., digits=max(1L, getOption("digits") - 3L),
                           signif.stars=FALSE){

  if (!is.null(cl<- x$call)) {
    cat("Call:\n")
    dput(cl)
    cat("\n")
  }
  if (!is.null(x$fail)) {
    cat(" Coxph failed.", x$fail, "\n")
    return()
  }
  savedig <- options(digits = digits)
  on.exit(options(savedig))

  coef <- matrix(x$coefficients[1,])
  if(length(x$var[[1]])>1){
    se <- matrix(sqrt(diag(x$var[[1]])))
  }else{
    se <- sqrt(x$var[[1]])
  }
  if(is.null(coef) | is.null(se))
    stop("Input is not valid")

  if (is.null(x$naive.var)) {
    tmp <- cbind(coef,exp(coef), se, coef/se,
                 pchisq((coef/se)^2, 1, lower.tail=FALSE))
    dimnames(tmp) <- list(colnames(x$coefficients), c("coef", "exp(coef)",
                                                      "se(coef)", "z", "p"))
  }
  else {
    nse <- sqrt(diag(x$naive.var))
    tmp <- cbind(coef, exp(coef), nse, se, coef/se,
                 pchisq((coef/se)^2, 1, lower.tail=FALSE))
    dimnames(tmp) <- list(names(coef), c("coef", "exp(coef)",
                                         "se(coef)", "robust se", "z", "p"))
  }

  if (inherits(x, "coxphms")) {
    # print it group by group
    # lazy: I don't want to type x$cmap many times
    #  remove transitions with no covariates
    cmap <- x$cmap[, colSums(x$cmap) > 0, drop=FALSE]
    cname <- colnames(cmap)
    printed <- rep(FALSE, length(cname))
    for (i in 1:length(cname)) {
      # if multiple colums of tmat are identical, only print that
      #  set of coefficients once
      if (!printed[i]) {
        j <- apply(cmap, 2, function(x) all(x == cmap[,i]))
        printed[j] <- TRUE

        tmp2 <- tmp[cmap[,i],, drop=FALSE]
        names(dimnames(tmp2)) <- c(paste(cname[j], collapse=", "), "")
        # restore character row names
        rownames(tmp2) <- rownames(cmap)[cmap[,i]>0]
        printCoefmat(tmp2, digits=digits, P.values=TRUE,
                     has.Pvalue=TRUE,
                     signif.stars = signif.stars, ...)
        cat("\n")
      }
    }

    cat(" States: ", paste(paste(seq(along.with=x$states), x$states, sep='= '),
                           collapse=", "), '\n')
    # cat(" States: ", paste(x$states, collapse=", "), '\n')
    if (FALSE) { # alternate forms, still deciding which I like
      stemp <- x$states
      names(stemp) <- 1:length(stemp)
      print(stemp, quote=FALSE)
    }
  }
  else printCoefmat(tmp, digits=digits, P.values=TRUE, has.Pvalue=TRUE,
                    signif.stars = signif.stars, ...)

  logtest <- -2 * (x$loglik_init[1] - x$loglik[1])
  if (is.null(x$df)) df <- sum(!is.na(coef))
  else  df <- round(sum(x$df),2)
  cat("\n")
  cat("Likelihood ratio test=", format(round(logtest, 2)), "  on ",
      df, " df,", " p=",
      format.pval(pchisq(logtest, df, lower.tail=FALSE), digits=digits),
      "\n",  sep="")
  omit <- x$na.action
  cat("n=", x$n)
  if (!is.null(x$nevent)) cat(", number of events=", x$nevent, "\n")
  else cat("\n")
  if (length(omit))
    cat("\   (", naprint(omit), ")\n", sep="")
  invisible(x)

  if(x$nbootstraps > 1){
    cat("\n--- Other results with bootstrap with summary() ---")
  }
}


#' Summary method for coxphGPU object
#'
#' Use `summary()` method to see confidence interval for covariates with two
#' process : normal distribution and bootstrap (if `bootstrap > 1`).
#' @param object a coxphGPU object
#' @param conf.int level for computation of the confidence intervals.
#' @param scale vector of scale factors for the coefficients, defaults to 1. The
#'   printed coefficients, se, and confidence intervals will be associated with
#'   one scale unit.
#' @param ... additional argument(s) for methods.
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
summary.coxphGPU <- function(object, ..., conf.int = 0.95, scale = 1){

  cox<-object

  cov_names <- colnames(cox$coefficients)
  beta <- matrix(cox$coefficients[1,] * scale)
  if (is.null(cox$coefficients)) {   # Null model
    return(object)  #The summary method is the same as print in this case
  }
  nabeta <- !(is.na(beta))          #non-missing coefs
  beta2 <- beta[nabeta]
  if(is.null(beta) | is.null(cox$var))
    stop("Input is not valid")
  se <- matrix(sqrt(diag(cox$var[[1]])) * scale)
  if (!is.null(cox$naive.var)) nse <- sqrt(diag(cox$naive.var))

  rval<-list(call=cox$call,
             # fail=cox$fail,
             # na.action=cox$na.action,
             n=cox$n,
             loglik=cox$loglik,
             nbootstraps = cox$nbootstraps,
             conf.int_level = conf.int)
  if (!is.null(cox$nevent)) rval$nevent <- cox$nevent

  if (is.null(cox$naive.var)) {
    tmp <- cbind(beta, exp(beta), se, beta/se,
                 pchisq((beta/ se)^2, 1, lower.tail=FALSE))
    dimnames(tmp) <- list(cov_names, c("coef",
                                       "exp(coef)",
                                       "se(coef)",
                                       "z",
                                       "Pr(>|z|)"))
  }else {
    tmp <- cbind(beta, exp(beta), nse, se, beta/se,
                 pchisq((beta/ se)^2, 1, lower.tail=FALSE))
    dimnames(tmp) <- list(cov_names, c("coef",
                                       "exp(coef)",
                                       "se(coef)",
                                       "robust se",
                                       "z",
                                       "Pr(>|z|)"))
  }
  rval$coefficients <- tmp

  if (conf.int) {
    z <- qnorm((1 + conf.int)/2, 0, 1)
    tmp <- cbind(exp(beta),
                 exp(-beta),
                 exp(beta - z * se),
                 exp(beta + z * se))
    dimnames(tmp) <- list(cov_names,c("exp(coef)",
                                      "exp(-coef)",
                                      paste("lower .", round(100 * conf.int, 2), sep = ""),
                                      paste("upper .", round(100 * conf.int, 2), sep = "")))
    rval$conf.int <- tmp
  }

  if(object$nbootstraps > 1){

    probs = c((1-conf.int)/2,1-(1-conf.int)/2)

    # confidence Interval for coefficients (default 95%)
    rval$conf.int_bootstrap = apply(object$coef_bootstrap, 2, stats::quantile, p = probs)

    # # confidence Interval for loglik (default 95%)
    # rval$loglik_CI = stats::quantile(object$loglik, p = probs)
  }

  df <- length(beta2)

  logtest <- -2 * (cox$loglik_init[1] - cox$loglik[1])

  rval$logtest <- c(test=logtest,
                    df=df,
                    pvalue= pchisq(logtest, df, lower.tail=FALSE))
  rval$sctest <- c(test=cox$score[1],
                   df=df,
                   pvalue= pchisq(cox$score[1], df, lower.tail=FALSE))
  rval$rsq<-c(rsq=1-exp(-logtest/cox$n),
              maxrsq=1-exp(2*cox$loglik_init[1]/cox$n))
  rval$waldtest<-c(test=as.vector(round(cox$wald.test[1], 2)),
                   df=df,
                   pvalue= pchisq(as.vector(cox$wald.test[1]), df,
                                  lower.tail=FALSE))

  if (!is.null(cox$rscore))
    rval$robscore<-c(test=cox$rscore[1],
                     df=df,
                     pvalue= pchisq(cox$rscore[1], df, lower.tail=FALSE))
  rval$used.robust<-!is.null(cox$naive.var)

  if (!is.null(cox$concordance)) {
    # throw away the extra info, in the name of backwards compatability
    rval$concordance <- cox$concordance[[1]][6:7]
    names(rval$concordance) <- c("C", "se(C)")
  }
  if (inherits(cox, "coxphms")) {
    rval$cmap <- cox$cmap
    rval$states <- cox$states
  }

  class(rval) = "summary.coxphGPU"
  return(rval)
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
                                   digits = max(getOption('digits')-3, 3),
                                   signif.stars = getOption("show.signif.stars")){
  #x = rval
  if (!is.null(x$call)) {
    cat("Call:\n")
    dput(x$call)
    cat("\n")
  }
  if (!is.null(x$fail)) {
    cat(" Coxreg failed.", x$fail, "\n")
    return()
  }

  if(x$nbootstraps > 1){
    cat("Results without bootstrap :\n\n")
  }
  savedig <- options(digits = digits)
  on.exit(options(savedig))

  omit <- x$na.action
  cat("  n=", x$n)
  if (!is.null(x$nevent)) cat(", number of events=", x$nevent, "\n")
  else cat("\n")
  if (length(omit))
    cat("   (", naprint(omit), ")\n", sep="")

  if (nrow(x$coefficients)==0) {   # Null model
    cat ("   Null model\n")
    return()
  }

  if(!is.null(x$coefficients)) {
    cat("\n")
    printCoefmat(x$coefficients, digits=digits,
                 signif.stars=signif.stars, ...)
  }
  if(!is.null(x$conf.int)) {
    cat("\n")
    print(x$conf.int)
  }

  cat("\n")

  if (!is.null(x$concordance)) {
    cat("Concordance=", format(round(x$concordance[1],3)),
        " (se =", format(round(x$concordance[2], 3)),")\n")
  }
  cat("Rsquare=", format(round(x$rsq["rsq"],3)),
      "  (max possible=", format(round(x$rsq["maxrsq"],3)),
      ")\n" )

  pdig <- max(1, getOption("digits")-4)  # default it too high IMO
  cat("Likelihood ratio test= ", format(round(x$logtest["test"], 2)), "  on ",
      x$logtest["df"], " df,", "   p=",
      format.pval(x$logtest["pvalue"], digits=pdig),
      "\n", sep = "")

  cat("Wald test            = ", format(round(x$waldtest["test"], 2)), "  on ",
      x$waldtest["df"], " df,", "   p=",
      format.pval(x$waldtest["pvalue"], digits=pdig),
      "\n", sep = "")
  cat("Score (logrank) test = ", format(round(x$sctest["test"], 2)), "  on ",
      x$sctest["df"]," df,", "   p=",
      format.pval(x$sctest["pvalue"], digits=pdig), sep ="")
  if (is.null(x$robscore))
    cat("\n\n")
  else cat(",   Robust = ", format(round(x$robscore["test"], 2)),
           "  p=",
           format.pval(x$robscore["pvalue"], digits=pdig), "\n\n", sep="")

  if (x$used.robust)
    cat("  (Note: the likelihood ratio and score tests",
        "assume independence of\n     observations within a cluster,",
        "the Wald and robust score tests do not).\n")
  invisible()

  if(x$nbootstraps > 1){
    cat(" ---------------- \n")
    cat(paste0("Confidence interval with ", x$nbootstraps,
               " bootstraps for exp(coef), conf.level = ",
               x$conf.int_level," :\n"))
    print(signif(t(exp(x$conf.int_bootstrap))))
  }
}


#' Coef method for coxphGPU object
#'
#' @param object coxphGPU object.
#' @param ... additional argument(s) for methods.
#' @exportS3Method coef coxphGPU
#' @noRd
coef.coxphGPU <- function(object, ...){
  object$coefficients
}
