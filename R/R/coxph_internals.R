## coxph internals functions from `survival` R package    ----------------------

# terms.inner

# Function to extract the arguments of a formula by distinguishing between what
# is before/after the "~"
terms.inner <- function(x) {
  if (inherits(x, "formula")) {
    if (length(x) == 3) {
      c(terms.inner(x[[2]]), terms.inner(x[[3]]))
    } else {
      terms.inner(x[[2]])
    }
  } else if (inherits(x, "call") &&
             (x[[1]] != as.name("$") && x[[1]] != as.name("["))) {
    if (x[[1]] == "+" || x[[1]] == "*" || x[[1]] == "-" || x[[1]] == ":") {
      # terms in a model equation, unary minus only has one argument
      if (length(x) == 3) {
        c(terms.inner(x[[2]]), terms.inner(x[[3]]))
      } else {
        terms.inner(x[[2]])
      }
    } else if (x[[1]] == as.name("Surv")) {
      unlist(lapply(x[-1], terms.inner))
    } else {
      terms.inner(x[[2]])
    }
  } else {
    (deparse(x))
  }
}


# drop.special

# This routine is loosely based in drop.terms.
#   In a terms structure, the factors attribute is a matrix with row and column
# names.  The predvars and dataClasses attributites, if present, index to the
# row names; as do values of the specials attribute.  The term.labels attribute
# aligns with the column names.
#  For most model formula the row and column names nicely align, but not always.
# [.terms, unfortunately, implicitly assumes that they do align.
#
#  Unlike drop.terms, do not remove offset terms in the process
drop.special <- function(termobj, i, addparen = FALSE) {
  # First step is to rebuild the formula using term.labels and reformulate
  # Map row name to the right column name
  ff <- attr(termobj, "factors")
  index <- match(rownames(ff)[i], colnames(ff))
  if (any(is.null(index))) stop("failure in drop.specials function")

  newterms <- attr(termobj, "term.labels")[-index]
  # the above ignores offsets, add them back in
  if (length(attr(termobj, "offset")) > 0) {
    newterms <- c(newterms, rownames(ff)[attr(termobj, "offset")])
  }

  rvar <- if (attr(termobj, "response") == 1) termobj[[2L]]

  # Adding () around each term is for a formula containing  + (sex=='male')
  #   It's a crude fix and causes the formula to look different
  if (addparen) {
    newformula <- reformulate(paste0("(", newterms, ")"),
                              response = rvar,
                              intercept = attr(termobj, "intercept"),
                              env = environment(termobj)
    )
  } else {
    newformula <- reformulate(newterms,
                              response = rvar,
                              intercept = attr(termobj, "intercept"),
                              env = environment(termobj)
    )
  }
  if (length(newformula) == 0L) newformula <- "1"

  # addition of an extra specials label causes no harm
  result <- terms(newformula, specials = names(attr(termobj, "specials")))

  # now add back the predvars and dataClasses attributes; which do contain
  # the response and offset.
  index2 <- seq.int(nrow(ff))[-i]
  if (!is.null(attr(termobj, "predvars"))) {
    attr(result, "predvars") <- attr(termobj, "predvars")[c(1, index2 + 1)]
  }
  if (!is.null(attr(termobj, "dataClasses"))) {
    attr(result, "dataClasses") <- attr(termobj, "dataClasses")[index2]
  }

  result
}


# Automatically generated from the noweb directory
parsecovar1 <- function(flist, statedata) {
  if (any(sapply(flist, function(x) !inherits(x, "formula")))) {
    stop("an element of the formula list is not a formula")
  }
  if (any(sapply(flist, length) != 3)) {
    stop("all formulas must have a left and right side")
  }

  # split the formulas into a right hand and left hand side
  lhs <- lapply(flist, function(x) x[-3]) # keep the ~
  rhs <- lapply(flist, function(x) x[[3]]) # don't keep the ~

  rhs <- parse_rightside(rhs)
  # deal with the left hand side of the formula
  # the next routine cuts at '+' signs
  pcut <- function(form) {
    if (length(form) == 3) {
      if (form[[1]] == "+") {
        c(pcut(form[[2]]), pcut(form[[3]]))
      } else if (form[[1]] == "~") {
        pcut(form[[2]])
      } else {
        list(form)
      }
    } else {
      list(form)
    }
  }
  lcut <- lapply(lhs, function(x) pcut(x[[2]]))
  env1 <- new.env(parent = parent.frame(2))
  env2 <- new.env(parent = env1)
  if (missing(statedata)) {
    assign("state", function(...) {
      list(
        stateid = "state",
        values = c(...)
      )
    }, env1)
    assign("state", list(stateid = "state"))
  } else {
    for (i in statedata) {
      assign(i, eval(list(stateid = i)), env2)
      tfun <- eval(parse(text = paste0(
        "function(...) list(stateid='",
        i, "', values=c(...))"
      )))
      assign(i, tfun, env1)
    }
  }
  lterm <- lapply(lcut, function(x) {
    lapply(x, function(z) {
      if (length(z) == 1) {
        temp <- eval(z, envir = env2)
        if (is.list(temp) && names(temp)[[1]] == "stateid") {
          temp
        } else {
          temp
        }
      } else if (length(z) == 3 && z[[1]] == ":") {
        list(left = eval(z[[2]], envir = env2), right = eval(z[[3]], envir = env2))
      } else {
        stop("invalid term: ", deparse(z))
      }
    })
  })
  list(rhs = rhs, lhs = lterm)
}
rightslash <- function(x) {
  if (!inherits(x, "call")) {
    return(x)
  } else {
    if (x[[1]] == as.name("/")) {
      return(list(x[[2]], x[[3]]))
    } else if (x[[1]] == as.name("+") || (x[[1]] == as.name("-") && length(x) == 3) ||
               x[[1]] == as.name("*") || x[[1]] == as.name(":") ||
               x[[1]] == as.name("%in%")) {
      temp <- rightslash(x[[3]])
      if (is.list(temp)) {
        x[[3]] <- temp[[1]]
        return(list(x, temp[[2]]))
      } else {
        temp <- rightslash(x[[2]])
        if (is.list(temp)) {
          x[[2]] <- temp[[2]]
          return(list(temp[[1]], x))
        } else {
          return(x)
        }
      }
    } else {
      return(x)
    }
  }
}
parse_rightside <- function(rhs) {
  parts <- lapply(rhs, rightslash)
  new <- lapply(parts, function(opt) {
    tform <- ~x # a skeleton, "x" will be replaced
    if (!is.list(opt)) { # no options for this line
      tform[[2]] <- opt
      list(
        formula = tform, ival = NULL, common = FALSE,
        shared = FALSE
      )
    } else {
      # treat the option list as though it were a formula
      temp <- ~x
      temp[[2]] <- opt[[2]]
      optterms <- terms(temp)
      ff <- rownames(attr(optterms, "factors"))
      index <- match(ff, c("common", "shared", "init"))
      if (any(is.na(index))) {
        stop(
          "option not recognized in a covariates formula: ",
          paste(ff[is.na(index)], collapse = ", ")
        )
      }
      common <- any(index == 1)
      shared <- any(index == 2)
      if (any(index == 3)) {
        optatt <- attributes(optterms)
        j <- optatt$variables[1 + which(index == 3)]
        j[[1]] <- as.name("list")
        ival <- unlist(eval(j, parent.frame()))
      } else {
        ival <- NULL
      }
      tform[[2]] <- opt[[1]]
      list(formula = tform, ival = ival, common = common, shared = shared)
    }
  })
  new
}
termmatch <- function(f1, f2) {
  # look for f1 in f2, each the factors attribute of a terms object
  if (length(f1) == 0) {
    return(NULL)
  } # a formula with only ~1
  irow <- match(rownames(f1), rownames(f2))
  if (any(is.na(irow))) stop("termmatch failure 1")
  hashfun <- function(j) sum(ifelse(j == 0, 0, 2^(seq(along.with = j))))
  hash1 <- apply(f1, 2, hashfun)
  hash2 <- apply(f2[irow, , drop = FALSE], 2, hashfun)
  index <- match(hash1, hash2)
  if (any(is.na(index))) stop("termmatch failure 2")
  index
}

parsecovar2 <- function(covar1, statedata, dformula, Terms, transitions, states) {
  if (is.null(statedata)) {
    statedata <- data.frame(state = states, stringsAsFactors = FALSE)
  } else {
    if (is.null(statedata$state)) {
      stop("the statedata data set must contain a variable 'state'")
    }
    indx1 <- match(states, statedata$state, nomatch = 0)
    if (any(indx1 == 0)) {
      stop(
        "statedata does not contain all the possible states: ",
        states[indx1 == 0]
      )
    }
    statedata <- statedata[indx1, ] # put it in order
  }

  # Statedata might have rows for states that are not in the data set,
  #  for instance if the coxph call had used a subset argument.  Any of
  #  those were eliminated above.
  # Likewise, the formula list might have rules for transitions that are
  #  not present.  Don't worry about it at this stage.
  allterm <- attr(Terms, "factors")
  nterm <- ncol(allterm)

  # create a map for every transition, even ones that are not used.
  # at the end we will thin it out
  # It has an extra first row for intercept (baseline)
  # Fill it in with the default formula
  nstate <- length(states)
  tmap <- array(0L, dim = c(nterm + 1, nstate, nstate))
  dmap <- array(seq_len(length(tmap)), dim = c(nterm + 1, nstate, nstate)) # unique values
  dterm <- termmatch(attr(terms(dformula), "factors"), allterm)
  dterm <- c(1L, 1L + dterm) # add intercept
  tmap[dterm, , ] <- dmap[dterm, , ]
  inits <- NULL

  if (!is.null(covar1)) {
    for (i in 1:length(covar1$rhs)) {
      rhs <- covar1$rhs[[i]]
      lhs <- covar1$lhs[[i]] # one rhs and one lhs per formula

      state1 <- state2 <- NULL
      for (x in lhs) {
        # x is one term
        if (!is.list(x) || is.null(x$left)) stop("term found without a ':' ", x)
        # left of the colon
        if (!is.list(x$left) && length(x$left) == 1 && x$left == 0) {
          temp1 <- 1:nrow(statedata)
        } else if (is.numeric(x$left)) {
          temp1 <- as.integer(x$left)
          if (any(temp1 != x$left)) stop("non-integer state number")
          if (any(temp1 < 1 | temp1 > nstate)) {
            stop("numeric state is out of range")
          }
        } else if (is.list(x$left) && names(x$left)[1] == "stateid") {
          if (is.null(x$left$value)) {
            stop("state variable with no list of values: ", x$left$stateid)
          } else {
            if (any(k = is.na(match(x$left$stateid, names(statedata))))) {
              stop(x$left$stateid[k], ": state variable not found")
            }
            zz <- statedata[[x$left$stateid]]
            if (any(k = is.na(match(x$left$value, zz)))) {
              stop(x$left$value[k], ": state value not found")
            }
            temp1 <- which(zz %in% x$left$value)
          }
        } else {
          k <- match(x$left, statedata$state)
          if (any(is.na(k))) stop(x$left[is.na(k)], ": state not found")
          temp1 <- which(statedata$state %in% x$left)
        }

        # right of colon
        if (!is.list(x$right) && length(x$right) == 1 && x$right == 0) {
          temp2 <- 1:nrow(statedata)
        } else if (is.numeric(x$right)) {
          temp2 <- as.integer(x$right)
          if (any(temp2 != x$right)) stop("non-integer state number")
          if (any(temp2 < 1 | temp2 > nstate)) {
            stop("numeric state is out of range")
          }
        } else if (is.list(x$right) && names(x$right)[1] == "stateid") {
          if (is.null(x$right$value)) {
            stop("state variable with no list of values: ", x$right$stateid)
          } else {
            if (any(k = is.na(match(x$right$stateid, names(statedata))))) {
              stop(x$right$stateid[k], ": state variable not found")
            }
            zz <- statedata[[x$right$stateid]]
            if (any(k = is.na(match(x$right$value, zz)))) {
              stop(x$right$value[k], ": state value not found")
            }
            temp2 <- which(zz %in% x$right$value)
          }
        } else {
          k <- match(x$right, statedata$state)
          if (any(is.na(k))) stop(x$right[k], ": state not found")
          temp2 <- which(statedata$state %in% x$right)
        }


        state1 <- c(state1, rep(temp1, length(temp2)))
        state2 <- c(state2, rep(temp2, each = length(temp1)))
      }
      npair <- length(state1) # number of state:state pairs for this line

      # update tmap for this set of transitions
      # first, what variables are mentioned, and check for errors
      rterm <- terms(rhs$formula)
      rindex <- 1L + termmatch(attr(rterm, "factors"), allterm)

      # the update.formula function is good at identifying changes
      # formulas that start with  "- x" have to be pasted on carefully
      temp <- substring(deparse(rhs$formula, width.cutoff = 500), 2)
      if (substring(temp, 1, 1) == "-") {
        dummy <- formula(paste("~ .", temp))
      } else {
        dummy <- formula(paste("~. +", temp))
      }

      rindex1 <- termmatch(attr(terms(dformula), "factors"), allterm)
      rindex2 <- termmatch(
        attr(terms(update(dformula, dummy)), "factors"),
        allterm
      )
      dropped <- 1L + rindex1[is.na(match(rindex1, rindex2))] # remember the intercept
      if (length(dropped) > 0) {
        for (k in 1:npair) tmap[dropped, state1[k], state2[k]] <- 0
      }

      # grab initial values
      if (length(rhs$ival)) {
        inits <- c(inits, list(
          term = rindex, state1 = state1,
          state2 = state2, init = rhs$ival
        ))
      }

      # adding -1 to the front is a trick, to check if there is a "+1" term
      dummy <- ~ -1 + x
      dummy[[2]][[3]] <- rhs$formula
      if (attr(terms(dummy), "intercept") == 1) rindex <- c(1L, rindex)

      # an update of "- sex" won't generate anything to add
      # dmap is simply an indexed set of unique values to pull from, so that
      #  no number is used twice
      if (length(rindex) > 0) { # rindex = things to add
        if (rhs$common) {
          j <- dmap[rindex, state1[1], state2[1]]
          for (k in 1:npair) tmap[rindex, state1[k], state2[k]] <- j
        } else {
          for (k in 1:npair) {
            tmap[rindex, state1[k], state2[k]] <- dmap[rindex, state1[k], state2[k]]
          }
        }
      }

      # Deal with the shared argument, using - for a separate coef
      if (rhs$shared && npair > 1) {
        j <- dmap[1, state1[1], state2[1]]
        for (k in 2:npair) {
          tmap[1, state1[k], state2[k]] <- -j
        }
      }
    }
  }
  i <- match("(censored)", colnames(transitions), nomatch = 0)
  if (i == 0) {
    t2 <- transitions
  } else {
    t2 <- transitions[, -i, drop = FALSE]
  } # transitions to 'censor' don't count
  indx1 <- match(rownames(t2), states)
  indx2 <- match(colnames(t2), states)
  tmap2 <- matrix(0L, nrow = 1 + nterm, ncol = sum(t2 > 0))

  trow <- row(t2)[t2 > 0]
  tcol <- col(t2)[t2 > 0]
  for (i in 1:nrow(tmap2)) {
    for (j in 1:ncol(tmap2)) {
      tmap2[i, j] <- tmap[i, indx1[trow[j]], indx2[tcol[j]]]
    }
  }

  # Remember which hazards had ph
  # tmap2[1,] is the 'intercept' row
  # If the hazard for colum 6 is proportional to the hazard for column 2,
  # the tmap2[1,2] = tmap[1,6], and phbaseline[6] =2
  temp <- tmap2[1, ]
  indx <- which(temp > 0)
  tmap2[1, ] <- indx[match(abs(temp), temp[indx])]
  phbaseline <- ifelse(temp < 0, tmap2[1, ], 0) # remembers column numbers
  tmap2[1, ] <- match(tmap2[1, ], unique(tmap2[1, ])) # unique strata 1,2, ...

  if (nrow(tmap2) > 1) {
    tmap2[-1, ] <- match(tmap2[-1, ], unique(c(0L, tmap2[-1, ]))) - 1L
  }

  dimnames(tmap2) <- list(
    c("(Baseline)", colnames(allterm)),
    paste(indx1[trow], indx2[tcol], sep = ":")
  )
  # mapid gives the from,to for each realized state
  list(
    tmap = tmap2, inits = inits, mapid = cbind(from = indx1[trow], to = indx2[tcol]),
    phbaseline = phbaseline
  )
}
parsecovar3 <- function(tmap, Xcol, Xassign, phbaseline = NULL) {
  # sometime X will have an intercept, sometimes not; cmap never does
  hasintercept <- (Xassign[1] == 0)
  ph.coef <- (phbaseline != 0) # any proportional baselines?
  ph.rows <- length(unique(phbaseline[ph.coef])) # extra rows to add to cmap
  cmap <- matrix(0L, length(Xcol) + ph.rows - hasintercept, ncol(tmap))
  uterm <- unique(Xassign[Xassign != 0L]) # terms that will have coefficients

  xcount <- table(factor(Xassign, levels = 1:max(Xassign)))
  mult <- 1L + max(xcount) # temporary scaling

  ii <- 0
  for (i in uterm) {
    k <- seq_len(xcount[i])
    for (j in 1:ncol(tmap)) {
      cmap[ii + k, j] <- if (tmap[i + 1, j] == 0) 0L else tmap[i + 1, j] * mult + k
    }
    ii <- ii + max(k)
  }

  if (ph.rows > 0) {
    temp <- phbaseline[ph.coef] # where each points
    for (i in unique(temp)) {
      # for each baseline that forms a reference
      j <- which(phbaseline == i) # the others that are proportional to it
      k <- seq_len(length(j))
      ii <- ii + 1 # row of cmat for this baseline
      cmap[ii, j] <- max(cmap) + k # fill in elements
    }
    newname <- paste0("ph(", colnames(tmap)[unique(temp)], ")")
  } else {
    newname <- NULL
  }

  # renumber coefs as 1, 2, 3, ...
  cmap[, ] <- match(cmap, sort(unique(c(0L, cmap)))) - 1L

  colnames(cmap) <- colnames(tmap)
  if (hasintercept) {
    rownames(cmap) <- c(Xcol[-1], newname)
  } else {
    rownames(cmap) <- c(Xcol, newname)
  }

  cmap
}


# This routine creates a stacked data set.
# The key input is the cmap matrix, which has one row for each column
#  of X and one column per transition.  It may have extra rows if there are
#  proportional baseline hazards
# The first row of smat contains state to strata information.
# Input data is X, Y, strata, and initial state (integer).
#
# For each transition the expanded data has a set of rows, all those whose
#  initial state makes them eligible for the transition.
# Strata is most often null; it encodes a users strata() addition(s). Such terms
#  occur less often in multistate models (in my experience so far.)
#
stacker <- function(cmap, smap, istate, X, Y, strata, states, dropzero = TRUE) {
  from.state <- as.numeric(sub(":.*$", "", colnames(cmap)))
  to.state <- as.numeric(sub("^.*:", "", colnames(cmap)))

  # just in case cmap has columns I don't need (I don't think this can
  #  happen
  check <- match(from.state, istate, nomatch = 0)
  if (any(check == 0)) {
    # I think that this is impossible
    warning("extra column in cmap, this is a bug") # debugging line
    # browser()
    cmap <- cmap[, check > 0]
    from.state <- from.state[check > 0]
    to.state <- to.state[check > 0]
  }

  # Don't create X and Y matrices for transitions with no covariates, for
  #  coxph calls.  But I need them for survfit.coxph.
  zerocol <- apply(cmap == 0, 2, all)
  if (dropzero && any(zerocol)) {
    cmap <- cmap[, !zerocol, drop = FALSE]
    smap <- smap[, !zerocol, drop = FALSE]
    smap[, ] <- match(smap, sort(unique(c(smap)))) # relabel as 1, 2,...
    from.state <- from.state[!zerocol]
    to.state <- to.state[!zerocol]
  }

  endpoint <- c(0, match(attr(Y, "states"), states))
  endpoint <- endpoint[1 + Y[, ncol(Y)]] # endpoint of each row, 0=censor

  # Jan 2021: changed from looping once per strata to once per transition.
  #  Essentially, a block of data for each unique column of cmap.  If two
  #  of those columns have the same starting state, it makes me nervous
  #  (statistically), but forge onward and sort the issues out in the
  #  fits.
  # Pass 1 to find the total data set size
  nblock <- ncol(cmap)
  n.perblock <- integer(nblock)
  for (i in 1:nblock) {
    n.perblock[i] <- sum(istate == from.state[i]) # can participate
  }

  # The constructed X matrix has a block of rows for each column of cmap
  n2 <- sum(n.perblock) # number of rows in new data
  newX <- matrix(0, nrow = n2, ncol = max(cmap))
  k <- 0
  rindex <- integer(n2) # original row for each new row of data
  newstat <- integer(n2) # new status
  Xcols <- ncol(X) # number of columns in X
  for (i in 1:nblock) {
    subject <- which(istate == from.state[i]) # data rows in strata
    nr <- k + seq(along.with = subject) # rows in the newX for this strata
    rindex[nr] <- subject
    nc <- cmap[, i]
    if (any(nc > Xcols)) { # constructed PH variables
      newX[nr, nc[nc > Xcols]] <- 1
      nc <- nc[1:Xcols]
    }
    newX[nr, nc[nc > 0]] <- X[subject, which(nc > 0)] # row of cmap= col of X

    event.that.counts <- (endpoint[subject] == to.state[i])
    newstat[nr] <- ifelse(event.that.counts, 1L, 0L)
    k <- max(nr)
  }

  # which transition each row of newX represents
  transition <- rep(1:nblock, n.perblock)

  # remove any rows where X is missing
  #  these arise when a variable is used only for some transitions
  #  the row of data needs to be tossed for the given ones, but will be
  #    okay for other transitions
  keep <- !apply(is.na(newX), 1, any)
  if (!all(keep)) {
    newX <- newX[keep, , drop = FALSE]
    rindex <- rindex[keep]
    newstat <- newstat[keep]
    transition <- transition[keep]
  }

  if (ncol(Y) == 2) {
    newY <- Surv(Y[rindex, 1], newstat)
  } else {
    newY <- Surv(Y[rindex, 1], Y[rindex, 2], newstat)
  }

  # newstrat will be an integer vector.
  newstrat <- smap[1, transition] # start with strata induced by multi-state
  # then add any strata from the users strata() terms
  if (is.matrix(strata)) {
    # this is the most complex case.
    maxstrat <- apply(strata, 2, max) # max in each colum of strata
    mult <- cumprod(c(1, maxstrat))
    temp <- max(mult) * newstrat
    for (i in 1:ncol(strata)) {
      k <- smap[i + 1, transition]
      temp <- temp + ifelse(k == 0, 0L, strata[i, rindex] * temp[i] - 1L)
    }
    newstrat <- match(temp, sort(unique(temp)))
  } else if (length(strata) > 0) {
    # strata will be an integer vector with elements of 1, 2 etc
    mult <- max(strata)
    temp <- mult * newstrat + ifelse(smap[2, transition] == 0, 0L, strata[rindex] - 1L)
    newstrat <- match(temp, sort(unique(temp)))
  }

  # give variable names to the new data  (some names get used more than once)
  #    vname <- rep("", ncol(newX))
  #    vname[cmap[cmap>0]] <- colnames(X)[row(cmap)[cmap>0]]
  first <- match(sort(unique(cmap[cmap > 0])), cmap) # first instance of each value
  vname <- rownames(cmap)[row(cmap)[first]]
  colnames(newX) <- vname
  list(
    X = newX, Y = newY, strata = as.integer(newstrat),
    transition = as.integer(transition), rindex = rindex
  )
}


# Routine to turn a Surv2 type dataset into a Surv type of data set
#  The task is pretty simple
#     1. An id with 10 rows will have 9 in the new data set, 1-9 contribute
#        covariates, and 2-10 the endpoints for those rows.
#     2. Missing covariates are filled in using last-value-carried forward.
#     3. A response, id, current state, and new data set are returned.
# If check=FALSE, it is being called by survcheck.  In that case don't fail
#   if there is a duplicate time, but rather let survcheck worry about it.
# The repeat attribute of the Surv2 object determines whether events can
#   "stutter", i.e., two of the same type that are adjacent, or only have
#    missing between them, count as a second event.
#
surv2data <- function(mf, check = FALSE) {
  Terms <- terms(mf)
  y <- model.response(mf)
  if (!inherits(y, "Surv2")) stop("response must be a Surv2 object")
  n <- nrow(y)
  states <- attr(y, "states")
  repeated <- attr(y, "repeated")

  id <- model.extract(mf, "id")
  if (length(id) != n) stop("id statement is required")

  # relax this some later day (or not?)
  if (any(is.na(id)) || any(is.na(y[, 1]))) {
    stop("id and time cannot be missing")
  }

  isort <- order(id, y[, 1])
  id2 <- id[isort]
  y2 <- y[isort, ]
  first <- !duplicated(id2)
  last <- !duplicated(id2, fromLast = TRUE)

  status <- y2[!first, 2]
  y3 <- cbind(y2[!last, 1], y2[!first, 1], ifelse(is.na(status), 0, status))
  if (!is.null(states)) {
    if (all(y3[, 1] == 0)) {
      y3 <- y3[, 2:3]
      attr(y3, "type") <- "mright"
    } else {
      attr(y3, "type") <- "mcounting"
    }
    attr(y3, "states") <- states
  } else {
    if (all(y3[, 1] == 0)) {
      y3 <- y3[, 2:3]
      attr(y3, "type") <- "right"
    } else {
      attr(y3, "type") <- "counting"
    }
  }
  class(y3) <- "Surv"

  if (!check && ncol(y3) == 3 && any(y3[, 1] == y3[, 2])) {
    stop("duplicated time values for a single id")
  }

  # We no longer need the last row of the data
  # tmerge3 expects the data in time within id order
  mf2 <- mf[isort, ][!last, ]
  id3 <- id2[!last]
  # Use LVCF on all the variables in the data set, other than
  #  the response and id.  The response is always first and
  #  the id and cluster will be at the end
  fixup <- !(names(mf) %in% c("(id)", "(cluster)"))
  if (is.factor(id3)) {
    idi <- as.integer(id3)
  } else {
    idi <- match(id3, unique(id3))
  }
  for (i in (which(fixup))[-1]) {
    miss <- is.na(mf2[[i]])
    if (any(miss)) {
      k <- .Call("tmerge3", idi, miss)
      update <- (miss & k > 0) # some values will be updated
      if (any(update)) { # some successful replacements
        if (is.matrix(mf2[[i]])) {
          mf2[[i]][update, ] <- mf2[[i]][k[update], ]
        } else {
          mf2[[i]][update] <- (mf2[[i]])[k[update]]
        }
      }
    }
  }

  # create the current state vector
  #  this is simply the non-lagged y -- except, we need to lag the state
  #  over missing values.  Do that with survcheck.
  istate <- y2[!last, 2]
  first <- !duplicated(id3)
  n <- nrow(y3)
  temp <- istate[first]
  if (all(is.na(temp) | (temp == 0))) {
    # special case -- no initial state for anyone
    temp <- survcheck(y3 ~ 1, id = id3)
  } else if (any(is.na(temp) | (temp == 0))) {
    stop("everyone or no one should have an initial state")
  } else {
    # survcheck does not like missing istate values.  Only the initial
    #  one for each subject needs to be correct though
    itemp <- istate[match(id3, id3)]
    if (is.null(states)) {
      temp <- survcheck(y3 ~ 1, id = id3, istate = itemp)
    } else {
      temp <- survcheck(y3 ~ 1,
                        id = id3,
                        istate = factor(itemp, 1:length(states), states)
      )
    }
  }

  # Treat any repeated events as censors
  if (!repeated) {
    ny <- ncol(y3)
    if (is.null(states)) {
      stutter <- y3[, ny] == temp$istate
    } else {
      itemp <- c(0L, match(attr(y3, "states"), temp$states, nomatch = 0L))
      stutter <- (itemp[1L + y3[, ny]] == as.integer(temp$istate))
    }
    if (any(stutter)) y3[stutter, ny] <- 0L
  }

  if (check) {
    list(
      y = y3, id = id3, istate = temp$istate, mf = mf2, isort = isort,
      last = last
    )
  } else { # put the data back into the original order
    jj <- order(isort[!last]) # this line is not obvious, but it works!
    list(y = y3[jj, ], id = id3[jj], istate = temp$istate[jj], mf = mf2[jj, ])
  }
}


# The multi-state routines need to check their input data
#  y = survival object
#  id = subject identifier
#  istate = starting state for each row, this can be missing.
# The routine creates a proper current-state vector accounting for censoring
#  (which should be called cstate for 'current state', but istate is retained)
#  If a subject started in state A for instance, and had observations
#  of (0, 10, B), (10, 15, censor), (15, 20, C) the current state is
#  (A, B, B).  It checks that against the input 'istate' if it is present
#  to generate checks.
# Multiple other checks as well
#
survcheck2 <- function(y, id, istate = NULL, istate0 = "(s0)") {
  n <- length(id)
  ny <- ncol(y)
  # the next few line are a debug for my code; survcheck2 is not visible
  #  to users so only survival can call it directly
  if (!is.Surv(y) || is.null(attr(y, "states")) ||
      any(y[, ncol(y)] > length(attr(y, "states")))) {
    stop("survcheck2 called with an invalid y argument")
  }
  to.names <- c(attr(y, "states"), "(censored)")

  if (length(istate) == 0) {
    inull <- TRUE
    cstate <- factor(rep(istate0, n))
  } else {
    if (length(istate) != n) stop("wrong length for istate")
    if (is.factor(istate)) {
      cstate <- istate[, drop = TRUE]
    } # drop unused levels
    else {
      cstate <- as.factor(istate)
    }
    inull <- FALSE
  }

  ystate <- attr(y, "states")
  # The vector of all state names is put in a nice printing order:
  #   initial states that are not destination states, then
  #   the destination states.  This keeps destinations in the order the
  #   user chose, while still putting initial states first.
  index <- match(levels(cstate), ystate, nomatch = 0)
  states <- c(levels(cstate)[index == 0], ystate)
  cstate2 <- factor(cstate, states)

  # Calculate the counts per id for each state, e.g., 10 subjects had
  #  3 visits to state 2, etc.
  # Count the censors, so that each subject gets a row in the table,
  #  but then toss that column
  tab1 <- table(id, factor(y[, ncol(y)], 0:length(ystate)))[, -1, drop = FALSE]
  tab1 <- cbind(tab1, rowSums(tab1))
  tab1.levels <- sort(unique(c(tab1))) # unique counts
  if (length(tab1.levels) == 1) {
    # In this special case the table command does not give a matrix
    #  A data set with no events falls here, for instance
    events <- matrix(tab1.levels, nrow = 1, ncol = (1 + length(ystate)))
  } else {
    events <- apply(tab1, 2, function(x) table(factor(x, tab1.levels)))
  }
  dimnames(events) <- list(
    "count" = tab1.levels,
    "state" = c(ystate, "(any)")
  )
  # remove columns with no visits
  novisit <- colSums(events[-1, , drop = FALSE]) == 0
  if (any(novisit)) events <- events[, !novisit]

  # Use a C routine to create 3 variables: a: an index of whether this is
  #   the first (1) or last(2) observation for a subject, 3=both, 0=neither,
  #  b. current state, and
  #  c. sign of (start of this interval - end of prior one)
  # start by making stat2 = status re-indexed to the full set of states
  ny <- ncol(y)
  sindx <- match(ystate, states)
  stat2 <- ifelse(y[, ny] == 0, 0L, sindx[pmax(1L, y[, ny])])
  id2 <- match(id, unique(id)) # we need unique integers
  if (ncol(y) == 2) {
    index <- order(id, y[, 1])
    check <- .Call(
      "multicheck", rep(0., n), y[, 1], stat2, id2,
      as.integer(cstate2), index - 1L
    )
  } else {
    index <- order(id, y[, 2], y[, 1])
    check <- .Call(
      "multicheck", y[, 1], y[, 2], stat2, id2,
      as.integer(cstate2), index - 1L
    )
  }

  if (inull && ny > 2) {
    # if there was no istate entered in, use the constructed one from
    # the check routine
    # if ny=2 then every row starts at time 0
    cstate2 <- factor(check$cstate, seq(along.with = states), states)
  }

  # create the transtions table
  # if someone has an intermediate visit, i.e., (0,10, 0)(10,20,1), don't
  #  report the false 'censoring' in the transitions table
  # make it compact by removing any cols that are all 0, and rows of
  #  states that never occur (sometimes the starting state is a factor
  #  with unused levels)
  keep <- (stat2 != 0 | check$dupid > 1) # not censored or last obs of this id
  transitions <- table(
    from = cstate2[keep],
    to = factor(
      stat2[keep], c(seq(along.with = states), 0),
      c(states, "(censored)")
    ),
    useNA = "ifany"
  )
  nr <- nrow(transitions)
  never <- (rowSums(transitions) + colSums(transitions[, 1:nr])) == 0
  transitions <- transitions[!never, colSums(transitions) > 0, drop = FALSE]

  # now continue with error checks
  # A censoring hole in the middle, such as happens with survSplit,
  #  uses "last state carried forward" in Cmultistate, which also
  #  sets the "gap" to 0 for the first obs of a subject
  mismatch <- (as.numeric(cstate2) != check$cstate)

  # gap = 0   (0, 10], (10, 15]
  # gap = 1   (0, 10], (12, 15]  # a hole in the time
  # gap = -1  (0, 10], (9, 15]   # overlapping times
  flag <- c(
    overlap = sum(check$gap < 0),
    gap = sum(check$gap > 0 & !mismatch),
    jump = sum(check$gap > 0 & mismatch),
    teleport = sum(check$gap == 0 & mismatch & check$dupid %% 2 == 0)
  )

  rval <- list(
    states = states, transitions = transitions,
    events = t(events), flag = flag,
    istate = factor(check$cstate, seq(along.with = states), states)
  )

  # add error details, if necessary
  if (flag["overlap"] > 0) {
    j <- which(check$gap < 0)
    rval$overlap <- list(row = j, id = unique(id[j]))
  }
  if (flag["gap"] > 0) {
    j <- which(check$gap > 0 & !mismatch)
    rval$gap <- list(row = j, id = unique(id[j]))
  }
  if (flag["jump"] > 0) {
    j <- which(check$gap > 0 & mismatch)
    rval$jump <- list(row = j, id = unique(id[j]))
  }
  if (flag["teleport"] > 0) {
    j <- which(check$gap == 0 & mismatch)
    rval$teleport <- list(row = j, id = unique(id[j]))
  }

  rval
}
