# Perform Weighted Cumulative Exposure analysis
wce <- WCE(
  data = drugdata, analysis = "Cox", nknots = 1, cutoff = 90, 
  constrained = "R", id = "Id", event = "Event",
  start = "Start", stop = "Stop", expos = "dose", 
  covariates = c("age", "sex")
)



## Not run:
# Confidence intervals for HR, as well as pointwise confidence bands
# for the estimated weight function can be obtained via bootstrap.

# Set the number of bootstrap resamples (set to 5 for demonstration purposes, should be higher)
B <- 5

# Obtain the list of ID for sampling
ID <- unique(drugdata$Id)

# Prepare vectors to extract estimated weight function and HR
# for the best-fitting model for each bootstrap resample
boot.WCE <- matrix(NA, ncol = 90, nrow = B)
boot.HR <- rep(NA, B)

# Sample IDs with replacement
for (i in 1:B) {
  ID.resamp <- sort(sample(ID, replace = TRUE))
  datab <- drugdata[drugdata$Id %in% ID.resamp,] # select observations but duplicated Id are ignored
  
  # Deal with duplicated Id and assign them new Id
  step <- 1
  repeat {
    # Select duplicated Id in ID.resamp
    ID.resamp <- ID.resamp[duplicated(ID.resamp) == TRUE]
    if (length(ID.resamp) == 0) break # stop when no more duplicated Id to deal with
    
    # Select observations but remaining duplicated Id are ignored
    subset.dup <- drugdata[drugdata$Id %in% ID.resamp,]
    
    # Assign new Id to duplicates
    subset.dup$Id <- subset.dup$Id + step * 10^ceiling(log10(max(drugdata$Id)))
    # 10^ceiling(log10(max(drugdata$Id))) is the power of 10
    # above the maximum Id from original data
    datab <- rbind(datab, subset.dup)
    step <- step + 1
  }
  
  # Fit the Weighted Cumulative Exposure model
  mod <- WCE(
    data = datab, analysis = "Cox", nknots = 1:3, cutoff = 90,
    constrained = "R", aic = FALSE, MatchedSet = NULL, id = "Id",
    event = "Event", start = "Start", stop = "Stop", expos = "dose",
    covariates = c("sex", "age")
  )
  
  # Return best WCE estimates and corresponding HR
  best <- which.min(mod$info.criterion)
  boot.WCE[i,] <- mod$WCEmat[best,]
  boot.HR[i] <- HR.WCE(mod, rep(1, 90), rep(0, 90))
}

# Summarize bootstrap results using percentile method
apply(boot.WCE, 2, quantile, p = c(0.05, 0.95))
quantile(boot.HR, p = c(0.05, 0.95))
## End(Not run)