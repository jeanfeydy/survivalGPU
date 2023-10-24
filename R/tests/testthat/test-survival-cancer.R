
test_that("survival-cancer", {
  skip_if_no_python()
  skip_if_no_modules()

  library(survival)

  options(na.action=na.exclude) # preserve missings
  options(contrasts=c('contr.treatment', 'contr.poly')) #ensure constrast type
  aeq <- function(x, y, ...) all.equal(as.vector(x), as.vector(y), ...)


  # TODO: remove this when the re-implementation of coxph is over
  # We currently do not support Efron ties, only Breslow
  ties <- "breslow"  # "efron"


  #
  # Test out all of the routines on a more complex data set
  #
  temp <- survfit(Surv(time, status) ~ ph.ecog, lung)
  summary(temp, times=c(30*1:11, 365*1:3))
  print(temp[2:3])

  temp <- survfit(Surv(time, status)~1, lung, type='fleming',
                  conf.int=.9, conf.type='log-log', error='tsiatis')
  summary(temp, times=30 *1:5)

  temp <- survdiff(Surv(time, status) ~ inst, lung, rho=.5)
  print(temp, digits=6)

  # verify that the zph routine does the actual score test
  dtime <- lung$time[lung$status==2]
  lung2 <- survSplit(Surv(time, status) ~ ., lung, cut=dtime)

  ########################################################################

  cfit1 <-coxph(Surv(time, status) ~ ph.ecog + ph.karno + pat.karno + wt.loss,
                lung, ties = ties)

  # New status variable : coxphGPU take a variable event with 0-1 values (lung$status are 1-2 values)
  lung$status_0_1 <- lung$status - 1

  cfit1_gpu <-coxphGPU(Surv(time, status_0_1) ~ ph.ecog + ph.karno + pat.karno + wt.loss,
                       lung, ties = ties)


  # Checks coefs right process
  test_that("Coefs right process", {
    expect_equal(
      round(as.vector(coef(cfit1)),2),
      round(as.vector(coef(cfit1_gpu)),2)
    )
  })

  ########################################################################

  cfit2 <-coxph(Surv(tstart, time, status) ~ ph.ecog + ph.karno + pat.karno +
                  wt.loss, lung2, ties = ties)

  # Currently 'lung2' doesn't work with coxphGPU because different intervals

  # cfit2_gpu <-coxphGPU(Surv(tstart, time, status) ~ ph.ecog + ph.karno +
  #                        pat.karno + wt.loss, lung2, ties = ties)

  # cfit2_gpu <-coxphGPU(Surv(tstart, time, status) ~ ph.ecog + ph.karno +
  #                        pat.karno + wt.loss, head(lung2, 600), ties = ties)
  #
  # cfit2_gpu <-coxphGPU(Surv(tstart, time, status) ~ ph.ecog + ph.karno +
  #                        pat.karno + wt.loss, head(na.omit(lung2), 320), ties = ties)
  # cfit2_gpu <-coxphGPU(Surv(tstart, time, status) ~ ph.ecog + ph.karno +
  #                        pat.karno + wt.loss, head(na.omit(lung2), 330), ties = ties)


  # # Checks coefs right process
  # test_that("Coefs counting process", {
  #   expect_equal(
  #     round(as.vector(coef(cfit2)),2),
  #     round(as.vector(coef(cfit2_gpu)),2)
  #   )
  # })

  ########################################################################

  # # Checks right - counting results
  # test_that("Loglik right - counting", {
  #   expect_equal(
  #     round(as.vector(cfit1$loglik),2),
  #     round(as.vector(c(cfit2_gpu$loglik_init,cfit2_gpu$loglik)),2)
  #   )
  # })
  #
  # test_that("Coefs right - counting", {
  #   expect_equal(
  #     round(as.vector(coef(cfit1)),2),
  #     round(as.vector(coef(cfit2_gpu)),2)
  #   )
  # })



  # no method cox.zph for coxphGPU objects
  # # the above verifies that the data set is correct
  zp1 <- cox.zph(cfit1, transform="log")
  # zp2 <- cox.zph(cfit2, transform="log")
  # # everything should match but the call
  # icall <- match("Call", names(zp1))
  # all.equal(unclass(zp2)[-icall], unclass(zp1)[-icall])

  # now compute score tests one variable at a time
  ncoef <- length(coef(cfit2))
  check <- double(ncoef)
  cname <- names(coef(cfit2))
  for (i in 1:ncoef) {
    temp <- log(lung2$time) * lung2[[cname[i]]]
    # score test for this new variable
    tfit <- coxph(Surv(tstart, time, status) ~ ph.ecog + ph.karno + pat.karno +
                    wt.loss + temp,
                  lung2, init=c(cfit2$coef, 0), iter = 0)

    check[i] <- tfit$score
  }
  aeq(check, zp1$table[1:ncoef,1]) # skip the 'global' test


  # check_survivalGPU <- double(ncoef)
  # for (i in 1:ncoef) {
  #   lung2$temp <- log(lung2$time) * lung2[[cname[i]]]
  #   # score test for this new variable
  #   tfit <- coxphGPU(Surv(tstart, time, status) ~ ph.ecog + ph.karno + pat.karno +
  #                      wt.loss + temp,
  #                    lung2, init = c(cfit2$coef, 0), iter = 0, ties = ties)
  #
  #   check_survivalGPU[i] <- tfit$score
  # }
  # lung2$temp <- NULL
  #
  # test_that("Check score tests", {
  #   expect_equal(
  #     round(check,2),
  #     round(check_survivalGPU,2)
  #   )
  # })


  #
  # Tests of using "."
  #
  fit1 <- coxph(Surv(time, status) ~ . - meal.cal - wt.loss - inst - status_0_1, lung)
  fit2 <- update(fit1, .~. - ph.karno)
  fit3 <- coxph(Surv(time, status) ~ age + sex + ph.ecog + pat.karno, lung)
  all.equal(fit2, fit3)

  fit1b <- coxph(Surv(tstart, time, status) ~ . - meal.cal - wt.loss - inst, lung2)
  fit2b <- update(fit1b, .~. - ph.karno)
  fit3b <- coxph(Surv(tstart, time, status) ~ age + sex + ph.ecog + pat.karno, lung2)
  all.equal(fit2b, fit3b)


  test_that("Use of . in formula - coefs", {
    expect_equal(
      round(as.vector(coef(fit2b)),5),
      round(as.vector(coef(fit3b)),5)
    )
  })

  test_that("Use of . in formula - var", {
    expect_equal(
      round(as.vector(fit2b$var),5),
      round(as.vector(fit3b$var),5)
    )
  })


  # "." in formula with coxphGPU
  fit1_gpu <- coxphGPU(Surv(time, status_0_1) ~ . - meal.cal - wt.loss - inst - status, lung, ties = ties)
  fit3_gpu <- coxphGPU(Surv(time, status_0_1) ~ age + sex + ph.ecog + ph.karno + pat.karno, lung, ties = ties)

  test_that("Use of . in right coxphGPU formula - coefs", {
    expect_equal(
      round(as.vector(coef(fit1_gpu)),5),
      round(as.vector(coef(fit3_gpu)),5)
    )
  })

  # fit1b_gpu <- coxphGPU(Surv(tstart, time, status) ~ . - meal.cal - wt.loss - inst, lung2)
  # fit3b_gpu <- coxphGPU(Surv(tstart, time, status) ~ age + sex + ph.ecog + ph.karno + pat.karno, lung2)
  #
  # lung2_without_NA <- na.omit(lung2[,c("tstart", "time", "status", "age", "sex", "ph.ecog", "ph.karno", "pat.karno")])
  # fit1b_gpu <- coxphGPU(Surv(tstart, time, status) ~ . , lung2_without_NA)
  # fit3b_gpu <- coxphGPU(Surv(tstart, time, status) ~ age + sex + ph.ecog + ph.karno + pat.karno, lung2_without_NA)

})
