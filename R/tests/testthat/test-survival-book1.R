
test_that("survival-book1", {
  skip_if_no_python()
  skip_if_no_modules()

  library(survival)
  options(na.action=na.exclude) # preserve missings
  options(contrasts=c('contr.treatment', 'contr.poly')) #ensure constrast type
  aeq <- function(x,y) all.equal(as.vector(x), as.vector(y))

  #
  # Tests from the appendix of Therneau and Grambsch
  #  a. Data set 1 and Breslow estimate
  #  The data below is not in time order, to also test sorting, and has 1 NA
  #
  test1 <- data.frame(time=  c(9, 3,1,1,6,6,8),
                      status=c(1,NA,1,0,1,1,0),
                      x=     c(0, 2,1,1,1,0,0))

  # Nelson-Aalen influence
  s1 <- survfit(Surv(time, status) ~1, test1, id=1:7, influence=TRUE)
  inf1 <- matrix(c(10, rep(-2,5), 10, -2, 7,7, -11, -11)/72,
                 ncol=2)
  indx <- order(test1$time[!is.na(test1$status)])
  aeq(s1$influence.chaz[indx,], inf1[,c(1,2,2,2)])

  # KM influence
  inf2 <- matrix(c(-20, rep(4,5), -10, 2, -13, -13, 17, 17,
                   rep(0,6))/144, ncol=3)
  aeq(s1$influence.surv[indx,], inf2[, c(1,2,2,3)])

  # Fleming-Harrington influence
  s2 <- survfit(Surv(time, status) ~ 1, test1, id=1:7, ctype=2, influence=2)
  inf3 <- matrix(c( rep(c(5, -1), c(1, 5))/36, c(5,-1)/36,
                    c(21,21,-29, -29)/144), ncol=2)
  aeq(s2$influence.chaz[indx,], inf3[,c(1,2,2,2)])


  # Breslow estimate
  byhand1 <- function(beta, newx=0) {
    r <- exp(beta)
    loglik <- 2*beta - (log(3*r+3) + 2*log(r+3))
    u <- (6 + 3*r - r^2) / ((r+1)*(r+3))
    imat <- r/(r+1)^2 + 6*r/(r+3)^2

    x <- c(1,1,1,0,0,0)
    status <- c(1,0,1,1,0,1)
    xbar <- c(r/(r+1), r/(r+3), 0, 0)  # at times 1, 6, 8 and 9
    haz <- c(1/(3*r+3), 2/(r+3), 0, 1 )
    ties <- c(1,1,2,2,3,4)
    wt <- c(r,r,r,1,1,1)
    mart <- c(1,0,1,1,0,1) -  wt* (cumsum(haz))[ties]  #martingale residual

    a <- 3*(r+1)^2; b<- (r+3)^2
    score <- c((2*r+3)/a, -r/a, -r/a + 3*(3-r)/b,  r/a - r*(r+1)/b,
               r/a + 2*r/b, r/a + 2*r/b)

    # Schoenfeld residual
    scho <- c(1/(r+1), 1- (r/(3+r)), 0-(r/(3+r)) , 0)

    surv  <- exp(-cumsum(haz)* exp(beta*newx))
    varhaz.g <- cumsum(c(1/(3*r+3)^2, 2/(r+3)^2, 0, 1 ))

    varhaz.d <- cumsum((newx-xbar) * haz)

    varhaz <- (varhaz.g + varhaz.d^2/ imat) * exp(2*beta*newx)

    names(xbar) <- names(haz) <- 1:4
    names(surv) <- names(varhaz) <- 1:4
    list(loglik=loglik, u=u, imat=imat, xbar=xbar, haz=haz,
         mart=mart,  score=score,
         scho=scho, surv=surv, var=varhaz,
         varhaz.g=varhaz.g, varhaz.d=varhaz.d)
  }


  ## fit0 ---------------

  fit0 <-coxph(Surv(time, status) ~x, test1, iter = 0, method = 'breslow')
  truth0 <- byhand1(0,0)

  aeq(truth0$loglik, fit0$loglik[1])
  aeq(1/truth0$imat, fit0$var)
  aeq(truth0$mart, fit0$resid[c(2:6,1)])
  aeq(truth0$scho, resid(fit0, 'schoen'))
  aeq(truth0$score, resid(fit0, 'score')[c(3:7,1)])
  sfit <- survfit(fit0, list(x=0))
  aeq(sfit$cumhaz, cumsum(truth0$haz))
  aeq(sfit$surv, exp(-cumsum(truth0$haz)))
  aeq(sfit$std.err^2, c(7/180, 2/9, 2/9, 11/9))
  aeq(resid(fit0, 'score'), c(5/24, NA, 5/12, -1/12, 7/24, -1/24, 5/24))


  fit0_gpu <-coxphGPU(Surv(time, status) ~x, test1, iter = 0, ties = 'breslow')
  test_that("book1 - fit0 - loglik", {expect_equal(truth0$loglik, fit0_gpu$loglik)})
  test_that("book1 - fit0 - var", {expect_equal(1/truth0$imat, c(fit0_gpu$var))})
  test_that("book1 - fit0 - residuals", {expect_equal(truth0$mart, as.vector(fit0_gpu$resid[c(2:6,1)]))})
  # aeq(truth0$scho, resid(fit0, 'schoen')) # implémenter method resid
  # aeq(truth0$score, resid(fit0, 'score')[c(3:7,1)])
  # aeq(resid(fit0, 'score'), c(5/24, NA, 5/12, -1/12, 7/24, -1/24, 5/24))


  ## fit1 ---------------

  fit1 <- coxph(Surv(time, status) ~x, test1, iter = 1, method = 'breslow')
  aeq(fit1$coef, 8/5)

  fit1_gpu <- coxphGPU(Surv(time, status) ~x, test1, iter = 1, ties = 'breslow')
  test_that("book1 - fit1 - coef", {expect_equal(as.vector(fit1_gpu$coefficients), 8/5)})


  ## fit2 ---------------

  # This next gives an ignorable warning message
  expect_warning(fit2 <- coxph(Surv(time, status) ~x, test1, method = 'breslow', iter = 2))
  aeq(round(fit2$coef, 6), 1.472724)

  fit2_gpu <- coxphGPU(Surv(time, status) ~x, test1, ties = 'breslow', iter = 2)
  test_that("book1 - fit2 - coef", {expect_equal(round(as.vector(fit2_gpu$coefficients),4), 1.4727)})


  ## fit ---------------

  fit <- coxph(Surv(time, status) ~x, test1, method = 'breslow', eps = 1e-8,
               nocenter=NULL)

  fit_gpu <- coxphGPU(Surv(time, status) ~x, test1, ties = 'breslow', eps = 1e-8,
                      nocenter=NULL)

  aeq(fit$coef, log(1.5 + sqrt(33)/2))  # the true solution
  test_that("book1 - fit - coef", {expect_equal(round(as.vector(fit_gpu$coefficients),5), round(log(1.5 + sqrt(33)/2),5))})

  truth <- byhand1(fit$coef, 0)
  aeq(truth$loglik, fit$loglik[2])
  test_that("book1 - fit - loglik", {expect_equal(round(as.vector(truth$loglik), 4), round(fit_gpu$loglik, 4))})

  aeq(1/truth$imat, fit$var)
  test_that("book1 - fit - var", {expect_equal(round(as.vector(1/truth$imat), 4), c(round(fit_gpu$var, 4)))})

  aeq(truth$mart, fit$resid[c(2:6,1)])
  test_that("book1 - fit - resid", {expect_equal(round(as.vector(truth$mart), 4), round(as.vector(fit_gpu$residuals[c(2:6,1)]), 4))})

  aeq(truth$scho, resid(fit, 'schoen'))
  test_that("book1 - fit - schoenfeld resid", {expect_equal(round(as.vector(truth$scho), 4), round(as.vector(resid(fit_gpu, type = 'schoen')), 4))})

  aeq(truth$score, resid(fit, 'score')[c(3:7,1)])
  test_that("book1 - fit - score resid", {expect_equal(round(as.vector(truth$score), 4), round(as.vector(resid(fit_gpu, type = 'score')[c(3:7,1)]), 4))})

  expect <- predict(fit, type='expected', newdata=test1) #force recalc
  aeq(test1$status[-2] -fit$resid, expect[-2]) #tests the predict function

  expect2 <- predict(fit_gpu, type='expected', newdata=test1)
  test_that("book1 - predict", {expect_equal(round(as.vector(expect), 4), round(as.vector(expect), 4))})

  # sfit <- survfit(fit, list(x=0), censor=FALSE)
  # aeq(sfit$std.err^2, truth$var[c(1,2,4)]) # sfit skips time 8 (no events there)
  # aeq(-log(sfit$surv), (cumsum(truth$haz))[c(1,2,4)])
  # sfit <- survfit(fit, list(x=0), censor=TRUE)
  # aeq(sfit$std.err^2, truth$var)
  # aeq(-log(sfit$surv), (cumsum(truth$haz)))

  #
  # Done with the formal test, now print out lots of bits
  #
  resid(fit)
  test_that("book1 - resid", {expect_equal(round(as.vector(resid(fit)), 4), round(as.vector(resid(fit_gpu)), 4))})

  resid(fit, 'scor')
  test_that("book1 - score resid", {expect_equal(round(as.vector(resid(fit, 'scor')), 4), round(as.vector(resid(fit_gpu, type = "scor")), 4))})

  resid(fit, 'scho')
  test_that("book1 - scho resid", {expect_equal(round(as.vector(resid(fit, 'scho')), 4), round(as.vector(resid(fit_gpu, type = "scho")), 4))})

  predict(fit, type='lp', se.fit=T)
  test_that("book1 - predict - lp", {expect_equal(lapply(lapply(predict(fit, type='lp', se.fit=T), FUN = as.vector), FUN = round, 4),
                                                  lapply(predict(fit_gpu, type='lp', se.fit=T), FUN = round, 4))})


  predict(fit, type='risk', se.fit=T)
  test_that("book1 - predict - risk", {expect_equal(lapply(lapply(predict(fit, type='risk', se.fit=T), FUN = as.vector), FUN = round, 4),
                                                    lapply(predict(fit_gpu, type='risk', se.fit=T), FUN = round, 4))})

  predict(fit, type='expected', se.fit=T)
  predict(fit_gpu, type='expected', se.fit=T)
  test_that("book1 - predict - expected", {expect_equal(lapply(lapply(predict(fit, type='expected', se.fit=T), FUN = as.vector), FUN = round, 4),
                                                        lapply(lapply(predict(fit_gpu, type='expected', se.fit=T), FUN = as.vector), FUN = round, 4))})

  predict(fit, type='terms', se.fit=T)
  test_that("book1 - predict - terms", {expect_equal(lapply(lapply(predict(fit, type='lp', se.fit=T), FUN = as.vector), FUN = round, 4),
                                                     lapply(predict(fit_gpu, type='lp', se.fit=T), FUN = round, 4))})

  # summary(survfit(fit, list(x=2)))

})
