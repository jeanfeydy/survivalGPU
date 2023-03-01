# CoxPH counting

    Code
      coxphGPU(Surv(Start, Stop, Event) ~ sex + age, data = drugdata)
    Output
      Call:
      coxphGPU.default(formula = Surv(Start, Stop, Event) ~ sex + age, 
          data = drugdata)
      
              coef exp(coef) se(coef)     z        p
      sex 0.620635  1.860108 0.117783 5.269 1.37e-07
      age 0.010696  1.010754 0.003964 2.698  0.00697
      
      Likelihood ratio test=33.88  on 2 df, p=4.392e-08
      n= 77038, number of events= 383 

---

    Code
      coxphGPU(Surv(Start, Stop, Event) ~ sex, data = drugdata)
    Output
      Call:
      coxphGPU.default(formula = Surv(Start, Stop, Event) ~ sex, data = drugdata)
      
            coef exp(coef) se(coef)     z        p
      sex 0.6373    1.8914   0.1177 5.416 6.09e-08
      
      Likelihood ratio test=26.59  on 1 df, p=2.521e-07
      n= 77038, number of events= 383 

