# CoxPH counting

    Code
      coxphGPU(Surv(Start, Stop, Event) ~ sex + age, data = drugdata, ties = "efron")
    Output
      Call:
      coxphGPU.default(formula = Surv(Start, Stop, Event) ~ sex + age, 
          data = drugdata, ties = "efron")
      
              coef exp(coef) se(coef)     z        p
      sex 0.620635  1.860108 0.117783 5.269 1.37e-07
      age 0.010696  1.010754 0.003964 2.698  0.00697
      
      Likelihood ratio test=33.88  on 2 df, p=4.392e-08
      n= 77038, number of events= 383 

---

    Code
      coxphGPU(Surv(Start, Stop, Event) ~ sex, data = drugdata, ties = "efron")
    Output
      Call:
      coxphGPU.default(formula = Surv(Start, Stop, Event) ~ sex, data = drugdata, 
          ties = "efron")
      
            coef exp(coef) se(coef)     z        p
      sex 0.6373    1.8914   0.1177 5.416 6.09e-08
      
      Likelihood ratio test=26.59  on 1 df, p=2.521e-07
      n= 77038, number of events= 383 

# CoxPH counting - Breslow

    Code
      coxphGPU(Surv(Start, Stop, Event) ~ sex + age, data = drugdata, ties = "breslow")
    Output
      Call:
      coxphGPU.default(formula = Surv(Start, Stop, Event) ~ sex + age, 
          data = drugdata, ties = "breslow")
      
              coef exp(coef) se(coef)     z        p
      sex 0.619201  1.857443 0.117770 5.258 1.46e-07
      age 0.010671  1.010728 0.003963 2.692  0.00709
      
      Likelihood ratio test=33.72  on 2 df, p=4.751e-08
      n= 77038, number of events= 383 

---

    Code
      coxphGPU(Surv(Start, Stop, Event) ~ sex, data = drugdata, ties = "breslow")
    Output
      Call:
      coxphGPU.default(formula = Surv(Start, Stop, Event) ~ sex, data = drugdata, 
          ties = "breslow")
      
            coef exp(coef) se(coef)    z        p
      sex 0.6333    1.8838   0.1177 5.38 7.45e-08
      
      Likelihood ratio test=26.46  on 1 df, p=2.689e-07
      n= 77038, number of events= 383 

