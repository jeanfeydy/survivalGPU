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

