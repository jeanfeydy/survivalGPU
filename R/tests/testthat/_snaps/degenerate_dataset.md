# CoxPH counting - Breslow

    Code
      coxph(Surv(Start, Stop, Event) ~ sex + age, data = drugdata, ties = "breslow",
      control = coxph.control(iter.max = 0))
    Output
      Call:
      coxph(formula = Surv(Start, Stop, Event) ~ sex + age, data = drugdata, 
          control = coxph.control(iter.max = 0), ties = "breslow")
      
             coef exp(coef) se(coef) z p
      sex 0.00000   1.00000  0.13770 0 1
      age 0.00000   1.00000  0.00394 0 1
      
      Likelihood ratio test=0  on 2 df, p=1
      n= 77038, number of events= 383 

---

    Code
      coxph(Surv(Start, Stop, Event) ~ sex + age, data = drugdata, ties = "breslow",
      control = coxph.control(iter.max = 1))
    Output
      Call:
      coxph(formula = Surv(Start, Stop, Event) ~ sex + age, data = drugdata, 
          control = coxph.control(iter.max = 1), ties = "breslow")
      
             coef exp(coef) se(coef)     z        p
      sex 0.73736   2.09040  0.11534 6.393 1.63e-10
      age 0.01058   1.01064  0.00397 2.665  0.00769
      
      Likelihood ratio test=32.7  on 2 df, p=7.919e-08
      n= 77038, number of events= 383 

