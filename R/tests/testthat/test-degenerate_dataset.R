
library(survival)
drugdata <- WCE::drugdata

# Check iter.max feature
expect_snapshot({
  coxph(Surv(Start,Stop, Event) ~ sex + age,
        data = drugdata,
        ties = "breslow",
        control = coxph.control(iter.max = 0))
})

expect_snapshot({
  coxph(Surv(Start,Stop, Event) ~ sex + age,
        data = drugdata,
        ties = "breslow",
        control = coxph.control(iter.max = 1))
})

# Small dataset with event at t5 for all
n_id <- 6
set.seed(1234)
df_event_t5 <- data.frame(Id = rep(1:n_id, each = 5),
                          Start = rep(0:4, n_id),
                          Stop = rep(1:5, n_id),
                          Event = rep(c(0, 0, 0, 0, 1), n_id),
                          cov1 = rnorm(5 * n_id))

coxph(Surv(Start,Stop, Event) ~ cov1,
      data = df_event_t5,
      ties = "breslow")

# Small dataset with two strates (Sex), but only one Id in a strata
n_id <- 15
set.seed(12345)
event_list <- lapply(rbinom(n_id, 1, 0.5), function(x) c(0, 0, 0, 0, x))
event <- do.call("c", event_list)
df_one_id_in_strata <- data.frame(Id = rep(1:n_id, each = 5),
                                  Start = rep(0:4, n_id),
                                  Stop = rep(1:5, n_id),
                                  Event = event,
                                  Sex = c(rep("1", 5), rep("2", 5 * (n_id - 1))),
                                  cov1 = rnorm(5 * n_id))

coxph(Surv(Start,Stop, Event) ~ cov1 + strata(Sex),
                    data = df_one_id_in_strata,
                    ties = "breslow")

coxph(Surv(Start,Stop, Event) ~ cov1,
      data = df_one_id_in_strata,
      ties = "breslow")

coxph(Surv(Start,Stop, Event) ~ cov1 + Sex,
      data = df_one_id_in_strata,
      ties = "breslow")
