hello_simulation <- function(){
  survivalgpu <- use_survivalGPU()

  print("Hello simulation")
}


# permalgoWCE <- function(
#   n_patients,
#   max_time, 
#   Xmat,
#   betas,
#   names, 
#   wce_status,
#   scenarios,
#   eventRandom,
#   censorRandom
# ){
#   survivalgpu <- use_survivalGPU()

#   simualtion <- survivalgpu$WCE_permalgo

#                  max_time, 
#                  Xmat,
#                  betas,
#                  names, 
#                  wce_status,
#                  scenarios,
#                  eventRandom,
#                  censorRandom)



simulation_paper <- function(
  n_patients,
  max_time, 
  scenario_name,
  HR_target
){
  survivalgpu <- use_survivalGPU()

  simulate_for_experiment = survivalgpu$simulate_for_experiment

simulation_iteration <- function(
    n_patients,
    max_time, 
    scenario_name,
    HR_target
){
    survivalgpu <- use_survivalGPU()

    simulate_for_experiment = survivalgpu$simulate_for_experiment

    dataset = simulate_for_experiment(
        n_patients = n_patients, 
        max_time = max_time, 
        scenario_name = scenario_name,
        HR_target = HR_target
    )

    model_cpu = WCE::WCE(
        data = dataset,
        analysis = "Cox",
        nknots=c(3),
        cutoff=180,
        id="patients",
        event = "events",
        start = "start",
        stop = "stop",
        expos = "dose",
    )

    exposed   <- rep(1, 180)
    unexposed <- rep(0, 180)

    HR_cpu = WCE::HR.WCE(model_cpu, exposed, unexposed)

    model_gpu = wceGPU(
        data = dataset,
        nknots = 3, 
        cutoff = 180, 
        id = "patients",
        event = "events",
        start = "start",
        stop = "stop",
        expos = "dose",
        constrained = "r",
        verbosity = 0
    )

    HR_gpu = HR(model_gpu, exposed, unexposed)

    return(list(HR_cpu = HR_cpu, HR_gpu = HR_gpu))
}  print("before going python")
  dataset = simulate_for_experiment(
    n_patients = n_patients, 
    max_time = max_time, 
    scenario_name = scenario_name,
    HR_target = HR_target
  )

  return(dataset)


}






