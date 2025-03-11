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

  print("before going python")
  dataset = simulate_for_experiment(
    n_patients = n_patients, 
    max_time = max_time, 
    scenario_name = scenario_name,
    HR_target = HR_target
  )

  return(dataset)


}






