simulation_paper <- function(
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

  return(dataset)


}
