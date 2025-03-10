import sys

sys.path.append("../../../python")
from survivalgpu.simulation import simulate_for_experiment


max_time = 365
HR_target = 4
scenario_name = "exponential_scenario"



for n_patients in [100,500]: #[500,1000,5000,10000,50000]:
    data = simulate_for_experiment(n_patients, 
                        max_time,
                        HR_target, 
                        scenario_name)
    
    data.to_csv(f"../../benchmark_datasets/{n_patients}.csv", index=False)