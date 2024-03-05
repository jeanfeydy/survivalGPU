#!/bin/bash

cd /home/survivalGPU/simulations/


# python simulation_and_experiment.py 
lauch_experiment.sh
Rscript gpu_computation_time_comparison.r
# python computation_time_gpu.py
# python analysis.py

# python post_clep_analysis_error_based.py

# rm "Simulation_results/simulation_parameters.json"