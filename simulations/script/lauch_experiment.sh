#!/bin/bash

cd /home/survivalGPU/simulations/

mkdir 


python simulation_and_experiment.py 

Rscript simulate_WCEmat.r

python analysis.py

python post_clep_analysis_error_based.py

# rm "Simulation_results/simulation_parameters.json"