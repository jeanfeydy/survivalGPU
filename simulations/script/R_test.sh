#!/bin/bash

EXPERIMENT_NAME="test of this fucking R thingy"


echo lauchning experiment $EXPERIMENT_NAME
cd /home/survivalGPU/simulations/
# mkdir ../Simulation_results/$EXPERIMENT_NAME
# mkdir ../Simulation_results/$EXPERIMENT_NAME/models
Rscript simulate_WCEmat.r
