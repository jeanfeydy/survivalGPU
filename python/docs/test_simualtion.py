import torch
import os
import sys

sys.path.append("dev/survivalGPU/python")
from survivalgpu.simulation import simulate_dataset, WCECovariate, TimeDependentCovariate, ConstantCovariate, global_Xmat_wce_mat, get_probas,matching_algo,event_censor_generation,event_FUP_Ti_generation, get_dataset

wce_covariate_1 = WCECovariate(name="wce_1",
                            doses= [1,1.5,2,2.5,3],
                            scenario_name="exponential_scenario",
                            HR_target=1.5)


wce_covariate_2 = WCECovariate(name="wce_2",
                            doses= [1,1.5,2,2.5,3],
                            scenario_name="exponential_scenario",
                            HR_target=2)

list_wce_covariate = [wce_covariate_1,wce_covariate_2]
list_wce_covariate = []

constant_covariate = ConstantCovariate(name="constant",
                                        values=[0,1],
                                        weights=[1,2],
                                        beta = 0.5)

time_dependent_covariate = TimeDependentCovariate(name="time_dependent",
                                                values = [0,1,1.5,2,2.5,3],
                                                beta = 0.7)

list_cox_covariate = [constant_covariate,time_dependent_covariate]
list_cox_covariate = [constant_covariate]


import numpy as np

import torch

device = "cpu"

n_patients = 1000
n_covariates = len(list_wce_covariate) + len(list_cox_covariate)
max_time = 100



dataset = simulate_dataset(max_time = max_time, n_patients = n_patients, 
                     list_wce_covariates = list_wce_covariate, 
                     list_cox_covariates= list_cox_covariate)

print(dataset)



