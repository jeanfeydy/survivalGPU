import torch
import os
import sys

sys.path.append("dev/survivalGPU/python")
from survivalgpu.simulation import simulate_dataset, WCECovariate, TimeDependentCovariate, ConstantCovariate, global_Xmat_wce_mat, get_probas,matching_algo,event_censor_generation,event_FUP_Ti_generation, get_dataset
from survivalgpu.coxph import coxph_R
from survivalgpu.wce import wce_R

import matplotlib.pyplot as plt

wce_covariate_1 = WCECovariate(name="wce_1",
                            doses= [1,1.5,2,2.5,3],
                            scenario_name="exponential_scenario",
                            HR_target=1.5)


wce_covariate_2 = WCECovariate(name="wce_2",
                            doses= [1,1.5,2,2.5,3],
                            scenario_name="exponential_scenario",
                            HR_target=2)

list_wce_covariate = [wce_covariate_1,wce_covariate_2]
list_wce_covariate = [wce_covariate_1]

constant_covariate = ConstantCovariate(name="constant",
                                        values=[0,1],
                                        weights=[1,2],
                                        beta = 1)

time_dependent_covariate = TimeDependentCovariate(name="time_dependent",
                                                values = [0,1,1.5,2,2.5,3],
                                                beta = 0.5)

list_cox_covariate = [constant_covariate,time_dependent_covariate]
list_cox_covariate = []


import numpy as np

import torch

device = "cpu"

n_patients = 5000
n_covariates = len(list_wce_covariate) + len(list_cox_covariate)
max_time = 365



dataset = simulate_dataset(max_time = max_time, n_patients = n_patients, 
                     list_wce_covariates = list_wce_covariate, 
                     list_cox_covariates= list_cox_covariate)

print(dataset)


# res = coxph_R(
#             dataset,
#             "stop",
#             "events",
#             ["constant"],
#             bootstrap=1,
#             profile=None,
#         )

# print(res["coef"])

result = wce_R(data= dataset, 
               ids = "patients", 
               covars = None,
               stop = "stop",
               doses = "wce_1", 
               events = "events",
               cutoff = 180,
               nknots = 2,
               constrained = "Right")


print(result)


def plot_wce(
        WCE_object,
        scenario =None,
        HR_target = None):
    
    WCE_mat = WCE_object["WCEmat"]
            
    plt.plot(np.arange(0, WCEmat.shape[1]),WCE_mat[0], c = "black", label = "calculated WCE function")

    if WCE_mat.shape[0] > 1:
        quantiles = np.quantile(WCE_mat, [0.025, 0.975], axis = 0)
        plt.plot(np.arange(0, WCEmat.shape[1]),quantiles[0], c = "black", linestyle = "--")
        plt.plot(np.arange(0, WCEmat.shape[1]),quantiles[1], c = "black", linestyle = "--")

    
    
    if scenario is not None:
        scenario_shape = get_scenario(scenario,365)[:WCEmat.shape[1]] * np.log(HR_target)
        plt.plot((np.arange(0, WCEmat.shape[1])),scenario_shape[:WCEmat.shape[1]], c = "red", label = "scenario shape")


plot_wce(result, scenario = "exponential_scenario", HR_target = 1.5)
    
# vecnu
# hr_matrix = np.exp(np.dot(WCEmat, vecnum)) / np.exp(np.dot(WCEmat, vecdenom))
# hr =hr_matrix[0]
# print(hr)
# print()

def HR(WCE_object, vecnum, vecdenom):

    WCEmat = WCE_object["WCEmat"]
    n_bootsraps = WCEmat.shape[0]
    HR_matrix = np.exp(np.dot(WCEmat, vecnum)) / np.exp(np.dot(WCEmat, vecdenom))

    HR = [HR_matrix[0]]
    if n_bootsraps == 1:
        return HR
    
    else:
        HR_quantiles = np.quantile(HR_matrix, [0.025, 0.975], axis = 0)
        HR.append(HR_quantiles[0])
        HR.append(HR_quantiles[1])
        return HR

vecnum = np.ones(180)
vecdenom = np.zeros(180)
HR(result, vecnum, vecdenom)

print(HR)