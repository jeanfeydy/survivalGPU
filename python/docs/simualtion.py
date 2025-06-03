import torch
import os
import sys

sys.path.append("dev/survivalGPU/python")
from survivalgpu.simulation import simulate_dataset, WCECovariate, TimeDependentCovariate, ConstantCovariate, simulate_dataset
from survivalgpu.coxph import coxph_R
from survivalgpu.wce import wce_R

import matplotlib.pyplot as plt
import numpy as np
import torch

wce_covariate_1 = WCECovariate(name="wce_1",
                            values = [1,1.5,2,2.5,3],
                            scenario_name="exponential_scenario",
                            HR_target=1.5)


wce_covariate_2 = WCECovariate(name="wce_2",
                            values = [1,1.5,2,2.5,3],
                            scenario_name="exponential_scenario",
                            HR_target=2)

list_wce_covariates = [wce_covariate_1,wce_covariate_2]
list_wce_covariates = [wce_covariate_1,wce_covariate_2]

constant_covariate = ConstantCovariate(name="constant",
                                        values=[0,1],
                                        weights=[1,2],
                                        coef = np.log(2))

time_dependent_covariate = TimeDependentCovariate(name="time_dependent",
                                                values = [0,1,1.5,2,2.5,3],
                                                coef = np.log(1.5))

time_dependent_covariate_cumulative = TimeDependentCovariate(name="time_dependent",
                                                values = [0,1,1.5,2,2.5,3],
                                                coef = np.log(1.5),
                                                cumulative = True,
                                                cutoff = 3)



max_time = 10
n_patients = 5






# covariate = constant_covariate.initialize_experiment(max_time=max_time, n_patients=n_patients).generate_Xvector()

# covariate = time_dependent_covariate.initialize_experiment(max_time=max_time, n_patients=n_patients).generate_Xvector().cumulative_exposure(cutoff = 10).Xvector

constant_covariate = constant_covariate.initialize_experiment(max_time=max_time, n_patients=n_patients)
print(constant_covariate.Xvector)
time_dependent_covariate = time_dependent_covariate.initialize_experiment(max_time=max_time, n_patients=n_patients)
print(time_dependent_covariate.Xvector)
time_dependent_covariate_cumulative = time_dependent_covariate_cumulative.initialize_experiment(max_time=max_time, n_patients=n_patients)
print(time_dependent_covariate_cumulative.Xvector)

wce_covariate_1 = wce_covariate_1.initialize_experiment(max_time=max_time, n_patients=n_patients)
print(wce_covariate_1.Xvector)
print(wce_covariate_1.WCEvector)


quit()

# cox_result = coxph_R(

# result_cox = coxph_R(data= dataset,
#                      stop = "stop",
#                      death= "events",
#                      covars = ["constant", "time_dependent"])

# print(result_cox["coef"])

                     


result = wce_R(data= dataset, 
               ids = "patients", 
               covars = ["constant","time_dependent"],
               stop = "stop",
               doses = "wce_1", 
               events = "events",
               cutoff = 180,
               nknots = 2,
               constrained = "Right")





print(result["coef"])


# def plot_wce(
#         WCE_object,
#         scenario =None,
#         HR_target = None):
    
#     WCE_mat = WCE_object["WCEmat"]
            
#     plt.plot(np.arange(0, WCEmat.shape[1]),WCE_mat[0], c = "black", label = "calculated WCE function")

#     if WCE_mat.shape[0] > 1:
#         quantiles = np.quantile(WCE_mat, [0.025, 0.975], axis = 0)
#         plt.plot(np.arange(0, WCEmat.shape[1]),quantiles[0], c = "black", linestyle = "--")
#         plt.plot(np.arange(0, WCEmat.shape[1]),quantiles[1], c = "black", linestyle = "--")

    
    
#     if scenario is not None:
#         scenario_shape = get_scenario(scenario,365)[:WCEmat.shape[1]] * np.log(HR_target)
#         plt.plot((np.arange(0, WCEmat.shape[1])),scenario_shape[:WCEmat.shape[1]], c = "red", label = "scenario shape")


# plot_wce(result, scenario = "exponential_scenario", HR_target = 1.5)
    
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
RH_result = HR(result, vecnum, vecdenom)

print(RH_result)




