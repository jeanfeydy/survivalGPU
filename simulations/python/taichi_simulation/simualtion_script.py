import numpy as np
import random
import pandas as pd
import taichi as ti
import taichi.math as tm
import torch

import time

import scenarios
import simulate_WCEmat



ti.init(arch=ti.cpu)

n_patients_list = [100,200,500,1000] #[100,200,500,1000,2000,5000,10000,20000,50000,100000]
max_time = 365
cutoff = 180
HR_target = 1.5

doses = [0.5,1,1.5,2,2.5,3]

scenario= "exponential_scenario"
ti.init(arch=ti.gpu)


# computation_time_results = pd.DataFrame

computation_times = []




for n_patients in n_patients_list:


    print(f"start simulation for {n_patients} patients")

    start_computation_time = time.perf_counter()
    Xmat = simulate_WCEmat.generate_Xmat(max_time,n_patients,doses)
    wce_mat = simulate_WCEmat.generate_wce_mat(scenario_name= scenario, Xmat = Xmat, cutoff = cutoff, max_time= max_time)
    numpy_wce, elapsed_matching_time, elapsed_dataset_time = simulate_WCEmat.torch_get_dataset_gpu(Xmat, wce_mat, 1.5)
    end_computation_time = time.perf_counter()
    elapsed_computation_time = end_computation_time - start_computation_time 


    computation_times.append(elapsed_computation_time)

    print(f"The simulation for {n_patients} patients took: {computation_times}s")

# print(np.array[np.array(n_patients),np.array(computation_times))
    


data = np.array([n_patients_list,computation_times]).transpose()
print(data.shape)


print(data)

df_computation_times = pd.DataFrame(data, columns = ["n_patients", "computation_times"])
df_computation_times.to_csv("computation_times_results")






