import numpy as np
import random
import pandas as pd
import taichi as ti
import taichi.math as tm
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity


import time

import scenarios


# TODO : modify the TDhist to be able to manage a bigger variety of cases, 
# maybe create another  TDhist that is more in tune with the kind of data given by the SNDS
def TDhist(observation_time,doses):
    """
    This function is used to generate individual time-dependant exposure history 
    Generate prescirption of different duration and doses 
    """

    duration = int(7 + 7*np.round(np.random.lognormal(mean =0.5, sigma =0.8, size = 1)).item())
    # duration is in weeks *

    dose = random.choice(doses)
    exposure_vector = np.repeat(dose,repeats = duration)



    while len(exposure_vector) <= observation_time:
        intermission = int(7 + 7*np.round(np.random.lognormal(mean =0.5, sigma =0.8, size = 1)).item())
        duration = int(7 + 7*np.round(np.random.lognormal(mean =0.5, sigma =0.8, size = 1)).item())

        dose = random.choice(doses)

        exposure_vector = np.concatenate((exposure_vector,np.repeat(0,repeats = intermission),np.repeat(dose,repeats = duration)))

    return exposure_vector[:observation_time]


# TODO : here should have a way to use more things for TDhist
def generate_Xmat(observation_time,n_patients,doses):
    """
    Generate the Xmat of TDHist for each indivudual patient
    """
    Xmat = np.matrix([TDhist(observation_time,doses) for i in range(n_patients)]).transpose()
    return Xmat



def generate_wce_vector(u, scenario_shape, Xmat):
    """
    This function generate the wce vector of the n patients at time u
    """
    t_array = np.arange(1,u+1)


    wce = np.multiply(scenario_shape[:u].reshape(u,1),Xmat[:u,:])
    if u == 1:
        return wce
    return np.sum(wce, axis = 0)



def generate_wce_mat(scenario_name, Xmat, cutoff, max_time):
    """
    This function generate the wce mattrix that keep the WCE wieght of all the patient at
    all the times intill the cutoff
    """
    scenario_shape = np.concatenate((scenarios.get_scenario(scenario_name,cutoff),np.zeros(max_time-cutoff)))
    wce_mat = np.vstack([generate_wce_vector(u, scenario_shape, Xmat) for u in range(1,max_time+1)])
    return wce_mat


def event_censor_generation(max_time, n_patients, censoring_ratio):
    """
    This function generate random event and censoring times 
    It will a proportion of censoring_times according to the censoring ration 
    """

    if censoring_ratio > 1:
        raise ValueError("The censoring ration must be inferior to 1")
    if censoring_ratio < 0:
        raise ValueError("The censoring ration must be positive")

     # Event times : Uniform[1;365] for all scenarios
    eventRandom = np.round(np.random.uniform(low = 1, high = max_time, size = n_patients)).astype(int)

    # TODO Maybe change the way the censoring times are determined to that there is no randolness on the number of 
    # TODO patients that are not censored
    censorRandom = np.round(np.random.uniform(low = 1, high = max_time* int(1/censoring_ratio), size = n_patients)).astype(int)

    event = np.array([1 if eventRandom[i]< censorRandom[i] else 0 for i in range(len(eventRandom))])
    FUP_Ti = np.minimum(eventRandom,censorRandom)


    return event, FUP_Ti


def matching_algo(wce_mat, max_time:int, n_patients:int, HR_target):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    total_checking_choice = 0
    
    events, FUP_tis = event_censor_generation(max_time, n_patients, censoring_ratio=0.5)


    ids_torch = torch.arange(0,n_patients, dtype = int).to(device)
    wce_id_indexes = torch.zeros(len(ids_torch),dtype = int).to(device)





    wce_mat_torch = torch.from_numpy(wce_mat).to(device)

    
    
    for i in range(n_patients):
        if i % (n_patients/20) == 0:
            print(i)
        
     
        event = events[i]
        time_event = FUP_tis[i]

        if event == 0:
            
            id_index = torch.randint(0,len(ids_torch),(1,))
            wce_id = ids_torch[id_index]
            ids_torch = ids_torch[ids_torch != wce_id]


        else:

            wce_mat_current_torch = wce_mat_torch[:,ids_torch]
            exp_vals = torch.exp(HR_target * wce_mat_current_torch[time_event - 1,])
            exp_sum = torch.sum(exp_vals)
            proba_torch = exp_vals/exp_sum
            id_index = torch.multinomial(input = proba_torch, num_samples= 1)
            wce_id = ids_torch[id_index]
            ids_torch = ids_torch[ids_torch != wce_id]


        wce_id_indexes[i] = wce_id

            

    wce_id_indexes = np.array(wce_id_indexes.to("cpu"))
    


    return wce_id_indexes, events, FUP_tis



def get_dataset(Xmat, wce_mat, HR_target):

    max_time,n_patients = wce_mat.shape[0], wce_mat.shape[1]
    wce_id_indexes, events, FUP_tis = matching_algo(wce_mat, max_time,n_patients, HR_target) # wce_mat



    ordered_events = np.array(events)[wce_id_indexes]
    ordered_FUP_tis = np.array(FUP_tis)[wce_id_indexes]
    

    data_field = ti.field(dtype=ti.i64, shape=(n_patients*max_time,5))
    print(data_field.shape)

    Xmat_transposed = Xmat.transpose()



    patient_time_field= ti.field(dtype=int, shape=(n_patients, max_time))




    @ti.kernel
    def iteration_dataset(ordered_events:ti.types.ndarray() ,ordered_FUP_tis:ti.types.ndarray(), 
                          wce_id_indexes:ti.types.ndarray(), Xmat_transposed:ti.types.ndarray(),
                          max_time:int
                          ):

        line = 0 

        for patient_id, time in patient_time_field:


            event = ordered_events[patient_id]
            b= ordered_FUP_tis[patient_id]
            


            if time < ordered_FUP_tis[patient_id]:

                patient = patient_id
                event = 0
                dose = Xmat_transposed[patient_id,time]
                time_start = time
                time_stop = time +1 

                ##TODO cahnge 365 by max_time
                line = patient_id*max_time + time

                data_field[line,0] = patient_id +1
                data_field[line,1] = time
                data_field[line,2] = time +1 
                data_field[line,3] = 0
                data_field[line,4] = Xmat_transposed[patient_id,time]          
                




            elif time == ordered_FUP_tis[patient_id]:
                patient = patient_id
                event = ordered_events[patient_id]
                dose = Xmat_transposed[patient_id,time]
                time_start = time
                time_stop = time +1 

                line = patient_id*max_time + time

                data_field[line,0] = patient_id +1 
                data_field[line,1] = time
                data_field[line,2] = time +1 
                data_field[line,3] = event
                data_field[line,4] = Xmat_transposed[patient_id,time]
            


        

    iteration_dataset(ordered_events,ordered_FUP_tis,wce_id_indexes,Xmat_transposed,max_time)



    data_numpy = data_field.to_numpy()

    filtered_data = data_numpy[~np.all(data_numpy == 0, axis=1)]

    
    return filtered_data

n_patients = 10
max_time = 365
cutoff = 180
HR_target = 1.5

Xmat = generate_Xmat(max_time,n_patients,[1,2,3])

scenario= "exponential_scenario"


wce_mat = generate_wce_mat(scenario_name= scenario, Xmat = Xmat, cutoff = cutoff, max_time= max_time)




#############################################""


ti.init(arch=ti.gpu)

start_simulation_time = time.perf_counter()

# wce_id_indexes, events, FUP_tis = cpu_matching_algo(wce_mat, max_time,n_patients, HR_target=1.5) # wce_mat


numpy_wce = get_dataset(Xmat, wce_mat, 1.5)

end_simulation_time = time.perf_counter()

elapsed_simulation_time = end_simulation_time - start_simulation_time 

print(f"Simulation_time : {elapsed_simulation_time}")


df_wce = pd.DataFrame(numpy_wce, columns = ["patient","start","stop","event","dose"])


df_wce.to_csv("test_df")



# 5000 patients : 158 GPU

