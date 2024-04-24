import numpy as np
import random
import pandas as pd
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
from pathlib import Path 
from scipy.stats import norm


from .coxph import coxph_torch


import time



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
    u_t_array = u  - t_array 


    
    wce = np.multiply(scenario_shape[u_t_array].reshape(u,1),Xmat[t_array -1,:])

    return np.sum(wce, axis = 0)



def generate_wce_mat(scenario_name, Xmat, max_time):
    """
    This function generate the wce mattrix that keep the WCE wieght of all the patient at
    all the times intill the cutoff
    """

    max_time = int(max_time)
    scenario_shape = get_scenario(scenario_name,365)

    wce_mat = np.vstack([generate_wce_vector(u, scenario_shape, Xmat) for u in range(1,max_time+1)])


    return wce_mat

def event_censor_from_R(eventRandom, censorRandom):
    event = np.array([1 if eventRandom[i]<= censorRandom[i] else 0 for i in range(len(eventRandom))]).astype(int)
    FUP_Ti = np.minimum(eventRandom,censorRandom).astype(int)
    return event, FUP_Ti

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
    # print(eventRandom)
    

    # TODO Maybe change the way the censoring times are determined to that there is no randolness on the number of 
    # TODO patients that are not censored
    censorRandom = np.round(np.random.uniform(low = 1, high = max_time* int(1/censoring_ratio), size = n_patients)).astype(int)
    # print(censorRandom)

    event = np.array([1 if eventRandom[i]< censorRandom[i] else 0 for i in range(len(eventRandom))]).astype(int)
    # print(event)
    FUP_Ti = np.minimum(eventRandom,censorRandom).astype(int)
    # print(FUP_Ti)

    # quit()

    sorted_indices = np.argsort(FUP_Ti)
    event = event[sorted_indices]
    FUP_Ti = FUP_Ti[sorted_indices]


    return event, FUP_Ti


def matching_algo(wce_mat, max_time:int, n_patients:int, HR_target,events, FUP_tis):
    events = events.copy()
    FUP_tis = FUP_tis.copy()
    wce_mat = wce_mat.copy()
    events = np.array(events, dtype = int)
    FUP_tis = np.array(FUP_tis, dtype = int)

    df_wce_mat = pd.DataFrame(wce_mat)
    df_wce_mat.to_csv("wce_mat_not_pure")


    n_patients = int(n_patients)
    max_time = int(max_time)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    total_checking_choice = 0
    


    ids_torch = torch.arange(0,n_patients, dtype = int).to(device)
    wce_id_indexes = torch.zeros(len(ids_torch),dtype = int).to(device)


    wce_mat_torch = torch.from_numpy(wce_mat).to(device)

        
    for i in range(n_patients):   

    
        event = events[i]
        time_event = FUP_tis[i]

        if event == 0:
            
            id_index = torch.randint(0,len(ids_torch),(1,))
            wce_id = ids_torch[id_index]
            ids_torch = ids_torch[ids_torch != wce_id]


        else:
            wce_mat_current_torch = wce_mat_torch[:,ids_torch]


            exp_vals = torch.exp(np.log(HR_target) * wce_mat_current_torch[time_event - 1,]) 
            exp_sum = torch.sum(exp_vals)
            proba_torch = exp_vals/exp_sum
            selection_list = torch.zeros(len(proba_torch))
            # for i in range(100000):
            #     id_index = torch.multinomial(input = proba_torch, num_samples= 1)
            #     selection_list[id_index] += 1
            # print(selection_list/100000)
            id_index = torch.multinomial(input = proba_torch, num_samples= 1)

   
            wce_id = ids_torch[id_index]
            ids_torch = ids_torch[ids_torch != wce_id]



        wce_id_indexes[i] = wce_id

            

    wce_id_indexes = np.array(wce_id_indexes.to("cpu"))
    


    return wce_id_indexes




def get_dataset_test_R(Xmat, wce_mat, HR_target, FUP_tis, events, wce_id_indexes):


    print(FUP_tis)
    print(events)

    wce_id_indexes = np.array(wce_id_indexes, dtype = int) -1

    max_time,n_patients = wce_mat.shape[0], wce_mat.shape[1]
     # wce_mat


    ordered_events = np.array(events, dtype = int) # [wce_id_indexes]
    ordered_FUP_tis = np.array(FUP_tis, dtype = int) # [wce_id_indexes]

    print(ordered_FUP_tis.transpose().squeeze())
    ordered_FUP_tis = ordered_FUP_tis.transpose().squeeze()
    ordered_events = ordered_events.transpose().squeeze()
    Xmat_transposed = Xmat.transpose()[wce_id_indexes,]





    number_lines = ordered_FUP_tis.sum()

    patient_id_array = np.zeros(number_lines, dtype = int)
    event_array = np.zeros(number_lines, dtype = int)
    time_start_array = np.zeros(number_lines, dtype = int)
    time_stop_array = np.zeros(number_lines, dtype = int)
    doses_aray = np.zeros(number_lines, dtype = np.float64)

    i = 0




    dataset_start = time.perf_counter()
    for patient_id in range(n_patients):

        
        for time_start in range(ordered_FUP_tis[patient_id] -1):
            
            patient_id_array[i] = patient_id +1
            time_start_array[i] = time_start
            time_stop_array[i] = time_start +1
            doses_aray[i] = Xmat_transposed[patient_id,time_start]
            i += 1
        patient_id_array[i] = patient_id +1
        time_start_array[i] = time_start +1
        time_stop_array[i] = time_start +2
        event_array[i] = ordered_events[patient_id]
        doses_aray[i] = Xmat_transposed[patient_id,time_start]
        i += 1
    dataset_end = time.perf_counter()
    elapsed_dataset_time = dataset_end-dataset_start

    print("elapsed_time dataset :",elapsed_dataset_time)

    # print(patient_id_array)
    # print(time_start_array)
    # print(time_stop_array)
    # print(event_array)
    # print(doses_aray)


    df_wce = pd.DataFrame()
    # ["patient","start","stop","event","dose"])
    df_wce["patient"] = patient_id_array
    df_wce["start"] = time_start_array
    df_wce["stop"] = time_stop_array
    df_wce["event"] = event_array
    df_wce["dose"] = doses_aray

    print(df_wce)


    
    return df_wce



def get_dataset(Xmat, max_time,n_patients, HR_target, FUP_tis, events, wce_id_indexes):

    FUP_tis = np.array(FUP_tis, dtype = int)
    events = np.array(events, dtype = int)
    n_patients = int(n_patients)



    wce_id_indexes = np.array(wce_id_indexes, dtype = int) 

     # wce_mat


    ordered_events = events
    ordered_FUP_tis = FUP_tis


    Xmat_transposed = Xmat.transpose()




    number_lines = ordered_FUP_tis.sum()

    patient_id_array = np.zeros(number_lines, dtype = int)
    event_array = np.zeros(number_lines, dtype = int)
    time_start_array = np.zeros(number_lines, dtype = int)
    time_stop_array = np.zeros(number_lines, dtype = int)
    doses_aray = np.zeros(number_lines, dtype = np.float64)

    i = 0




    dataset_start = time.perf_counter()


    for patient_id in range(n_patients):



        
        for time_start in range(ordered_FUP_tis[patient_id]):
            
            patient_id_array[i] = patient_id +1
            time_start_array[i] = time_start
            time_stop_array[i] = time_start +1
            doses_aray[i] = Xmat_transposed[wce_id_indexes[patient_id],time_start]
            
            if time_start == ordered_FUP_tis[patient_id] - 1:
                event_array[i] = ordered_events[patient_id]
            else:
                event_array[i] = 0
            i += 1
 
    dataset_end = time.perf_counter()
    elapsed_dataset_time = dataset_end-dataset_start



    df_wce = pd.DataFrame()
    # ["patient","start","stop","event","dose"])
    df_wce["patient"] = patient_id_array
    df_wce["start"] = time_start_array
    df_wce["stop"] = time_stop_array
    df_wce["event"] = event_array
    df_wce["dose"] = doses_aray



    
    return df_wce


def save_dataframe(numpy_wce, n_patients,HR_target, scenario):

    df_wce = pd.DataFrame(numpy_wce, columns = ["patient","start","stop","event","dose"])
    print( str(HR_target))
    saving_path = Path("../../simulated_datasets") / scenario / str(HR_target) / str(n_patients) / "dataset.csv"
    df_wce.to_csv(saving_path)

def simulate_dataset(max_time, n_patients, doses, scenario, HR_target):


    max_time = int(max_time)
    n_patients = int(n_patients)

    Xmat = generate_Xmat(max_time,n_patients,doses)
    wce_mat = generate_wce_mat(scenario_name= scenario, Xmat = Xmat, max_time= max_time)
    df_wce_mat = pd.DataFrame(wce_mat)

    events, FUP_tis = event_censor_generation(max_time, n_patients, censoring_ratio=0.5)
    wce_id_indexes  = matching_algo(wce_mat, max_time,n_patients, HR_target,events, FUP_tis)
    numpy_wce = get_dataset(Xmat, max_time,n_patients, HR_target, FUP_tis,events,wce_id_indexes)
    df_wce = pd.DataFrame(numpy_wce, columns = ["patient","start","stop","event","dose"])

    return df_wce



#### 
def exponential_scenario(u_t, name = False):
    return((7 * np.exp(-7*u_t/365))) # divide by 365 in order to have a t in days

def bi_linear_scenario(u_t):
    if u_t < 50:
        return (1- (u_t/365)/(50/365))
    return 0 

def early_peak_scenario(u_t):
    return norm.pdf(u_t/365, 0.04, 0.05)


def inverted_u_scenario(u_t):
    return norm.pdf(u_t/365, 0.04, 0.05)



def get_scenario(scenario_name: int,max_time:int):
    """
    For each scenario function implemeted, this function will take into input the scenario name and the cutoff
    and return the list of the scenario shape normalized so that the sum of the weights is 
    equal to 1.

    The scenario function mus be defined and added to the dicitonanry scenario list
    """

    scenario_list = {
        "exponential_scenario" : exponential_scenario,
        "bi_linear_scenario" : bi_linear_scenario,
        "early_peak_scenario" :early_peak_scenario,
        "inverted_u_scenario" : inverted_u_scenario,

    }

    try:
        scenario_function = scenario_list[scenario_name]
    except KeyError:
        print("The scenario ",scenario_name, " is not defined")


    scenario_list = []


    normalization_factor = 0

    for i in range(0,365):
        normalization_factor += scenario_function(i)

    for i in range(0,max_time):
        scenario_list.append(scenario_function(i)) 

    scenario_list = np.array(scenario_list)


    return scenario_list/normalization_factor


        