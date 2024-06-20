import numpy as np
import random
import pandas as pd
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
from pathlib import Path 
from scipy.stats import norm


from .coxph import coxph_torch
from .utils import device


import time



# TODO : modify the TDhist to be able to manage a bigger variety of cases, 
# maybe create another  TDhist that is more in tune with the kind of data given by the SNDS
def TDhist(max_time,doses):
    """
    This function is used to generate individual time-dependant exposure history 
    Generate prescription of different duration and doses 
    """

    duration = int(7 + 7*np.round(np.random.lognormal(mean =0.5, sigma =0.8, size = 1)).item())
    # duration is in weeks *

    dose = random.choice(doses)
    exposure_vector = np.repeat(dose,repeats = duration)



    while len(exposure_vector) <= max_time:
        intermission = int(7 + 7*np.round(np.random.lognormal(mean =0.5, sigma =0.8, size = 1)).item())
        duration = int(7 + 7*np.round(np.random.lognormal(mean =0.5, sigma =0.8, size = 1)).item())

        dose = random.choice(doses)

        exposure_vector = np.concatenate((exposure_vector,np.repeat(0,repeats = intermission),np.repeat(dose,repeats = duration)))

    return exposure_vector[:max_time]



# can test






def event_FUP_Ti_generation(eventRandom, censorRandom):
    events = np.array([1 if eventRandom[i]<= censorRandom[i] else 0 for i in range(len(eventRandom))]).astype(int)
    FUP_Ti = np.minimum(eventRandom,censorRandom).astype(int)

    sorted_indices = np.argsort(FUP_Ti)
    events = events[sorted_indices]
    FUP_Ti = FUP_Ti[sorted_indices]

    return events, FUP_Ti


def event_censor_generation(max_time, n_patients, censoring_ratio):
    if censoring_ratio > 1:
        raise ValueError("The censoring ration must be inferior to 1")
    if censoring_ratio < 0:
        raise ValueError("The censoring ration must be positive")

    eventRandom = np.round(np.random.uniform(low = 1, high = max_time, size = n_patients)).astype(int)
    censorRandom = np.round(np.random.uniform(low = 1, high = max_time* int(1/censoring_ratio), size = n_patients)).astype(int)

    return eventRandom, censorRandom


class Covariate:
    def __init__(self, name):
        self.name = name

    def initialize_experiment(self, n_patients, max_time):
        self.n_patients = n_patients
        self.max_time = max_time

        return self
        
        

class ConstantCovariate(Covariate):
    def __init__(self, name, values,weights, beta):
        super().__init__(name)
        self.values = values
        self.weights = weights
        self.beta = beta

    def generate_Xvector(self):
        """
        Generate the Xvector of the constant covariates
        """

        proba = self.weights / np.sum(self.weights)
        
        Xvect = np.random.choice(self.values, size = self.n_patients, p = proba)
        Xvector = np.repeat(Xvect, self.max_time)
        self.Xvector = Xvector

        return self

class TimeDependentCovariate(Covariate):
    def __init__(self, name, values, beta):
        super().__init__(name)
        self.values = values
        self.beta = beta
    
    def generate_Xvector(self):

        Xvector = np.array([TDhist(self.max_time,self.values) for i in range(self.n_patients)],dtype=float).flatten()

        self.Xvector = Xvector
        
        return self
    

class WCECovariate(Covariate):
    def __init__(self, name, values, scenario_name, HR_target):
        self.name = name
        self.values = values
        self.scenario_name = scenario_name
        self.HR_target = HR_target 

    def generate_Xvector(self):
        """
        Generate the Xmat of TDHist for each individual patient
        """

        Xvector = np.array([TDhist(self.max_time,self.values) for i in range(self.n_patients)],dtype=float).flatten()
        self.Xvector = Xvector
        return self 

    
    def generate_WCEvector(self):
        """
        This function generate the wce matrix that keep the WCE weight of all the patient at
        all the times until the cutoff
        """

        try: 
            Xvector = self.Xvector
        except AttributeError:
            raise ValueError("The Xvector has not been generated yet")
        
        n_patients = self.n_patients
        max_time = self.max_time


        

        covariate_Xmat = Xvector.reshape(self.n_patients,self.max_time).transpose()

        scenario_shape = get_scenario(self.scenario_name, self.max_time)


        
        def generate_wce_vector(u, scenario_shape, covariate_Xmat):
            t_array = np.arange(1,u+1)
            u_t_array = u  - t_array 
            wce = np.multiply(scenario_shape[u_t_array].reshape(u,1),covariate_Xmat[t_array -1,:])

            

            return np.sum(wce, axis = 0)
            
        wce_mat = np.vstack([generate_wce_vector(u, scenario_shape, covariate_Xmat) for u in range(1,max_time+1)])


        
        WCEvector = np.zeros((max_time*n_patients))

        for i in range(self.n_patients):
            WCEvector[i*max_time:(i+1)*max_time] = wce_mat[:,i]
    

        self.WCEvector = WCEvector

        return self
    

def generate_Xmat(list_wce_covariates:list[WCECovariate], 
                        list_cox_covariates:list[(TimeDependentCovariate, ConstantCovariate)],
                        max_time, n_patients):
    

    n_wce_covariates = len(list_wce_covariates)
    n_cox_covariates = len(list_cox_covariates)
    n_covariates = n_wce_covariates + n_cox_covariates


    Xmat = np.zeros((max_time*n_patients, n_covariates +1))

    
    Xmat[:,0] = np.repeat(np.arange(n_patients),max_time)

    i = 1

    for covariate in list_wce_covariates:
        Xmat[:,i] = covariate.Xvector
        i += 1

    for covariate in list_cox_covariates:
        Xmat[:,i] = covariate.Xvector
        i+=1
        

    return Xmat

def generate_WCEmat(list_wce_covariates:list[WCECovariate], 
                        list_cox_covariates:list[(TimeDependentCovariate, ConstantCovariate)],
                        max_time, n_patients):
    

    n_wce_covariates = len(list_wce_covariates)
    n_cox_covariates = len(list_cox_covariates)
    n_covariates = n_wce_covariates + n_cox_covariates


    WCEmat = np.zeros((max_time*n_patients, n_covariates +1))

    
    WCEmat[:,0] = np.repeat(np.arange(n_patients),max_time)

    i = 1

    for covariate in list_wce_covariates:
        WCEmat[:,i] = covariate.WCEvector
        i += 1

    for covariate in list_cox_covariates:
        WCEmat[:,i] = covariate.Xvector
        i+=1
        

    return WCEmat


def get_WCEmat_time_event(WCEmat, time_event, max_time):
    """
    This function is used to get the WCE matrix at a given time event
    """

    WCEmat_time_event = WCEmat[(time_event-1)::max_time,:]

    return WCEmat_time_event





def get_probas(WCEmat_time_event, HR_target_list):
    """
    This function is used to get the probability of selection of each patient
    """

    partial_proba_list = WCEmat_time_event[:,1:] * torch.log(HR_target_list)
    exp_vals = torch.exp(partial_proba_list.sum(dim = 1)) 
    exp_sum = torch.sum(exp_vals)
    probas = exp_vals/exp_sum  



    return probas
    

    
def matching_algo(WCEmat: np.ndarray,
                  n_wce_covariates:int, 
                  n_cox_covariates:int,
                  HR_target_list:np.ndarray,
                  max_time:int,
                  n_patients:int,
                  events: list[int],
                  FUP_tis: list[int]):
    
    

    events = events.copy()
    FUP_tis = FUP_tis.copy()
    events = np.array(events, dtype = int)
    FUP_tis = np.array(FUP_tis, dtype = int)




    covariates_id_indexes = torch.arange(0,n_patients, dtype = int).to(device)


    n_covariates = n_wce_covariates + n_cox_covariates


    wce_mat_current = torch.from_numpy(WCEmat).to(device)



    selected_indices = torch.zeros(n_patients,dtype = int).to(device)

    non_selected_indices = torch.arange(0,n_patients)

    WCEmat_current = torch.from_numpy(WCEmat).to(device)
    HR_target_tensor = torch.from_numpy(HR_target_list).to(device)




       
    for i in range(n_patients):   

    
        event = events[i]
        time_event = FUP_tis[i]

        event = 1


        if event == 0:

            id_index = torch.randint(0,len(non_selected_indices),(1,))
            WCEmat_current = torch.cat((WCEmat_current[:id_index*max_time] , WCEmat_current[(id_index+1)*max_time:]))
            wce_id = non_selected_indices[id_index]
            non_selected_indices = non_selected_indices[non_selected_indices != wce_id]





        else:


            WCEmat_time_event = get_WCEmat_time_event(WCEmat_current, time_event, max_time)
            probas = get_probas(WCEmat_time_event, HR_target_tensor)
            id_index = torch.multinomial(input = probas, num_samples= 1)
            wce_id = non_selected_indices[id_index]
            WCEmat_current = torch.cat((WCEmat_current[:id_index*max_time] , WCEmat_current[(id_index+1)*max_time:]))
            non_selected_indices = non_selected_indices[non_selected_indices != wce_id]


        selected_indices[i] = wce_id




            

    selected_indices = np.array(selected_indices.to("cpu"))
    


    return selected_indices







def get_dataset(Xmat,covariate_names, n_patients, FUP_tis, events, wce_id_indexes, max_time):
    """
    Generate a dataset based on the given inputs.

    Args:
        Xmat (numpy.ndarray): The input matrix.
        max_time (int): The maximum time.
        n_patients (int): The number of patients.
        HR_target (float): The target hazard ratio.
        FUP_tis (list): The follow-up times.
        events (list): The events.
        wce_id_indexes (list): The WCE ID indexes.

    Returns:
        pandas.DataFrame: The generated dataset.
    """
    FUP_tis = np.array(FUP_tis, dtype=int)
    events = np.array(events, dtype=int)
    n_patients = int(n_patients)

    wce_id_indexes = np.array(wce_id_indexes, dtype=int)

    ordered_events = events
    ordered_FUP_tis = FUP_tis


    number_lines = ordered_FUP_tis.sum()

    patient_id_array = np.zeros(number_lines, dtype=int)
    fup_id_array = np.zeros(number_lines, dtype=int)
    event_array = np.zeros(number_lines, dtype=int)
    time_start_array = np.zeros(number_lines, dtype=int)
    time_stop_array = np.zeros(number_lines, dtype=int)
    doses_aray = np.zeros(number_lines, dtype=np.float64)

    i = 0

    dataset_start = time.perf_counter()


    covariate_dict = {}
    
    for covariate_name in covariate_names:
        covariate_dict[covariate_name] = np.zeros(number_lines, dtype=np.float64)

    n_covariates = len(covariate_names)


    id_t0 = 0
  

    for patient_id in range(n_patients):

        Fup = ordered_FUP_tis[patient_id]
        patient_id_array[id_t0:id_t0 + Fup] = patient_id + 1
        fup_id_array[id_t0:id_t0 + Fup] = Fup
        time_start_array[id_t0:id_t0 + Fup] = np.arange(Fup)
        time_stop_array[id_t0:id_t0 + Fup] = np.arange(1,Fup+1)
        event_array[id_t0:id_t0 + Fup] = 0
        if ordered_events[patient_id] == 1:
            event_array[id_t0 + Fup - 1] = 1

        Xmat_id = wce_id_indexes[patient_id]

        for covariate_id in range(n_covariates):
            covariate_dict[covariate_names[covariate_id]][id_t0:id_t0 + Fup] = Xmat[Xmat_id*max_time:Xmat_id*max_time + Fup, covariate_id+1]
        id_t0 += Fup

    dataset_end = time.perf_counter()
    elapsed_dataset_time = dataset_end - dataset_start



    df_wce = pd.DataFrame()
    df_wce["patients"] = patient_id_array
    df_wce["fup"] = fup_id_array
    df_wce["start"] = time_start_array
    df_wce["stop"] = time_stop_array
    df_wce["events"] = event_array
    for covariate_name in covariate_names:
        df_wce[covariate_name] = covariate_dict[covariate_name]

    return df_wce


def save_dataframe(numpy_wce, n_patients,HR_target, scenario):

    df_wce = pd.DataFrame(numpy_wce, columns = ["patients","start","stop","events","doses"])
    saving_path = Path("../../simulated_datasets") / scenario / str(HR_target) / str(n_patients) / "dataset.csv"
    df_wce.to_csv(saving_path)



def simulate_dataset(max_time, n_patients, 
                     list_wce_covariates: list[WCECovariate], 
                     list_cox_covariates:list[(TimeDependentCovariate, ConstantCovariate)]):


    max_time = int(max_time)
    n_patients = int(n_patients)

    n_wce_covariates = len(list_wce_covariates)
    n_cox_covariates = len(list_cox_covariates)
    n_covariates = n_wce_covariates + n_cox_covariates


    for covariate in list_wce_covariates:
        covariate = covariate.initialize_experiment(max_time = max_time,n_patients = n_patients).generate_Xvector().generate_WCEvector()
    for covariate in list_cox_covariates:
        covariate = covariate.initialize_experiment(max_time = max_time,n_patients = n_patients).generate_Xvector()


    eventRandom, censorRandom = event_censor_generation(max_time, n_patients, censoring_ratio=0.5)
    events, FUP_tis = event_FUP_Ti_generation(eventRandom, censorRandom)

    Xmat = generate_Xmat(list_wce_covariates, 
                         list_cox_covariates, 
                         max_time = max_time,
                         n_patients = n_patients)
    

    WCEmat = generate_WCEmat(list_wce_covariates, 
                             list_cox_covariates, 
                             max_time = max_time,
                             n_patients = n_patients)
    



    HR_target_list = np.zeros(n_covariates)

    i = 0
    for covariate in list_wce_covariates:  
        HR_target_list[i] = covariate.HR_target
        i+=1 
    
    for covariate in list_cox_covariates:   
        HR_target_list[i] = np.exp(covariate.beta)
        i+=1 



    wce_id_selected = matching_algo(WCEmat = WCEmat,
                                    n_cox_covariates=n_cox_covariates,
                                    n_wce_covariates=n_wce_covariates,
                                    HR_target_list=HR_target_list,
                                    max_time=max_time,
                                    n_patients=n_patients,
                                    events=events,
                                    FUP_tis = FUP_tis)
    
    


    covariate_names = [covariate.name for covariate in list_wce_covariates] + [covariate.name for covariate in list_cox_covariates]

    
    # wce_id_indexes  = matching_algo(WCEmat = WCEmat, 
    #                                 n_wce_covariates = n_wce_covariates,
    #                                 n_cox_covariates = n_cox_covariates, 
    #                                 max_time = max_time,
    #                                 n_patients = n_patients,
    #                                 events = events, 
    #                                 FUP_tis = FUP_tis)
    


    dataset = get_dataset(Xmat = Xmat,
                      covariate_names = covariate_names, 
                      n_patients =n_patients, 
                      FUP_tis = FUP_tis, 
                      events = events, 
                      wce_id_indexes = wce_id_selected,
                      max_time =max_time)
    
    # df_wce = pd.DataFrame(numpy_wce, columns = ["patients","start","stop","events","doses"])

    
    return dataset





def simulate_dataset_coxph(Xmat, scenario, betas):
    """
    This version of the permutation algorithm generate a dataset 
    """




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
    return norm.pdf(u_t/365, 0.2, 0.06)


def get_scenario(scenario_name: int, max_time: int):
    """
    Get the scenario list based on the given scenario name and maximum time.

    Parameters:
    - scenario_name (int): The name of the scenario to retrieve.
    - max_time (int): The maximum time for which to clear the scenario list.
    Returns:
    - scenario_list (numpy.ndarray): The generated scenario list.

    Raises:
    - ValueError: If the given scenario name is not defined in the scenario list.

    """

    scenario_list = {
        "exponential_scenario": exponential_scenario,
        "bi_linear_scenario": bi_linear_scenario,
        "early_peak_scenario": early_peak_scenario,
        "inverted_u_scenario": inverted_u_scenario,
    }

    try:
        scenario_function = scenario_list[scenario_name]
    except KeyError:
        raise ValueError(f"The scenario '{scenario_name}' is not defined")

    scenario_list = []
    normalization_factor = 0

    for i in range(0, 365):
        normalization_factor += scenario_function(i)

    for i in range(0, max_time):
        scenario_list.append(scenario_function(i))

    scenario_list = np.array(scenario_list)

    return scenario_list / normalization_factor


        