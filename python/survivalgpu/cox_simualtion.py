import numpy as np
import random
# import pandas as pd
# import torch
# import torchvision.models as models
# from torch.profiler import profile, record_function, ProfilerActivity
# from pathlib import Path 
# from scipy.stats import norm


# from .coxph import coxph_torch


# import time


import numpy as np




class Covariate:
    def __init__(self, name, values):
        self.name = name
        self.values = values
        

class ConstantCovariate(Covariate):
    def __init__(self, name, values,weights):
        super().__init__(name, values)
        self.weights = weights

    def generate_Xmat(self, observation_time, n_patients):
        """
        Generate the Xmat of the constant covariates
        """
        proba = self.weights / np.sum(self.weights)
        Xvect = np.random.choice(self.values, size = n_patients, p = proba)
        Xmat = np.repeat(Xvect, observation_time).reshape(n_patients,observation_time).transpose()


        return Xmat

class TimeDependentCovariate(Covariate):
    def __init__(self, name, values):
        super().__init__(name, values)
    
    def generate_Xmat(self, observation_time, n_patients):
        """
        Generate the Xmat of TDHist for each individual patient
        """
        Xmat = np.array([TDhist(observation_time,self.values) for i in range(n_patients)]).transpose()
        
        return Xmat
    

    
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

def generate_Xmat(covariates :list[Covariate],observation_time,n_patients):

    Xmat = np.zeros((observation_time,n_patients*len(covariates)))



    covariate_matrix_list = [covariate.generate_Xmat(observation_time, n_patients) for covariate in covariates]
    print(covariate_matrix_list)



    for patient_number in range(n_patients):
        for covariate_number in range(len(covariates)):
            Xmat[:,patient_number*len(covariates)  + covariate_number] = covariate_matrix_list[covariate_number][:,patient_number]
    
       
    return Xmat



covariates = [ConstantCovariate("Age", [0,1], [1,3]),
                            ConstantCovariate("Sex", [0,1], [1,1]),
                            TimeDependentCovariate("Variable1", [1, 1.5, 2, 2.5, 3]),
                            TimeDependentCovariate("Variable2", [1, 2, 3, 4, 5])]

Xmat = generate_Xmat(covariates, 10, 3)

print(Xmat)
    


def constant_covariate_Xmat(covariate : ConstantCovariate,
                            n_patients: int):
    """
    Generate the Xmat of the constant covariates
    """
    Xmat = np.zeros(n_patients)

    weights = covariate.weights
    proba = weights / np.sum(weights)


    Xmat = np.random.choice(covariate.values, size = n_patients, p = proba)

    return Xmat




# TODO : here should have a way to use more things for TDhist
# cannot test
def generate_Xmat(observation_time,n_patients,doses):
    """
    Generate the Xmat of TDHist for each indivudual patient
    """
    Xmat = np.matrix([TDhist(observation_time,doses) for i in range(n_patients)]).transpose()
    return Xmat

# can test
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



def event_FUP_Ti_generation(eventRandom, censorRandom):
    event = np.array([1 if eventRandom[i]<= censorRandom[i] else 0 for i in range(len(eventRandom))]).astype(int)
    FUP_Ti = np.minimum(eventRandom,censorRandom).astype(int)

    sorted_indices = np.argsort(FUP_Ti)
    event = event[sorted_indices]
    FUP_Ti = FUP_Ti[sorted_indices]

    return event, FUP_Ti


def event_censor_generation(max_time, n_patients, censoring_ratio):
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

    return eventRandom, censorRandom




def matching_algo(Xmat, max_time:int, n_patients:int, betas,events, FUP_tis):
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
            wce_mat_current = wce_mat_torch[:,ids_torch]

            probas = get_probas(wce_mat_current, HR_target, time_event)
            id_index = torch.multinomial(input = probas, num_samples= 1)

   
            wce_id = ids_torch[id_index]
            ids_torch = ids_torch[ids_torch != wce_id]



        wce_id_indexes[i] = wce_id

            

    wce_id_indexes = np.array(wce_id_indexes.to("cpu"))
    


    return wce_id_indexes

def get_probas(wce_mat_current, HR_target, time_event):

    exp_vals = torch.exp(np.log(HR_target) * wce_mat_current[time_event - 1,]) 
    exp_sum = torch.sum(exp_vals)
    probas = exp_vals/exp_sum  
    
    return probas



def get_dataset(Xmat, max_time, n_patients, HR_target, FUP_tis, events, wce_id_indexes):
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

    Xmat_transposed = Xmat.transpose()

    number_lines = ordered_FUP_tis.sum()

    patient_id_array = np.zeros(number_lines, dtype=int)
    event_array = np.zeros(number_lines, dtype=int)
    time_start_array = np.zeros(number_lines, dtype=int)
    time_stop_array = np.zeros(number_lines, dtype=int)
    doses_aray = np.zeros(number_lines, dtype=np.float64)

    i = 0

    dataset_start = time.perf_counter()

    for patient_id in range(n_patients):
        for time_start in range(ordered_FUP_tis[patient_id]):
            patient_id_array[i] = patient_id + 1
            time_start_array[i] = time_start
            time_stop_array[i] = time_start + 1
            doses_aray[i] = Xmat_transposed[wce_id_indexes[patient_id], time_start]

            if time_start == ordered_FUP_tis[patient_id] - 1:
                event_array[i] = ordered_events[patient_id]
            else:
                event_array[i] = 0
            i += 1

    dataset_end = time.perf_counter()
    elapsed_dataset_time = dataset_end - dataset_start

    df_wce = pd.DataFrame()
    df_wce["patients"] = patient_id_array
    df_wce["start"] = time_start_array
    df_wce["stop"] = time_stop_array
    df_wce["events"] = event_array
    df_wce["doses"] = doses_aray

    return df_wce


def save_dataframe(numpy_wce, n_patients,HR_target, scenario):

    df_wce = pd.DataFrame(numpy_wce, columns = ["patients","start","stop","events","doses"])
    print( str(HR_target))
    saving_path = Path("../../simulated_datasets") / scenario / str(HR_target) / str(n_patients) / "dataset.csv"
    df_wce.to_csv(saving_path)

def simulate_dataset(max_time, n_patients, doses, scenario, betas):


    max_time = int(max_time)
    n_patients = int(n_patients)

    Xmat = generate_Xmat(max_time,n_patients,doses)
    df_wce_mat = pd.DataFrame(wce_mat)

    df_Xmat = pd.DataFrame(Xmat)

    events, FUP_tis = event_censor_generation(max_time, n_patients, censoring_ratio=0.5)
    cox_id_indexes  = matching_algo(Xmat, max_time,n_patients, betas,events, FUP_tis)
    numpy_wce = get_dataset(Xmat, max_time,n_patients, HR_target, FUP_tis,events,wce_id_indexes)
    df_wce = pd.DataFrame(numpy_wce, columns = ["patient","start","stop","event","dose"])

    return df_wce





def simulate_dataset_coxph(Xmat, betas):
    """
    This version of the permutation algorithm generate a dataset 
    """

    Max_time = Xmat.shape[0]
    n_patients = Xmat.shape[1]/len(betas)

    events, FUP_tis = event_censor_generation(max_time, n_patients, censoring_ratio=0.5)

    



    


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


def get_scenario(scenario_name: int,max_time:int):
    """
    For each scenario function implemented, this function will take into input the scenario name and the cutoff
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
        raise ValueError(f"The scenario '{scenario_name}' is not defined")


    scenario_list = []


    normalization_factor = 0

    for i in range(0,365):
        normalization_factor += scenario_function(i)

    for i in range(0,max_time):
        scenario_list.append(scenario_function(i)) 

    scenario_list = np.array(scenario_list)


    return scenario_list/normalization_factor


        