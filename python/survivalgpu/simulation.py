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
    Generate prescirption of different duration and doses 
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


# TODO : here should have a way to use more things for TDhist
# cannot test
def generate_Xmat(max_time,n_patients,doses):
    """
    Generate the Xmat of TDHist for each indivudual patient
    """
    Xmat = np.matrix([TDhist(max_time,doses) for i in range(n_patients)]).transpose()
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



def generate_wce_mat(scenario_name, Xmat):
    """
    This function generate the wce matrix that keep the WCE weight of all the patient at
    all the times until the cutoff
    """

    max_time = Xmat.shape[0]
    scenario_shape = get_scenario(scenario_name,365)

    wce_mat = np.vstack([generate_wce_vector(u, scenario_shape, Xmat) for u in range(1,max_time+1)])


    return wce_mat



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

     # Event times : Uniform[1;365] for all scenarios
    eventRandom = np.round(np.random.uniform(low = 1, high = max_time, size = n_patients)).astype(int)
    # print(eventRandom)
    

    # TODO Maybe change the way the censoring times are determined to that there is no randolness on the number of 
    # TODO patients that are not censored
    censorRandom = np.round(np.random.uniform(low = 1, high = max_time* int(1/censoring_ratio), size = n_patients)).astype(int)

    return eventRandom, censorRandom


class Covariate:
    def __init__(self, name):
        self.name = name
        
        

class ConstantCovariate(Covariate):
    def __init__(self, name, values,weights, beta):
        super().__init__(name)
        self.values = values
        self.weights = weights
        self.beta = beta

    def generate_Xmat(self, max_time, n_patients):
        """
        Generate the Xmat of the constant covariates
        """

        proba = self.weights / np.sum(self.weights)
        Xvect = np.random.choice(self.values, size = n_patients, p = proba)
        Xmat = np.repeat(Xvect, max_time).reshape(n_patients,max_time).transpose()
        self.Xmat = Xmat

        return self

class TimeDependentCovariate(Covariate):
    def __init__(self, name, values, beta):
        super().__init__(name)
        self.values = values
        self.beta = beta
    
    def generate_Xmat(self, max_time, n_patients):

        Xmat = np.array([TDhist(max_time,self.values) for i in range(n_patients)]).transpose()
        self.Xmat = Xmat
        
        return self
    

class WCECovariate(Covariate):
    def __init__(self, name, doses, scenario_name, HR_target):
        self.name = name
        self.doses = doses
        self.scenario_name = scenario_name
        self.HR_target = HR_target 

    def generate_Xmat(self, max_time, n_patients):
        """
        Generate the Xmat of TDHist for each individual patient
        """

        Xmat = np.array([TDhist(max_time,self.doses) for i in range(n_patients)]).transpose()
        self.Xmat = Xmat
        return self 

    
    def generate_wce_mat(self):
        """
        This function generate the wce matrix that keep the WCE weight of all the patient at
        all the times until the cutoff
        """

        try: 
            Xmat = self.Xmat
        except AttributeError:
            raise ValueError("The Xmat has not been generated yet")
        
        scenario_shape = get_scenario(self.scenario_name,365)

        max_time = Xmat.shape[0]


        wce_mat = np.vstack([generate_wce_vector(u, scenario_shape, Xmat) for u in range(1,max_time+1)])

        self.wce_mat = wce_mat

        return self
    



def global_Xmat_wce_mat(list_wce_covariates:list[WCECovariate], 
                        list_cox_covariates:list[(TimeDependentCovariate, ConstantCovariate)],
                        max_time, n_patients):
    


    n_wce_covariates = len(list_wce_covariates)
    n_cox_covariates = len(list_cox_covariates)
    n_covariates = n_wce_covariates + n_cox_covariates



    wce_matrix = np.array(np.zeros(( max_time + 1,n_patients * n_covariates)))

    Xmat_matrix = np.array(np.zeros(( max_time + 1,n_patients * n_covariates)))


    for i in range(n_patients):
        wce_matrix[0,i*n_covariates:i*n_covariates+n_covariates] = i
        Xmat_matrix[0,i*n_covariates:i*n_covariates+n_covariates] = i

    for covariate_number in range(n_wce_covariates):

        for patient_number in range(n_patients):
            
            column_index = patient_number * n_covariates + covariate_number



            wce_matrix[1:, column_index] = np.array(list_wce_covariates[covariate_number].wce_mat[:,patient_number])


            Xmat_matrix[1:, column_index] = np.array(list_wce_covariates[covariate_number].Xmat[:,patient_number])


    for covariate_number in range(n_cox_covariates):

        for patient_number in range(n_patients):
            column_index = patient_number * n_covariates + n_wce_covariates + covariate_number
            wce_matrix[1:, column_index] = np.array(list_cox_covariates[covariate_number].Xmat[:,patient_number])
            Xmat_matrix[1:, column_index] = np.array(list_cox_covariates[covariate_number].Xmat[:,patient_number])


    

    return Xmat_matrix, wce_matrix


def get_probas(wce_matrix_torch, HR_target_list, time_event):
    """
    This function is used to get the probability of selection of each patient
    """


    

    n_covariates = len(HR_target_list)
    current_n_patients = wce_matrix_torch.shape[1]/n_covariates

    # non_selected_indices = wce_matrix_torch[0,torch.arange(0, n_patients*n_covariates, n_covariates).to(device)]


    HR_target_vector = torch.tensor(np.repeat(HR_target_list,current_n_patients)).to(device)

    partial_proba_list = wce_matrix_torch[time_event] * torch.log(HR_target_vector)

    proba_list = partial_proba_list.view(-1,n_covariates)
    proba_list = proba_list.sum(dim = 1)



    exp_vals = torch.exp(proba_list) 

    exp_sum = torch.sum(exp_vals)
    probas = exp_vals/exp_sum  
    


    return probas
    

    
def matching_algo(wce_mat: np.ndarray,
                  n_wce_covariates:int, 
                  n_cox_covariates:int,
                  HR_target_list:list[float],
                  max_time:int,
                  n_patients:int,
                  events: list[int],
                  FUP_tis: list[int]):
    
    

    events = events.copy()
    FUP_tis = FUP_tis.copy()
    events = np.array(events, dtype = int)
    FUP_tis = np.array(FUP_tis, dtype = int)




    covariates_id_indexes = torch.arange(0,n_patients, dtype = int).to(device)

    coef_mat = torch.zeros(len(covariates_id_indexes),dtype = int).to(device)

    n_covariates = n_wce_covariates + n_cox_covariates


    wce_mat_current = torch.from_numpy(wce_mat).to(device)




    # non_selected_indices = torch.arange(0,n_patients, dtype = int).to(device)
    selected_indices = torch.zeros(n_patients,dtype = int).to(device)

    non_selected_indices = wce_mat_current[0,torch.arange(0, n_patients*n_covariates, n_covariates).to(device)]

    
       
    for i in range(n_patients):   

    
        event = events[i]
        time_event = FUP_tis[i]

        

        if event == 0:

            id_index = torch.randint(0,len(non_selected_indices),(1,))
            wce_id = non_selected_indices[id_index]
            wce_mat_current = torch.cat((wce_mat_current[:,:id_index*n_covariates], wce_mat_current[:,(id_index+1)*n_covariates:]), dim = 1)
            # HR_target_list = torch.cat(HR_target_list[:id_index], HR_target_list[id_index+n_covariates:])
 

            non_selected_indices = non_selected_indices[non_selected_indices != wce_id]


        else:

   
            probas = get_probas(wce_mat_current, HR_target_list, time_event)
            id_index = torch.multinomial(input = probas, num_samples= 1)
            wce_id = non_selected_indices[id_index]
            wce_mat_current = torch.cat((wce_mat_current[:,:id_index*n_covariates], wce_mat_current[:,(id_index+1)*n_covariates:]), dim = 1)

            non_selected_indices = non_selected_indices[non_selected_indices != wce_id]
          



        selected_indices[i] = wce_id




            

    selected_indices = np.array(selected_indices.to("cpu"))
    


    return selected_indices







def get_dataset(Xmat,covariate_names, n_patients, FUP_tis, events, wce_id_indexes):
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
    

    for patient_id in range(n_patients):
        for time_start in range(ordered_FUP_tis[patient_id]):
            patient_id_array[i] = patient_id + 1
            time_start_array[i] = time_start
            time_stop_array[i] = time_start + 1
            for covariate_id in range(len(covariate_names)):
      
                covariate_dict[covariate_names[covariate_id]][i] = Xmat[time_start +1, (patient_id)*n_covariates + covariate_id]

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

    # generate the Xmat and wce_mat for the wce covariates
    list_wce_covariates = [covariate.generate_Xmat(max_time = max_time,n_patients=n_patients).generate_wce_mat() for covariate in list_wce_covariates]
    # generate the Xmat for the cox covariates
    list_cox_covariates = [covariate.generate_Xmat(max_time = max_time,n_patients = n_patients) for covariate in list_cox_covariates]


    eventRandom, censorRandom = event_censor_generation(max_time, n_patients, censoring_ratio=0.5)
    events, FUP_tis = event_FUP_Ti_generation(eventRandom, censorRandom)

    Xmat_matrix, wce_mat = global_Xmat_wce_mat(list_wce_covariates, list_cox_covariates, max_time, n_patients)


    HR_target_list = np.zeros(n_covariates)

    i = 0
    for covariate in list_wce_covariates:  
        HR_target_list[i] = covariate.HR_target
        i+=1 
    
    for covariate in list_cox_covariates:   
        HR_target_list[i] = np.exp(covariate.beta)
        i+=1 

    wce_id_selected = matching_algo(wce_mat = wce_mat,
                                    n_cox_covariates=n_cox_covariates,
                                    n_wce_covariates=n_wce_covariates,
                                    HR_target_list=HR_target_list,
                                    max_time=max_time,
                                    n_patients=n_patients,
                                    events=events,
                                    FUP_tis = FUP_tis)

    covariate_names = [covariate.name for covariate in list_wce_covariates] + [covariate.name for covariate in list_cox_covariates]

    
    # wce_id_indexes  = matching_algo(wce_mat = wce_mat, 
    #                                 n_wce_covariates = n_wce_covariates,
    #                                 n_cox_covariates = n_cox_covariates, 
    #                                 max_time = max_time,
    #                                 n_patients = n_patients,
    #                                 events = events, 
    #                                 FUP_tis = FUP_tis)
    


    dataset = get_dataset(Xmat = Xmat_matrix,
                      covariate_names = covariate_names, 
                      n_patients =n_patients, 
                      FUP_tis = FUP_tis, 
                      events = events, 
                      wce_id_indexes = wce_id_selected)
    
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
    - max_time (int): The maximum time for which to generate the scenario list.

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


        