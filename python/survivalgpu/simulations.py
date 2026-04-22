from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from .utils import default_device

# Generation data


# def cox_permalgo(
#         num_subjects
# ):


# def wce_permalgo(
#         n_patients,
#         max_time,
#         events,
#         final_times,
#         cox_Xmat,
#         cox_betas,
#         cox_names=None,
#         WCE_Xmat=None,
#         WCE_betas=None,
#         WCE_scenarios=None,
#         WCE_names=None):


#     X_column_length = n_patients * max_time

#     # get if this is a dataset containing WCE covariates
#     if WCE_Xmat is not None:
#         is_WCE = True


#     else:
#         is_WCE = False

#     #check the length of the Xmat
#     if cox_Xmat.shape[0] != X_column_length:
#         msg = f"The cox_Xmat has a wrong number of rows, expected {X_column_length}, got {cox_Xmat.shape[0]}"
#         raise ValueError(msg)



# def permalgo(
#         n_patients,
#         max_time,
#         events,
#         final_times,
#         Xmat,
#         betas,
#         X_names=None):
#     """
#     Docstring for cex_permalgo

#     :param n_patients: the number of patients
#     :param max_time: the maximum follow-up time
#     :param events: the event indicators
#     :param final_times: the final follow-up times
#     :param Xmat: the covariate matrix, with shape (n_patients * max_time, n_covariates)
#     :param betas: the regression coefficients

#     :return: a dataframe containing the simulated dataset with the iindependant
#         covariates and the event indicators associated with a probability density
#         function corresponding to the likelihood of the covariates and betas
#     """

#     X_column_length = n_patients * max_time

#     #check the length of the Xmat
#     if Xmat.shape[0] != X_column_length:
#         msg = f"The Xmat has a wrong number of rows, expected {X_column_length}, got {Xmat.shape[0]}"
#         raise ValueError(msg)

#     # check that the number of betas is equal to the number of covariates, defined by the number of columns in the Xmat
#     if Xmat.shape[1] != len(betas):
#         msg = f"The number of betas is not equal to the number of covariates, expected {Xmat.shape[1]}, got {len(betas)}"
#         raise ValueError(msg)

#     # check the

#     # check that the length of events and final_times is equal to n_patients
#     if len(events) != n_patients:
#         msg = f"The length of events is not equal to the number of patients, expected {n_patients}, got {len(events)}"
#         raise ValueError(msg)

#     # check that the number of names is equal to the number of covariates, if names are given
#     if X_names is not None:
#         if len(X_names) != Xmat.shape[1]:
#             msg = f"The number of names is not equal to the number of covariates, expected {Xmat.shape[1]}, got {len(X_names)}"
#             raise ValueError(msg)


#     # convert betas to numpy array
#     betas = np.asarray(betas)


#     wce_id_selected = matching_algo(Xmat = Xmat,
#                                     HR_target_list=HR_target_list,
#                                     max_time=max_time,
#                                     n_patients=n_patients,
#                                     events=events,
#                                     FUP_tis = FUP_tis)



#     cox_betas = np.asarray(cox_betas)
#     if WCE_betas is not None:
#         WCE_betas = np.asarray(WCE_betas)
#         beta_list = np.concatenate([cox_betas, WCE_betas])
#     else:
#         beta_list = cox_betas

#     return beta_list


def TDhist(max_time, doses, rng):
    """
    This function is used to generate individual time-dependant exposure history
    Generate prescription of different duration and doses
    """

    duration = int(7 + 7 * np.round(rng.lognormal(mean=0.5, sigma=0.8, size=1)).item())
    # duration is in weeks *

    dose = rng.choice(doses)
    exposure_vector = np.repeat(dose, repeats=duration)

    while len(exposure_vector) <= max_time:
        intermission = int(7 + 7 * np.round(rng.lognormal(mean=0.5, sigma=0.8, size=1)).item())
        duration = int(7 + 7 * np.round(rng.lognormal(mean=0.5, sigma=0.8, size=1)).item())
        exposure_vector = np.concatenate(
            (
                exposure_vector,
                np.repeat(0, repeats=intermission),
                np.repeat(dose, repeats=duration),
            )
        )
    return exposure_vector[:max_time]



def event_FUP_Ti_generation(eventRandom, censorRandom):
    events = np.array([1 if eventRandom[i]<= censorRandom[i] else 0 for i in range(len(eventRandom))]).astype(int)
    FUP_Ti = np.minimum(eventRandom,censorRandom).astype(int)

    sorted_indices = np.argsort(FUP_Ti)
    events = events[sorted_indices]
    FUP_Ti = FUP_Ti[sorted_indices]

    return events, FUP_Ti


def event_censor_generation(max_time, n_patients, censoring_ratio, rng):
    if censoring_ratio > 1:
        msg = "The censoring ratio must be inferior to 1"
        raise ValueError(msg)

    if censoring_ratio < 0:
        msg = "The censoring ratio must be positive"
        raise ValueError(msg)

    eventRandom = np.round(
        rng.uniform(low=1, high=max_time, size=n_patients)
    ).astype(int)

    censorRandom = np.round(
        rng.uniform(low=1, high=max_time * int(1 / censoring_ratio), size=n_patients)
    ).astype(int)

    return eventRandom, censorRandom


class Covariate:
    def __init__(self, name):
        self.name = name


class ConstantCovariate(Covariate):
    """
    This class is used to define a constant cox covariate, it's a covariate that impacts
    the values taken by the patients at each time point are constant
    It impact the log-likelihood like in the cox model

    Attributes :
    - name : the name of the covariate
    - values : the possible values of the covariate
    - weights : the weights of each value, for each patient, the value is chosen with a probability proportional to the weights
    - coef : the coefficient of the covariate in the cox model (the coefficient is equal to the log of the hazard ratio of the covariate
    for a value of 1 compared to a value of 0)

    """
    def __init__(self, name, values,weights, coef):
        super().__init__(name)
        self.values = values
        self.weights = weights
        self.coef = coef


    def initialize_experiment(self, n_patients, max_time, rng):
        self.n_patients = n_patients
        self.max_time = max_time
        self.generate_Xvector(rng=rng)
        return self

    def generate_Xvector(self, rng):

        proba = self.weights / np.sum(self.weights)

        Xvect = rng.choice(self.values, size=self.n_patients, p=proba)
        Xvector = np.repeat(Xvect, self.max_time)

        self.Xvector = Xvector
        return self


class TimeDependentCovariate(Covariate):
    """
    This class is used to define a time-dependent cox covariate, it's a covariate that impact
    The values are time-dependant and can be cumulative or not. If they are cumulative the values are summed over a
    period determined by the cutoff
    It impact the log-likelihood like in the cox model

    Attributes :
    - name : the name of the covariate
    - values : the possible values of the covariate
    - coef : the coefficient of the covariate in the cox model (the coefficient is equal to the log of the hazard ratio of the covariate
    for a value of 1 compared to a value of 0)
    - cumulative : a boolean that indicates if the values are cumulative or not
    - cutoff : the cutoff period for the cumulative values, if a cumulative covariate is not given a cutoff, the cutoff is set to the max_time

    """
    def __init__(self, name, values, coef, cumulative = False, cutoff = None):
        super().__init__(name)
        self.values = values
        self.coef = coef
        self.cumulative = cumulative
        self.cutoff = cutoff

    def initialize_experiment(self, n_patients, max_time, rng):
        self.n_patients = n_patients
        self.max_time = max_time
        self.generate_Xvector(rng=rng)

        if self.cumulative:
            if self.cutoff is None:
                self.cutoff = self.max_time
            self.cumulate_exposure(cutoff=self.cutoff)

        return self

    def generate_Xvector(self, rng):

        Xvector = np.array([TDhist(self.max_time, self.values, rng) for i in range(self.n_patients)],dtype=float).flatten()
        self.Xvector = Xvector

        return self

    def cumulate_exposure(self, cutoff):

        try:
            Xvector = self.Xvector
        except AttributeError as err:
            msg = "The Xvector has not been generated yet"
            raise ValueError(msg) from err



        Xmat = Xvector.reshape(self.n_patients,self.max_time).transpose()



        cumulative_Xmat = np.zeros((self.max_time,self.n_patients))


        for j in range(self.n_patients):
            vector = Xmat[:,j]
            for i in range(self.max_time):
                sum = np.sum(vector[max(0,i-cutoff):i+1])
                cumulative_Xmat[i,j] = sum


        self.Xvector = cumulative_Xmat.transpose().flatten()


        return self

class CoxCovariate(Covariate):
    """
    This class is used to define a time-dependent cox covariate, it's a covariate that impact
    The values are time-dependant and can be cumulative or not. If they are cumulative the values are summed over a
    period determined by the cutoff
    It impact the log-likelihood like in the cox model

    Attributes :
    - name : the name of the covariate
    - values : the possible values of the covariate
    - coef : the coefficient of the covariate in the cox model (the coefficient is equal to the log of the hazard ratio of the covariate
    for a value of 1 compared to a value of 0)
    - cumulative : a boolean that indicates if the values are cumulative or not
    - cutoff : the cutoff period for the cumulative values, if a cumulative covariate is not given a cutoff, the cutoff is set to the max_time

    """
    def __init__(self, name, Xvector, coef):
        super().__init__(name)
        self.coef = coef
        self.Xvector = Xvector

    def initialize_experiment(self, n_patients, max_time):
        self.n_patients = n_patients
        self.max_time = max_time

        return self



    # def generate_Xvector(self):

    #     Xvector = np.array([TDhist(self.max_time,self.values) for i in range(self.n_patients)],dtype=float).flatten()

    #     self.Xvector = Xvector

    #     return self

    # def cumulate_exposure(self,cutoff):

    #     try:
    #         Xvector = self.Xvector
    #     except AttributeError:
    #         raise ValueError("The Xvector has not been generated yet")


    #     Xmat = Xvector.reshape(self.n_patients,self.max_time).transpose()



    #     cumulative_Xmat = np.zeros((self.max_time,self.n_patients))


    #     for j in range(self.n_patients):
    #         vector = Xmat[:,j]
    #         for i in range(self.max_time):
    #             sum = np.sum(vector[max(0,i-cutoff):i+1])
    #             cumulative_Xmat[i,j] = sum


    #     self.Xvector = cumulative_Xmat.transpose().flatten()


    #     return self

class WCECovariate_new(Covariate):
    """
    This class is used to define a WCE covariate, it's a covariate that impact the log likelihood following a time-dependent pattern
    of cumulative exposure. The WCE covariate is defined by a scenario that is used to generate the weight of the covariate at each time point

    Attributes :
    - name : the name of the covariate
    - values : the possible values of the covariate
    - scenario_name : the name of the scenario used to generate the weight of the covariate
    - HR_target : the target hazard ratio of the covariate given a value of 1 for a time equal to the cutoff compared to a value of 0 for the time of the cutoff
    """
    def __init__(self, name, Xvector, scenario_name, HR_target):
        self.name = name
        self.scenario_name = scenario_name
        self.HR_target = HR_target
        self.Xvector = Xvector


    def initialize_experiment(self, n_patients, max_time):
        self.n_patients = n_patients
        self.max_time = max_time
        # self.generate_Xvector()
        self.generate_WCEvector()

        return self

    def generate_Xvector(self, rng):
        """
        Generate the Xmat of TDHist for each individual patient
        """
        Xvector = np.array([TDhist(self.max_time, self.values, rng) for i in range(self.n_patients)],dtype=float).flatten()
        self.Xvector = Xvector
        return self


    def generate_WCEvector(self):
        """
        This function generates the WCE matrix that keeps the WCE weight of all
        patients at all times until the cutoff.
        """

        try:
            Xvector = self.Xvector
        except AttributeError as err:
            msg = "The Xvector has not been generated yet"
            raise ValueError(msg) from err

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



        WCEvector = np.zeros(max_time*n_patients)

        for i in range(self.n_patients):
            WCEvector[i*max_time:(i+1)*max_time] = wce_mat[:,i]


        self.WCEvector = WCEvector

        return self


class WCECovariate(Covariate):
    """
    This class is used to define a WCE covariate, it's a covariate that impact the log likelihood following a time-dependent pattern
    of cumulative exposure. The WCE covariate is defined by a scenario that is used to generate the weight of the covariate at each time point

    Attributes :
    - name : the name of the covariate
    - values : the possible values of the covariate
    - scenario_name : the name of the scenario used to generate the weight of the covariate
    - HR_target : the target hazard ratio of the covariate given a value of 1 for a time equal to the cutoff compared to a value of 0 for the time of the cutoff
    """
    def __init__(self, name, values, scenario_name, HR_target):
        self.name = name
        self.values = values
        self.scenario_name = scenario_name
        self.HR_target = HR_target


    def initialize_experiment(self, n_patients, max_time, rng):
        self.n_patients = n_patients
        self.max_time = max_time
        self.generate_Xvector(rng=rng)
        self.generate_WCEvector()

        return self

    def generate_Xvector(self, rng):
        """
        Generate the Xmat of TDHist for each individual patient
        """
        Xvector = np.array([TDhist(self.max_time, self.values, rng) for i in range(self.n_patients)],dtype=float).flatten()
        self.Xvector = Xvector
        return self


    def generate_WCEvector(self):
        """
        This function generates the WCE matrix that keeps the WCE weight of all
        patients at all times until the cutoff.
        """

        try:
            Xvector = self.Xvector
        except AttributeError as err:
            msg = "The Xvector has not been generated yet"
            raise ValueError(msg) from err

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



        WCEvector = np.zeros(max_time*n_patients)

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

    return WCEmat[(time_event-1)::max_time,:]





def get_probas(WCEmat_time_event, HR_target_list):
    """
    This function is used to get the probability of selection of each patient
    """

    partial_proba_list = WCEmat_time_event[:,1:] * torch.log(HR_target_list)
    exp_vals = torch.exp(partial_proba_list.sum(dim = 1))
    exp_sum = torch.sum(exp_vals)



    return exp_vals/exp_sum



def matching_algo(WCEmat: np.ndarray,
                #   n_wce_covariates:int,
                #   n_cox_covariates:int,
                  HR_target_list:np.ndarray,
                  max_time:int,
                  n_patients:int,
                  events: list[int],
                  FUP_tis: list[int],
                  torch_generator: torch.Generator | None = None):



    events = events.copy()
    FUP_tis = FUP_tis.copy()
    events = np.array(events, dtype = int)
    FUP_tis = np.array(FUP_tis, dtype = int)





    selected_indices = torch.zeros(n_patients,dtype = int).to(default_device)

    non_selected_indices = torch.arange(0,n_patients).to(default_device)

    WCEmat_current = torch.from_numpy(WCEmat).to(default_device)
    HR_target_tensor = torch.from_numpy(HR_target_list).to(default_device)





    for i in range(n_patients):


        event = events[i]
        time_event = FUP_tis[i]

        event = 1


        if event == 0:

            id_index = torch.randint(
                0,
                len(non_selected_indices),
                (1,),
                device=non_selected_indices.device,
                generator=torch_generator,
            )
            WCEmat_current = torch.cat((WCEmat_current[:id_index*max_time] , WCEmat_current[(id_index+1)*max_time:]))
            wce_id = non_selected_indices[id_index]
            non_selected_indices = non_selected_indices[non_selected_indices != wce_id]





        else:


            WCEmat_time_event = get_WCEmat_time_event(WCEmat_current, time_event, max_time)
            probas = get_probas(WCEmat_time_event, HR_target_tensor)
            id_index = torch.multinomial(input = probas, num_samples= 1, generator=torch_generator)
            wce_id = non_selected_indices[id_index]
            WCEmat_current = torch.cat((WCEmat_current[:id_index*max_time] , WCEmat_current[(id_index+1)*max_time:]))
            non_selected_indices = non_selected_indices[non_selected_indices != wce_id]


        selected_indices[i] = wce_id


    return np.array(selected_indices.to("cpu"))


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

    # dataset_start = time.perf_counter()


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

    # dataset_end = time.perf_counter()
    # elapsed_dataset_time = dataset_end - dataset_start



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
                     list_covariates: list[WCECovariate, TimeDependentCovariate, ConstantCovariate],
                     compress = False,
                     seed: int | None = None):
    rng = np.random.default_rng(seed)
    torch_generator = None
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch_generator = torch.Generator(device=default_device)
        torch_generator.manual_seed(seed)



    list_wce_covariates = []
    list_cox_covariates = []

    if list_covariates is None:
        msg = "The list of covariates is None"
        raise ValueError(msg)

    for covariate in list_covariates:

        if type(covariate) is WCECovariate:
            list_wce_covariates.append(covariate)
        elif type(covariate) in [TimeDependentCovariate, ConstantCovariate]:
            list_cox_covariates.append(covariate)
        else:
            msg = "The covariate is not recognized as a WCE or a Cox covariate"
            raise ValueError(msg)



    max_time = int(max_time)
    n_patients = int(n_patients)

    n_wce_covariates = len(list_wce_covariates)
    n_cox_covariates = len(list_cox_covariates)
    n_covariates = n_wce_covariates + n_cox_covariates


    for covariate in list_wce_covariates:
        covariate.initialize_experiment(max_time=max_time, n_patients=n_patients, rng=rng)
    for covariate in list_cox_covariates:
        covariate.initialize_experiment(max_time=max_time, n_patients=n_patients, rng=rng)

    eventRandom, censorRandom = event_censor_generation(max_time, n_patients, censoring_ratio=0.5, rng=rng)
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
        HR_target_list[i] = np.exp(covariate.coef)
        i+=1



    wce_id_selected = matching_algo(WCEmat = WCEmat,
                                    # n_cox_covariates=n_cox_covariates,
                                    # n_wce_covariates=n_wce_covariates,
                                    HR_target_list=HR_target_list,
                                    max_time=max_time,
                                    n_patients=n_patients,
                                    events=events,
                                    FUP_tis = FUP_tis,
                                    torch_generator=torch_generator)




    covariate_names = [covariate.name for covariate in list_wce_covariates] + [covariate.name for covariate in list_cox_covariates]


    # df_wce = pd.DataFrame(numpy_wce, columns = ["patients","start","stop","events","doses"])

    dataset = get_dataset(Xmat = Xmat,
                      covariate_names = covariate_names,
                      n_patients =n_patients,
                      FUP_tis = FUP_tis,
                      events = events,
                      wce_id_indexes = wce_id_selected,
                      max_time =max_time)


    if compress:
        return compress_dataset(dataset)

    return dataset

def compress_dataset(dataset):
    covariate_cols = dataset.columns[5:]
    shifted = dataset.groupby('patients')[covariate_cols].shift()
    dataset['any_change'] = (dataset[covariate_cols] != shifted).any(axis=1)
    dataset['group_id'] = dataset.groupby('patients')['any_change'].cumsum()
    dataset = dataset.drop(columns=['any_change'])
    return dataset.groupby(['patients', 'group_id']).agg({
        'start': 'first',
        'stop': 'last',
        'events': 'last',
        **{col: 'first' for col in covariate_cols}
    }).reset_index()


def simulate_dataset_batch( max_time, n_patients, list_covariates, batchsize = None, compress = True, seed: int | None = None):

    print(batchsize)
    print(n_patients)
    print(batchsize % n_patients)

    if batchsize is None:
        batchsize = n_patients

    if n_patients % batchsize != 0:
        msg = "The batch size must be proportional to the number of patients, in order to have a complete batch"
        raise ValueError(msg)

    n_batches = batchsize // n_patients

    dataset = simulate_dataset(max_time = max_time, n_patients = n_patients, list_covariates = list_covariates, compress = compress, seed=seed)


    for i in range(1,n_batches):

        current_seed = None if seed is None else seed + i
        batch_dataset = simulate_dataset(max_time = max_time, n_patients = n_patients, list_covariates = list_covariates, compress = compress, seed=current_seed)

        batch_dataset["patients"] += batchsize * i

        dataset = pd.concat((dataset, batch_dataset))

    return dataset















def simulate_for_experiment(n_patients, max_time,HR_target, scenario_name, seed: int | None = None):

    wce_covariate = WCECovariate(
        name = "dose",
        values = [1,1.5,2,2.5,3],
        scenario_name = scenario_name,
        HR_target = HR_target)


    return simulate_dataset(
        max_time = max_time,
        n_patients = n_patients,
        list_covariates = [wce_covariate],
        seed=seed)

def WCE_permalgo(n_patients,
                 max_time,
                 Xmat,
                 betas,
                 names,
                 wce_status: list[bool],
                 scenarios,
                #  eventRandom,
                #  censorRandom
                 ):


    list_covariates = []

    print(wce_status)

    print(Xmat.shape)

    for i in range(len(wce_status)):
        if wce_status[i]:
            wce_covariate_name = names[i]
            scenario_name = scenarios[i]
            HR_target = betas[i]
            Xvector = Xmat[:,i]
            wce_covariate = WCECovariate(name = wce_covariate_name,
                                         Xvector = Xvector,
                                         scenario_name = scenario_name,
                                         HR_target = HR_target)
            list_covariates.append(wce_covariate)

        elif not wce_status[i]:
            cox_covariate_name = names[i]
            coef = betas[i]
            cox_covariate = CoxCovariate(name = cox_covariate_name,
                                                   Xvector = Xvector,
                                                   coef = coef)
            list_covariates.append(cox_covariate)


    print(list_covariates)


    return simulate_dataset(max_time = max_time,
                               n_patients = n_patients,
                               list_covariates = list_covariates)


# def cox_benchmark_simualtion(n_patients, n_intervals, max_time,):
#     dataset = []

#     wce_covariate = WCECovariate(
#         name = "dose",
#         values = [1,1.5,2,2.5,3],
#         scenario_name = scenario_name,
#         HR_target = HR_target)





#     dataset = simulate_dataset(
#         max_time = max_time,
#         n_patients = n_patients,
#         list_covariates = [wce_covariate])


#     print(type(wce_covariate))

#     return dataset

####
def exponential_scenario(u_t):
    return(7 * np.exp(-7*u_t/365)) # divide by 365 in order to have a t in days

def bi_linear_scenario(u_t):
    if u_t < 50:
        return (1- (u_t/365)/(50/365))
    return 0

def early_peak_scenario(u_t):
    return norm.pdf(u_t/365, 0.04, 0.05)


def inverted_u_scenario(u_t):
    return norm.pdf(u_t/365, 0.2, 0.06)

def constant_scenario(u_t):
    if u_t <= 180:
        return 1/180
    else:
        return 0

def hat_scenario(u_t):
    if u_t < 180:
        return (u_t/180)
    elif (u_t >= 180) and (u_t < 240):
        return 1
    else:
        return (1 - (u_t-240)/180)






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
    except KeyError as err:
        msg = f"The scenario '{scenario_name}' is not defined"
        raise ValueError(msg) from err

    scenario_list = []
    normalization_factor = 0

    for i in range(max_time):
        normalization_factor += scenario_function(i)

    for i in range(max_time):
        scenario_list.append(scenario_function(i))

    scenario_list = np.array(scenario_list)

    return scenario_list / normalization_factor
