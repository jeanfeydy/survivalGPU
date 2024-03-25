import numpy as np
import random
import pandas as pd
import taichi as ti

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



def generate_wce_mat(scenario_name, Xmat, cutoff):
    """
    This function generate the wce mattrix that keep the WCE wieght of all the patient at
    all the times intill the cutoff
    """
    scenario_shape = scenarios.get_scenario(scenario_name,cutoff)
    wce_mat = np.vstack([generate_wce_vector(u, scenario_shape, Xmat) for u in range(1,cutoff+1)])
    return wce_mat


# def get_dataset

# def get_dataset(Xmat, wce_mat,):

def matching(wce_mat: np.matrix):
    """
    This is the matr
    """
    n_patients = wce_mat.shape[1]
    print(n_patients)


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
    print(max_time* int(1/censoring_ratio))
    censorRandom = np.round(np.random.uniform(low = 1, high = max_time* int(1/censoring_ratio), size = n_patients)).astype(int)

    event = [1 if eventRandom[i]< censorRandom[i] else 0 for i in range(len(eventRandom))]
    FUP_Ti = np.minimum(eventRandom,censorRandom)


    return event, FUP_Ti


    # Censoring times : Uniform[1;730] for all scenarios

@ti.kernel
def gpu_matching(wce_mat_current:ti.types.ndarray() ,time_event: int ):

    proba =  0

    return proba

def cpu_matching(wce_mat_current,time_event, HR_target ):

    print(wce_mat_current)
    print()
    print(np.exp(wce_mat_current[time_event,]))
    print(time_event)

    print(wce_mat_current)

    probas = np.array(np.exp(HR_target * wce_mat_current[time_event,])/np.sum(np.exp(HR_target * wce_mat_current[time_event,])))

    return probas



def matching_algo(wce_mat, max_time:int, n_patients:int, HR_target):
#wce_mat:np.matrix,
    
    events, FUP_tis = event_censor_generation(max_time, n_patients, censoring_ratio=0.5)
    
    # print(events)
    # print(FUP_tis)

    ids = np.arange(0,n_patients, dtype = int)
    
    
    for i in range(n_patients):
        # print(i)
        # print(ids)
        # print(f"len ids : {len(ids)}")
        # print(f"len(events): {len(events)}")
        # print(f"i: {i}")

        event = events[i]
        time_event = FUP_tis[i]
        # print(event)

        #if event == 0:
        if event ==0:
            
            id_index = np.random.randint(0,len(ids))
            ids = np.delete(ids,id_index) 
            # wce_mat_current = np.array(wce_mat)[ids]
            # print(wce_mat_current.shape)
        else:
            # gpu_matching(wce_mat, time_event)
            # print(wce_mat)
            # print(ids)
            wce_mat_current = wce_mat[:,ids]
            print(wce_mat_current.shape)
            print(f"current wce_mat shape  : {wce_mat_current.shape}")
            print(wce_mat_current.shape)
            probas = cpu_matching(wce_mat_current, time_event,HR_target)

            print(probas)
            # id_index = np.random.randint(0,len(ids))
            # ids = np.delete(ids,id_index) 


            



        

        # print(ids)



        # print(len(ids))



    x = ti.field





    


    # df_event = pd.DataFrame(data,
    #                         columns = ["patient","eventRandom","censorRandom" ],
    #                         )    
    
    # print(df_event)

    # df_event["patient"] =  df_event["patient"].astype(int)
    # df_event['FUP_Ti'] = df_event[['eventRandom', 'censorRandom']].min(axis = 1).astype(int)
    # df_event['event'] = (df_event['FUP_Ti'] == df_event['eventRandom']).astype(int)
    # print(df_event)
    # # print(df_event.groupby('patient')[['eventRandom', 'censorRandom']].transform(min))
    # df_event = df_event.drop(["eventRandom","censorRandom"], axis = 1)
    # print(df_event.dtypes)
    # # print(df_event)
    # print(df_event)


    # # patient_order = df_event["patient"]

    # for i in patient_order:

    #     event = df_event["event"]
    #     print(event)










# print(Xmat, "\n")

# wce_vector = generate_wce_vector(u = 15,scenario= scenario, Xmat = Xmat, normalization_factor= 0.9)
# for u in range(1,20):
#     print(generate_wce_vector(u,scenario= scenario, Xmat = Xmat, normalization_factor= 0.9))

# print("\n")
     
n_patients = 10
max_time = 365
cutoff = 180


Xmat = generate_Xmat(max_time,n_patients,[1,2,3])

scenario= "exponential_scenario"

wce_mat = generate_wce_mat(scenario_name= scenario, Xmat = Xmat, cutoff = cutoff)

# print(wce_mat)
# print(wce_mat.shape)
# # # print(wce_mat)
# print(wce_mat)


# eventRandom, censorRandom = event_censor_generation(max_time,n_patients,0.5)

# print(eventRandom)
# print(censorRandom)


# 

print(type(wce_mat))




# ti.init(arch=ti.gpu)

print(f"initial wce_mat shape : {wce_mat.shape}")

matching_algo(wce_mat, max_time,n_patients, HR_target=1.5) # wce_mat