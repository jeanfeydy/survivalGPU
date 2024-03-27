import numpy as np
import random
import pandas as pd
import taichi as ti
import taichi.math as tm

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


# def get_dataset

# def get_dataset(Xmat, wce_mat,):

def matching(wce_mat: np.matrix):
    """
    This is the matr
    """
    n_patients = wce_mat.shape[1]
    print(n_patients)


def parallilized_event_censor_generation(max_time, n_patients, censoring_ratio):
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

    print(event)
    print(FUP_Ti)

    times_event_0 = []
    times_event_1 = []

    for i in range(len(event)):
        if event[i] ==1:
            times_event_1.append(FUP_Ti[i])
        else:
            times_event_0.append(FUP_Ti[i])

        
    print()
    print()
    print(times_event_0)
    # event_data = FUP_Ti[]
    # print(event_data)


    return np.array(times_event_0), np.array(times_event_1)

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



    # event_data = FUP_Ti[]
    # print(event_data)


    return event, FUP_Ti


    # Censoring times : Uniform[1;730] for all scenarios






   

    
def cpu_matching(wce_mat_current,time_event, HR_target ):

    # print(wce_mat_current)
    # print()
    # print(np.exp(wce_mat_current[time_event,]))
    # print(time_event)

    # print(wce_mat_current)

   

    probas = np.array(np.exp(HR_target * wce_mat_current[time_event -1,])/np.sum(np.exp(HR_target * wce_mat_current[time_event -1,])))

    return probas.reshape(probas.shape[1])



def cpu_matching_algo(wce_mat, max_time:int, n_patients:int, HR_target):
    
    cpu_time = 0
    
    event_censor_start = time.perf_counter()
    events, FUP_tis = event_censor_generation(max_time, n_patients, censoring_ratio=0.5)
    event_censor_end = time.perf_counter()
    elapsed_event_censor = event_censor_end - event_censor_start

    wce_id_indexes = []
    

    ids = np.arange(0,n_patients, dtype = int)
    
    
    for i in range(n_patients):
        iteration_start = time.perf_counter()
        event = events[i]
        time_event = FUP_tis[i]

        if event ==0:
            
            id_index = np.random.randint(0,len(ids))
            wce_id = ids[id_index]
            ids = np.delete(ids,id_index) 

            

            elapsed_cpu_time = 0

        else:

            wce_mat_current = wce_mat[:,ids]

            
            cpu_start = time.perf_counter()
            probas = cpu_matching(wce_mat_current,time_event,HR_target)
            cpu_end = time.perf_counter()

            elapsed_cpu_time = cpu_end - cpu_start
            
            cpu_time += elapsed_cpu_time

       


            id_index = np.random.choice(np.arange(0,len(ids)), p = probas)
            wce_id = ids[id_index]
            ids = np.delete(ids,id_index) 

        wce_id_indexes.append(wce_id)

            

        
        iteration_end = time.perf_counter()
        elapsed_iteration = iteration_end - iteration_start 

    wce_id_indexes = np.array(wce_id_indexes)

    

    print(f"Time in cpu :{cpu_time}")

    

    


    return wce_id_indexes, events, FUP_tis


def get_dataset(Xmat, wce_mat, HR_target):

    max_time,n_patients = wce_mat.shape[0], wce_mat.shape[1]
    print(max_time)
    print(n_patients)
    wce_id_indexes, events, FUP_tis = cpu_matching_algo(wce_mat, max_time,n_patients, HR_target=1.5) # wce_mat
    
    df_wce = pd.DataFrame()
    df_Xmat = pd.DataFrame(Xmat)


    data_event = np.vstack([np.arange(n_patients),
                             events,
                             FUP_tis]).transpose()

    df_event = pd.DataFrame(data_event,
                             columns = ["patient_id","event","FUP_ti"])
    

    start_dataset_time = time.perf_counter()

    

    #### intialization matrix

    patient_index = 0



    fu_patient = df_event[df_event ["patient_id"] == patient_index]["FUP_ti"].item()
    event_patient = df_event[df_event ["patient_id"] == patient_index]["event"].item()

    if event_patient == 0:
        event_vec = np.append(np.zeros(fu_patient-1),1)
    else:
        event_vec = np.zeros(fu_patient)
    # print(event_vec)
        
    id_dose = wce_id_indexes[patient_index]

    data = Xmat[:fu_patient,id_dose].flatten()
    data_vector = np.array([data[:,i].item() for i in range(data.shape[1])])
    
    data_frame_matrix = np.vstack([np.repeat(patient_index,fu_patient),
                                  np.arange(fu_patient),
                                  np.arange(1,fu_patient+1),
                                  event_vec,
                                  data_vector]).transpose()
        
    
    for patient_index in range(1,len(wce_id_indexes)):
        fu_patient = df_event[df_event ["patient_id"] == patient_index]["FUP_ti"].item()
        event_patient = df_event[df_event ["patient_id"] == patient_index]["event"].item()

        if event_patient == 0:
            event_vec = np.append(np.zeros(fu_patient-1, dtype = int),1)
        else:
            event_vec = np.zeros(fu_patient)
        # print(event_vec)

        id_dose = wce_id_indexes[patient_index]

        # print("##############################", patient_index)

        # print(fu_patient)
        # print(id_dose)
        # print(np.repeat(patient_index,fu_patient).shape)
        # print(np.arange(fu_patient).shape)
        # print(np.arange(1,fu_patient+1).shape)
        # print(event_vec.shape)
        # print("here")
        data = Xmat[:fu_patient,id_dose].flatten()

        data_vector = np.array([data[:,i].item() for i in range(data.shape[1])])
        # print(data_vector.shape)

        new_data_frame_matrix = np.vstack([np.repeat(patient_index,fu_patient),
                                      np.arange(fu_patient),
                                      np.arange(1,fu_patient+1),
                                      event_vec,
                                      data_vector]
                                      ).transpose()
        

        data_frame_matrix = np.vstack([data_frame_matrix,new_data_frame_matrix])
        

 
        # print(dose)

        


    df_wce= pd.DataFrame(data_frame_matrix,
                            columns = ["patient","start","stop","event","dose"]
                            )
    
    print(df_wce)
    
        
        
    end_dataset_time = time.perf_counter()

    elapased_dataset_time = end_dataset_time -  start_dataset_time 
    print(f"Time in dataset : {elapased_dataset_time}")





    

    return df_wce



    # for patient_index in range(len(wce_id_indexes)):
    #     event_patient = 

def get_dataset_gpu(Xmat, wce_mat, HR_target):

    max_time,n_patients = wce_mat.shape[0], wce_mat.shape[1]
    print(max_time)
    print(n_patients)
    wce_id_indexes, events, FUP_tis = cpu_matching_algo(wce_mat, max_time,n_patients, HR_target=1.5) # wce_mat
    
    df_wce = pd.DataFrame()
    df_Xmat = pd.DataFrame(Xmat)

    # print(events)
    print(wce_id_indexes)
    ordered_events = np.array(events)[wce_id_indexes]
    ordered_FUP_tis = np.array(FUP_tis)[wce_id_indexes]
    # print(ordered_events)


    data_event = np.vstack([np.arange(n_patients),
                             events,
                             FUP_tis]).transpose()
    
    print(data_event)
    print("total_time : ", FUP_tis.sum())

    df_event = pd.DataFrame(data_event,
                             columns = ["patient_id","event","FUP_ti"])
    

    start_dataset_time = time.perf_counter()

    
    ti.init(arch = "gpu")

    data_field = ti.field(dtype=ti.i64, shape=(FUP_tis.sum(),5))
    print(data_field)

    Xmat_transposed = Xmat.transpose()

    # print(Xmat_transposed)



    times= np.arange(1,max_time+1)
    print(times)



    # @ti.func

    # id_and_times = ti.Vector(np.[list(range(n_patients)), list(range(max_time))])


    patient_time_field= ti.field(dtype=int, shape=(n_patients, max_time))

    print("###################")
    print(FUP_tis)
    print(wce_id_indexes)
    print("###################")

    print(wce_id_indexes.dtype)

    print(Xmat)
    print(Xmat[1,2])







    print(FUP_tis)
    # ["patient","start","stop","event","dose"]


    @ti.kernel
    def iteration_dataset(ordered_events:ti.types.ndarray() ,ordered_FUP_tis:ti.types.ndarray(), 
                          wce_id_indexes:ti.types.ndarray(), Xmat_transposed:ti.types.ndarray()
                          ):#, Xmat_transposed:ti.types.ndarray()):
        print("GPU KERNEL")

        line = 0 

        for patient_id, time in patient_time_field:
            # print(patient_index, time)
            # print(patient_id)
            # print(time)

            event = ordered_events[patient_id]
            b= ordered_FUP_tis[patient_id]
            


            if time < ordered_FUP_tis[patient_id]:

                patient = patient_id
                event = 0
                dose = Xmat_transposed[patient_id,time]
                time_start = time
                time_stop = time +1 

                data_field[line,0] = patient_id
                data_field[line,1] = 0
                data_field[line,2] = Xmat_transposed[patient_id,time]
                data_field[line,3] = time
                data_field[line,4] = time +1 
            
                line += 1 




            elif time == ordered_FUP_tis[patient_id]:
                patient = patient_id
                event = ordered_events[patient_id]
                dose = Xmat_transposed[patient_id,time]
                time_start = time
                time_stop = time +1 

                data_field[line,0] = patient_id
                data_field[line,1] = 0
                data_field[line,2] = Xmat_transposed[patient_id,time]
                data_field[line,3] = time
                data_field[line,4] = time +1 
            
                line += 1 

    iteration_dataset(ordered_events,ordered_FUP_tis,wce_id_indexes,Xmat_transposed)



    
    return data_field.to_numpy()
            
########################################################""
     
# n_patients = 100
# max_time = 365
# cutoff = 180



# Xmat = generate_Xmat(max_time,n_patients,[1,2,3])

# scenario= "exponential_scenario"

# wce_mat = generate_wce_mat(scenario_name= scenario, Xmat = Xmat, cutoff = cutoff, max_time= max_time)


# print(wce_mat)
# print(wce_mat.shape)
# # # print(wce_mat)
# print(wce_mat)


# eventRandom, censorRandom = event_censor_generation(max_time,n_patients,0.5)

# print(eventRandom)
# print(censorRandom)


# 

# print(type(wce_mat))



# ti.init(arch=ti.gpu)
# n_patients = 500
# max_time = 365
# cutoff = 180
# Xmat = generate_Xmat(max_time,n_patients,[1,2,3])
# scenario= "exponential_scenario"

# wce_mat = generate_wce_mat(scenario_name= scenario, Xmat = Xmat, cutoff = cutoff, max_time= max_time)
# gpu_time = matching_algo(wce_mat, max_time,n_patients, HR_target=1.5) # wce_mat



############## Real ones 

n_patients = 1000
max_time = 365
cutoff = 180

Xmat = generate_Xmat(max_time,n_patients,[1,2,3])

scenario= "exponential_scenario"


wce_mat = generate_wce_mat(scenario_name= scenario, Xmat = Xmat, cutoff = cutoff, max_time= max_time)
# ti.init(arch=ti.gpu)

# print(f"initial wce_mat shape : {wce_mat.shape}")

# start_gpu_time = time.perf_counter()

# gpu_time = matching_algo(wce_mat, max_time,n_patients, HR_target=1) # wce_mat

# end_gpu_time = time.perf_counter()

# elapsed_gpu_total_time = end_gpu_time - start_gpu_time 

# print(f"time in rest program : {elapsed_gpu_total_time - gpu_time}")
# print(f"total time gpu matching algo : {elapsed_gpu_total_time}")


############ CPU 

start_cpu_time = time.perf_counter()

# wce_id_indexes, events, FUP_tis = cpu_matching_algo(wce_mat, max_time,n_patients, HR_target=1.5) # wce_mat

numpy_wce = get_dataset_gpu(Xmat, wce_mat, 1.5)

end_cpu_time = time.perf_counter()

elapsed_cpu_total_time = end_cpu_time - start_cpu_time 

# print(f"time in rest program : {elapsed_cpu_total_time - cpu_time}")
print(f"total time cpu matching algo : {elapsed_cpu_total_time}")

df_wce = pd.DataFrame(numpy_wce, columns = ["patient","start","stop","event","dose"])

print(df_wce)




df_wce.to_csv("test_df")



# 5000 patients : 158 GPU

