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



def matching_algo(wce_mat, max_time:int, n_patients:int, HR_target ):
#wce_mat:np.matrix,
    
    gpu_time = 0
    
    events, FUP_tis = event_censor_generation(max_time, n_patients, censoring_ratio=0.5)
    
    # print(events)
    # print(FUP_tis)

    ids = np.arange(0,n_patients, dtype = int)
    
    
    for i in range(n_patients):
        iteration_start = time.perf_counter()
        elapsed_gpu_time = 0
        # print(i)
        # print(ids)
        # print(f"len ids : {len(ids)}")
        # print(f"len(events): {len(events)}")
        # print(f"i: {i}")
        iteration_start = time.perf_counter()

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

            
            # print(a)
            # print(wce_mat_current.shape)
            # print(f"current wce_mat shape  : {wce_mat_current.shape}")
            # print(wce_mat_current.shape)
            # probas = cpu_matching(wce_mat_current, time_event,HR_target)

            data_modif_start = time.perf_counter()
            wce_mat_current = wce_mat_current.astype(np.float32, copy=False)
            wce_mat_current_field = ti.field(float, wce_mat_current.shape)
            wce_mat_current_field.from_numpy(wce_mat_current)
            probas_gpu = ti.field(float,(wce_mat_current_field.shape[1]))

            data_modif_end = time.perf_counter()
            elapsed_data_modif_time = data_modif_end - data_modif_start
            print(f"data_modif time= {elapsed_data_modif_time}")
           
            @ti.kernel
            def gpu_matching(time_event: int, HR_target:float ) -> float:

                # sum_proba: ti.float64
                sum_proba = 0.0
                for i in range(wce_mat_current_field.shape[1]):
                    # print(tm.exp(HR_target * wce_mat_current_field[time_event,i]))
                    sum_proba += tm.exp(HR_target * wce_mat_current_field[time_event,i])       
                # print(sum_proba)   
                # print(1)     

                for i in range(wce_mat_current_field.shape[1]):
                    probas_gpu[i] = tm.exp(HR_target * wce_mat_current_field[time_event,i])/sum_proba
                return sum_proba
            
            gpu_start = time.perf_counter()
            sum_proba = gpu_matching(time_event,HR_target)
            gpu_end = time.perf_counter()

            elapsed_gpu_time = gpu_end - gpu_start
            # print(elapsed_gpu_time)
            print(elapsed_gpu_time)
            
            gpu_time += elapsed_gpu_time
            

           
            probas = probas_gpu.to_numpy()
            # print(f"GPU_sum : {sum_proba}")
            # probas = probas.reshape(len(ids))
        

            # print(probas)
            # print(probas.shape)
            # print(ids.shape)
       


            id_index = np.random.choice(np.arange(0,len(ids)), p = probas)
            # print("OK")
            ids = np.delete(ids,id_index) 
            # id_index = np.random.randint(0,len(ids))
            # ids = np.delete(ids,id_index) 

        iteration_end = time.perf_counter()
        elapsed_iteration = iteration_end - iteration_start 
        print(f"unparallelized iteration = {elapsed_iteration - elapsed_gpu_time}")
    print(f"Time in gpu :{gpu_time}")


    return gpu_time

def parallelized_matching_algo(wce_mat, max_time:int, n_patients:int, HR_target ):
    
    gpu_time = 0
    
    times_event_0, times_event_1 = parallilized_event_censor_generation(max_time, n_patients, censoring_ratio=0.5)



    
    # print(events)
    # print(FUP_tis)

    

    ids = np.arange(0,n_patients, dtype = int)


    # time_event_0 = [i if events[i] ==0 else 0: for i in range(len(events))]
    # print(time_event_0)

    
    
    for i in range(n_patients):
        iteration_start = time.perf_counter()
        elapsed_gpu_time = 0
        # print(i)
        # print(ids)
        # print(f"len ids : {len(ids)}")
        # print(f"len(events): {len(events)}")
        # print(f"i: {i}")
        iteration_start = time.perf_counter()

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

            
            # print(a)
            # print(wce_mat_current.shape)
            # print(f"current wce_mat shape  : {wce_mat_current.shape}")
            # print(wce_mat_current.shape)
            # probas = cpu_matching(wce_mat_current, time_event,HR_target)

            data_modif_start = time.perf_counter()
            wce_mat_current = wce_mat_current.astype(np.float32, copy=False)
            wce_mat_current_field = ti.field(float, wce_mat_current.shape)
            wce_mat_current_field.from_numpy(wce_mat_current)
            probas_gpu = ti.field(float,(wce_mat_current_field.shape[1]))

            data_modif_end = time.perf_counter()
            elapsed_data_modif_time = data_modif_end - data_modif_start
            print(f"data_modif time= {elapsed_data_modif_time}")
           
            @ti.kernel
            def gpu_matching(time_event: int, HR_target:float ) -> float:

                # sum_proba: ti.float64
                sum_proba = 0.0
                for i in range(wce_mat_current_field.shape[1]):
                    # print(tm.exp(HR_target * wce_mat_current_field[time_event,i]))
                    sum_proba += tm.exp(HR_target * wce_mat_current_field[time_event,i])       
                # print(sum_proba)   
                # print(1)     

                for i in range(wce_mat_current_field.shape[1]):
                    probas_gpu[i] = tm.exp(HR_target * wce_mat_current_field[time_event,i])/sum_proba
                return sum_proba
            
            gpu_start = time.perf_counter()
            sum_proba = gpu_matching(time_event,HR_target)
            gpu_end = time.perf_counter()

            elapsed_gpu_time = gpu_end - gpu_start
            # print(elapsed_gpu_time)
            print(elapsed_gpu_time)
            
            gpu_time += elapsed_gpu_time
            

           
            probas = probas_gpu.to_numpy()
            # print(f"GPU_sum : {sum_proba}")
            # probas = probas.reshape(len(ids))
        

            # print(probas)
            # print(probas.shape)
            # print(ids.shape)
       


            id_index = np.random.choice(np.arange(0,len(ids)), p = probas)
            # print("OK")
            ids = np.delete(ids,id_index) 
            # id_index = np.random.randint(0,len(ids))
            # ids = np.delete(ids,id_index) 

        iteration_end = time.perf_counter()
        elapsed_iteration = iteration_end - iteration_start 
        print(f"unparallelized iteration = {elapsed_iteration - elapsed_gpu_time}")
    print(f"Time in gpu :{gpu_time}")


    return gpu_time

def cpu_matching_algo(wce_mat, max_time:int, n_patients:int, HR_target):
#wce_mat:np.matrix,
    
    cpu_time = 0
    
    event_censor_start = time.perf_counter()
    events, FUP_tis = event_censor_generation(max_time, n_patients, censoring_ratio=0.5)
    event_censor_end = time.perf_counter()
    elapsed_event_censor = event_censor_end - event_censor_start
    # print(f"event_censor time : {elapsed_event_censor}")

    wce_id_indexes = []
    
    # print(events)
    # print(FUP_tis)

    ids = np.arange(0,n_patients, dtype = int)
    
    
    for i in range(n_patients):
        iteration_start = time.perf_counter()
        # if i%int(n_patients/20)== 0:
        #     print(i)
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

            elapsed_cpu_time = 0
            # wce_mat_current = np.array(wce_mat)[ids]
            # print(wce_mat_current.shape)
        else:
            # gpu_matching(wce_mat, time_event)
            # print(wce_mat)
            # print(ids)
            wce_mat_current = wce_mat[:,ids]

            
            # print(a)
            # print(wce_mat_current.shape)
            # print(f"current wce_mat shape  : {wce_mat_current.shape}")
            # print(wce_mat_current.shape)
            # probas = cpu_matching(wce_mat_current, time_event,HR_target)


            
            cpu_start = time.perf_counter()
            probas = cpu_matching(wce_mat_current,time_event,HR_target)
            cpu_end = time.perf_counter()

            elapsed_cpu_time = cpu_end - cpu_start
            # print(elapsed_cpu_time)
            
            cpu_time += elapsed_cpu_time
            # print(f"Time in cpu :{elapsed_cpu_time}")
            

           
            # print(f"GPU_sum : {sum_proba}")
            # probas = probas.reshape(len(ids))
        

            # print(probas)
            # print(probas.shape)
            # print(ids.shape)
       


            id_index = np.random.choice(np.arange(0,len(ids)), p = probas)
            # print("OK")
            ids = np.delete(ids,id_index) 
            # id_index = np.random.randint(0,len(ids))
            # ids = np.delete(ids,id_index) 
        wce_id_indexes.append(id_index)

            

        
        iteration_end = time.perf_counter()
        elapsed_iteration = iteration_end - iteration_start 
        # print(f"unparallelized iteration = {elapsed_iteration - elapsed_cpu_time}")

    

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

    start = True

    
    
    for patient_index in range(len(wce_id_indexes)):
        fu_patient = df_event[df_event ["patient_id"] == patient_index]["FUP_ti"].item()
        event_patient = df_event[df_event ["patient_id"] == patient_index]["event"].item()

        if event_patient == 0:
            event_vec = np.append(np.zeros(fu_patient-1),1)
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
        
        if start == True: 
            data_frame_matrix = new_data_frame_matrix
            start = False
        else: 
            data_frame_matrix = np.vstack([data_frame_matrix,new_data_frame_matrix])
        

 
        # print(dose)

        


    df_wce= pd.DataFrame(data_frame_matrix,
                            columns = ["patient","start","stop","event","dose"]
                            )
    
        
        
    end_dataset_time = time.perf_counter()

    elapased_dataset_time = end_dataset_time -  start_dataset_time 
    print(f"Time in dataset : {elapased_dataset_time}")





    

    return df_wce



    # for patient_index in range(len(wce_id_indexes)):
    #     event_patient = 



            
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

n_patients = 10000
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

df_wce = get_dataset(Xmat, wce_mat, 1.5)

end_cpu_time = time.perf_counter()

elapsed_cpu_total_time = end_cpu_time - start_cpu_time 

# print(f"time in rest program : {elapsed_cpu_total_time - cpu_time}")
print(f"total time cpu matching algo : {elapsed_cpu_total_time}")

df_wce.to_csv("test_df")



# 5000 patients : 158 GPU

