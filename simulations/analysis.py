
import csv
import torch 
import sys
import itertools
import argparse
import time


from pykeops.torch import LazyTensor




sys.path.append("../python")
from survivalgpu import use_cuda, device, float32, int32, int64
from survivalgpu.utils import numpy
from survivalgpu import wce_torch
from survivalgpu.wce_features import wce_features_batch


# the number of patients is defined in the data
# data description  "{n_patient}_{weight_function}"


# Simulation of Xmat

import json
import sys

# Read the JSON string from command line arguments
with open ("Simulation_results/simulation_parameters.json") as simulation_parameters_json:
    simulation_parameters = json.load(simulation_parameters_json)
# Convert JSON string back to a dictionary


# print(simulation_parameters)


experiment_name = simulation_parameters["experiment_name"]
print("###### Exeriment : ",experiment_name," #############")




n_patients_list = simulation_parameters["n_patients"]
weight_function_list = simulation_parameters["weight_function_list"]
n_bootstraps_list = simulation_parameters["n_bootstraps_list"]
nknots_list = simulation_parameters["nknots_list"]
cutoff_list = simulation_parameters["cutoff_list"]
constraint = simulation_parameters["constraint"]
# batchsizeS = [100] #not here

result_folder_path = "Simulation_results/" + experiment_name
experiment_dict_list = []



def WCE_experiment(n_patients, weight_function,n_bootsraps,nknots,cutoff,constraint,batchsize):
    starting_time = time.time()
    patient = []
    start = []
    stop = []
    events = []
    doses = []
    
    input_PATH = result_folder_path + "/WCEmat/" + weight_function + "_" + str(n_patients) +".csv"


    with open(input_PATH) as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            patient.append(float(row['patient']))
            start.append(float(row['start']))
            stop.append(float(row['stop']))
            events.append(float(row['event']))
            doses.append(float(row['dose']))

    patient = torch.tensor(patient, device = device, dtype = int32)
    start = torch.tensor(start, device = device, dtype = int32)
    stop = torch.tensor(stop, device = device, dtype = int32)
    events = torch.tensor(events, device = device, dtype = int32)
    doses = torch.tensor(doses, device = device, dtype = float32)


    result = wce_torch(ids = patient, doses = doses, events = events, times = start,
                          cutoff = cutoff, nknots = nknots,covariates = None, 
                          batchsize = batchsize, bootstrap = n_bootsraps, constrained = constraint,
                          verbosity = 0
                          )
    computation_time =  time.time() - starting_time

    return (result, computation_time)

def print_constrained(constraint):
    if constraint == None:
        return("None")
    return(constraint)







for (n_patients,weight_function,n_bootstraps,nknots, cutoff, constraint) in itertools.product(n_patients_list,weight_function_list, n_bootstraps_list,nknots_list,cutoff_list,constraint):
    # print("##### start experiment : ")

    path = result_folder_path + "/models/" + str(n_patients)+ "_" + weight_function + "_" + str(n_bootstraps) + "bootsraps" + str(nknots) +"knots" + "_" +print_constrained(constraint)
    

    print("n_patients = ", str(n_patients)," - weight_function =" , weight_function ," - n_bootstraps = " , 
          str(n_bootstraps), " - nknots =  ", str(nknots), " - cutoff = ", cutoff, "  - constraint = ",
          print_constrained(constraint))
    
    result, computation_time = WCE_experiment(n_patients,weight_function,n_bootstraps,nknots,cutoff,constraint,batchsize = 100)
    torch.save(result, path)

    experiment_dict = {
        "path" : path,
        "n_patients" : n_patients,
        "weight_function": weight_function,
        "n_bootstraps" : n_bootstraps,
        "nknots" : nknots,
        "cutoff" : cutoff,
        "constraint" : constraint,
        "computation_time" : computation_time
    }
    experiment_dict_list.append(experiment_dict)

# print(experiment_dict_list)

result_path = result_folder_path + "/" + experiment_name + ".csv"
print(result_path)

with open(result_path,"w", newline ='') as file:
    writer = csv.writer(file)
    fields = list(experiment_dict_list[0].keys())
    print(fields)
    writer.writerow(fields)

    for experiment_dict in experiment_dict_list:
        line = []
        for field in fields:
            line.append(experiment_dict[field])
        writer.writerow(line)
        

    


    #print("right constrained")
    #WCE_experiment(n_patients,weight_function,n_bootstraps,nknots,cutoff)


