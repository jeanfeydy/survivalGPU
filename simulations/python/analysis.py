
import csv
import torch 
import sys
import itertools


from pykeops.torch import LazyTensor




sys.path.append("../../python")
from survivalgpu import use_cuda, device, float32, int32, int64
from survivalgpu.utils import numpy
from survivalgpu import wce_torch
from survivalgpu.wce_features import wce_features_batch


# the number of patients is defined in the data
# data description  "{n_patient}_{weight_function}"


print("Begining of the program")



def WCE_experiment(n_patients, weight_function,n_bootsraps,nknots,cutoff,constraint,batchsize):

    patient = []
    start = []
    stop = []
    events = []
    doses = []
    
    input_PATH = "../WCEmat_data/" + weight_function + "_" + str(n_patients) +".csv"


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

    PATH = "../Simulation_results/16-01-2023/" + str(n_patients)+ "_" + weight_function + "_" + str(n_bootsraps) + "bootsraps" + str(nknots) +"knots" + "_" +print_constrained(constraint) +"_" + str(batchsize)  +"batchsize"

    torch.save(result, PATH)


def print_constrained(constraint):
    if constraint == None:
        return("None")
    return(constraint)
    



n_patients_list = [500,1000,5000]#,10000]
weight_function_list = ["exponential_weight"] #,"bi_linear_weight","constant_weight","early_peak_weight","inverted_u_weight","late_effect_weight"]
n_bootstraps_list = [10000]#,1000]
nknots_list = [1,2,3]
cutoff_list = [180]
constraint = ["Right"]#[None, "Right"]
batchsizeS = [100] #not here


print("variables defined")

for (n_patients,weight_function,n_bootstraps,nknots, cutoff, constraint, batchsize) in itertools.product(n_patients_list,weight_function_list, n_bootstraps_list,nknots_list,cutoff_list,constraint,batchsizeS):
    print("##### start experiment : ")


    print("n_patients = ", str(n_patients)," - weight_function =" , weight_function ," - n_bootstraps = " , 
          str(n_bootstraps), " - nknots =  ", str(nknots), " - cutoff = ", cutoff, "  - constraint = ",
          print_constrained(constraint)," - batchsize = ",batchsize)
    WCE_experiment(n_patients,weight_function,n_bootstraps,nknots,cutoff,constraint,batchsize)
    #print("right constrained")
    #WCE_experiment(n_patients,weight_function,n_bootstraps,nknots,cutoff)
