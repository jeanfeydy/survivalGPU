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



def WCE_experiment(n_patients, weight_function,n_bootsraps,nknots,cutoff):

    patient = []
    start = []
    stop = []
    events = []
    doses = []

    with open("../WCEmat_data/bi_linear_weight.csv") as file:
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
                          batchsize = 1, bootstrap = n_bootsraps,
                          verbosity = 0
                          )

    PATH = "../Simulation_results/{n_patients}_{weight_function}_{n_bootsraps}bootstraps_{nknots}knots"

    torch.save(result, PATH)


n_patients_list = [500,1000] #,5000,10000]
weight_function_list = ["early_peak_weight","bi_linear_weight"]
n_bootstraps_list = [1000]
nknots_list = [1,2,3]
cutoff_list = [180]


for (n_patients,weight_function,n_bootstraps,nknots, cutoff) in itertools.product(n_patients_list,weight_function_list, n_bootstraps_list,nknots_list,cutoff_list):
    WCE_experiment(n_patients,weight_function,n_bootstraps,nknots,cutoff)
