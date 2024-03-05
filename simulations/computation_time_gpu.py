
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


import json


def run_gpu(n_bootstraps,n_patients,weight_function,normalization):

    starting_time = time.time()
    patient = []
    start = []
    stop = []
    events = []
    doses = []
    
    input_PATH = "WCEmat/" + weight_function + "_" + str(normalization) + "_" + str(n_patients) +".csv"


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
    
    starting_time = time.time()

    if n_patients < 10000:
        batchsize = 100
    else:
        batchsize = 10

    result = wce_torch(ids = patient, doses = doses, events = events, times = start,
                          cutoff = 180, nknots = 1,covariates = None, 
                          batchsize = batchsize, bootstrap = n_bootstraps, constrained = "Right")
    computation_time =  time.time() - starting_time

    return computation_time


n_patients_list = [100,1000,10000,100000]
weight_function = "exponential_weight"
normalization = 1

for n_patients in n_patients_list:
    print("start computation n patient = ", n_patients)
    computation_time = run_gpu(n_bootstraps=1000,
                               n_patients=n_patients,
                               weight_function=weight_function,
                               normalization=normalization)
    print("Computation took : ", computation_time, " s")


