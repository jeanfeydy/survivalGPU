import csv
import torch 
import sys


sys.path.append("../../python")
from survivalgpu import use_cuda, device, float32, int32, int64
from survivalgpu.utils import numpy
from survivalgpu import wce_torch
from survivalgpu.wce_features import wce_features_batch



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

print("OK 1")
patient = torch.tensor(patient, dtype = int32)
start = torch.tensor(start, dtype = int32)
stop = torch.tensor(stop, dtype = int32)
events = torch.tensor(events, dtype = int32)
doses = torch.tensor(doses, dtype = float32)

print("OK 2")

wce_features_batch( ids = patient, times = start, doses = doses, nknots = 1, cutoff = 180, 
                   order=3, knots=None)

result = wce_torch(ids = patient, doses = doses, events = events, times = start,
                      cutoff = 180, nknots = 1,covariates = None)

print("OK 3")


print(result)





