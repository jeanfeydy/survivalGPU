import csv
import torch 
import sys


sys.path.append("../../python")
from survivalgpu import use_cuda, device, float32, int32, int64
from survivalgpu.utils import numpy
from survivalgpu import wce_torch


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

print(patient)

patient = torch.tensor(patient)
start = torch.tensor(start)
stop = torch.tensor(stop)
events = torch.tensor(events)
doses = torch.tensor(doses)


result = wce_torch(ids = patient, doses = doses, events = events, times = start,
                      cutoff = 180, nknots = 1,covariates = None)



print(result)





