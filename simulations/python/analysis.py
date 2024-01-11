import csv
import torch 
import sys


sys.path.append("../../python")
import survivalgpu
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

patient = torch.tensor(patient,dtype=torch.int32)
start = torch.tensor(start,dtype=torch.int32)
stop = torch.tensor(stop,dtype=torch.int32)
events = torch.tensor(events,dtype=torch.int32)
doses = torch.tensor(doses,dtype=torch.int32)


result = wce_torch(ids = patient, doses = doses, events = events, times = start,
                      cutoff = 180, nknots = 1,covariates = None)



print(result)





