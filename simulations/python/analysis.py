
import csv
import torch 
import sys
import itertools
import argparse



from pykeops.torch import LazyTensor




sys.path.append("../../python")
from survivalgpu import use_cuda, device, float32, int32, int64
from survivalgpu.utils import numpy
from survivalgpu import wce_torch
from survivalgpu.wce_features import wce_features_batch


# the number of patients is defined in the data
# data description  "{n_patient}_{weight_function}"


print("Begining of the program")

parser = argparse.ArgumentParser(description='Name the experiment')
parser.add_argument('experiment_name', type=str, help='the name of the experiment')
args = parser.parse_args()
experiment_name = args.experiment_name
print("###### Exeriment : ",experiment_name," #############")



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

    return result

def print_constrained(constraint):
    if constraint == None:
        return("None")
    return(constraint)




n_patients_list = [50]#,1000,5000]#,10000]
weight_function_list = ["exponential_weight"] #,"bi_linear_weight","constant_weight","early_peak_weight","inverted_u_weight","late_effect_weight"]
n_bootstraps_list = [1000]#,1000]
nknots_list = [1,2,3]
cutoff_list = [180]
constraint = ["Right"]#[None, "Right"]
# batchsizeS = [100] #not here

result_folder_path = "../Simulation_results/" + experiment_name
experiment_dict_list = []


for (n_patients,weight_function,n_bootstraps,nknots, cutoff, constraint) in itertools.product(n_patients_list,weight_function_list, n_bootstraps_list,nknots_list,cutoff_list,constraint):
    # print("##### start experiment : ")

    path = result_folder_path + "/models/" + str(n_patients)+ "_" + weight_function + "_" + str(n_bootstraps) + "bootsraps" + str(nknots) +"knots" + "_" +print_constrained(constraint)
    
    experiment_dict = {
        "path" : path,
        "n_patients" : n_patients,
        "weight_function": weight_function,
        "n_bootstraps" : n_bootstraps,
        "nknots" : nknots,
        "cutoff" : cutoff,
        "constraint" : constraint
    }
    print("n_patients = ", str(n_patients)," - weight_function =" , weight_function ," - n_bootstraps = " , 
          str(n_bootstraps), " - nknots =  ", str(nknots), " - cutoff = ", cutoff, "  - constraint = ",
          print_constrained(constraint))
    
    result = WCE_experiment(n_patients,weight_function,n_bootstraps,nknots,cutoff,constraint,batchsize = 100)
    torch.save(result, path)

    experiment_dict_list.append(experiment_dict)

# print(experiment_dict_list)

result_path = result_folder_path + "/" + experiment_name + ".csv"

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


