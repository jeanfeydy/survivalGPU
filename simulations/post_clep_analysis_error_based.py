import csv
import torch 
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
from statistics import mean
from pykeops.torch import LazyTensor
import numpy as np
import argparse


import json
import sys


sys.path.append("../python")
from survivalgpu import use_cuda, device, float32, int32, int64
from survivalgpu.utils import numpy
from survivalgpu import wce_torch
from survivalgpu.wce_features import wce_features_batch

# parser = argparse.ArgumentParser(description='Name the experiment')
# parser.add_argument('experiment_name', type=str, help='the name of the experiment')
# args = parser.parse_args()
# experiment_name = args.experiment_name

# Read the JSON string from command line arguments
simulation_variables_str = sys.argv[1]

# Convert JSON string back to a dictionary
simulation_variables = json.loads(simulation_variables_str)

# print(simulation_variables)


experiment_name = simulation_variables["experiment_name"]

path =  "Simulation_results/"+ experiment_name  + "/models/"
# print(path)


list_dir = os.listdir(path)
# print(list_dir)

weight_function = "exponential_weight"


# experiements_df = pd.read_csv("../Simulation_results/"+ experiment_name +"/"+experiment_name  +".csv")




def analyse_result(file_path):

    data = torch.load(file_path, map_location=torch.device('cpu'))


    weight_function_path = "weight_functions_shapes/" + weight_function + ".csv"




    with open (weight_function_path) as file:
        real_weights = []
        csvreader = csv.reader(file)
        next(csvreader)
        for row in csvreader:
            real_weights.append(float(row[0]))


    real_weights = torch.tensor(real_weights)
    mean_tensor = data["WCEmat"].mean(dim=0)

    mean_tensor = data["WCEmat"].mean(dim=0)
    std_tensor = data["WCEmat"].std(dim=0)

    list_sum_weights_diff = []

    list_AUC_diff = []

    for i in range(data["WCEmat"].shape[0]):
        sum_weights_diff = abs(data["WCEmat"][i] - real_weights)
        list_sum_weights_diff.append(sum_weights_diff.sum().item())

        AUC = np.trapz(data["WCEmat"][i],list(range(data["WCEmat"][i].shape[0])))
        # print(data["WCEmat"][i].shape)
        AUC_real = np.trapz(real_weights,list(range(real_weights.shape[0])))
        #print(sum_air_real)
        #print(sum_air) ### Ai ai ai sensé être à 1
        AUC_diff = abs(AUC_real -AUC)
        list_AUC_diff.append(AUC_diff)



    return(np.array(list_sum_weights_diff),np.array(list_AUC_diff))

with open ("Simulation_results/"+ experiment_name +"/"+experiment_name  +".csv") as file:
    experiment_list = []
    csv_reader = csv.DictReader(file)
    for experient in csv_reader:
        (list_sum_weights_diff, list_AUC_diff) =  analyse_result(experient['path'])

        # lower_bar_WD = 
        # print("weights_diff ",list_sum_weights_diff.mean())

        experient["sum_weight_diff_mean"] = list_sum_weights_diff.mean()
        experient["sum_weight_diff_95p"] = np.percentile(list_sum_weights_diff,95)
        experient["sum_weight_diff_5p"] = np.percentile(list_sum_weights_diff,5)

        experient["AUC_diff_mean"] = list_AUC_diff.mean()
        experient["AUC_diff_95p"] = np.percentile(list_AUC_diff,95)
        experient["AUC_diff_5p"] = np.percentile(list_AUC_diff,5)

        experiment_list.append(experient)


result_path = "Simulation_results/"+ experiment_name +"/analyzed_"+experiment_name  + ".csv"
print(result_path)

with open(result_path,"w", newline ='') as file:
    writer = csv.writer(file)
    fields = list(experiment_list[0].keys())
    print(fields)
    writer.writerow(fields)

    for experiment_dict in experiment_list:
        line = []
        for field in fields:
            line.append(experiment_dict[field])
        writer.writerow(line)
        



    # problem of cutoff, will show better eman for bigger cutoff as the curve will 
    # have no differnce when reach 0 
    # get the mean of each datapoint difference for each bootstraps 
    # do the 95% confidence interval 
        
    # Second method 
    # do tha air behind the curve 
    
    # air = 0
    # for i in range(data["WCEmat"].shape[0]):
    # this one is NOT affected by a bigger 


    # L'air sous la courba a une plus grosse différence que la moyenne des différences, peut être du à l'effet 
    # de la queue de la courbe 
    # quels sont les méthdoes statistiques ? 

 



  

    

    

    # print(mean_tensor)
    # print(mean_tensor + std_tensor)
    # print(mean_tensor - std_tensor)

    
    # plt.plot(mean_tensor.tolist(),color = "black", linewidth =0.85)
    # plt.plot((mean_tensor + std_tensor).tolist(), color = "black", linestyle = "dashed", linewidth =0.85)
    # plt.plot((mean_tensor - std_tensor).tolist(), color = "black",linestyle = "dashed", linewidth =0.85)
    # plt.plot(rows,color = "red", linewidth =1)
    # plt.title(file_name)
    # plt.xlabel("time (days)")
    # plt.ylabel("weight")
    # figure_path = path + "/results/" + file_name + ".png"
    # plt.savefig(figure_path)

    # print (figure_path)
    # plt.clf()
